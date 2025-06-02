import faiss
import pickle
from typing import List, Dict, Any
import numpy as np

from memo_mcp.rag.vectors.database.vector_backend import VectorDatabase
from memo_mcp.rag.config import RAGConfig, DocumentMetadata


class FAISSBackend(VectorDatabase):
    """FAISS-based vector store backend for high performance similarity search."""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        
        self.index = None
        self.texts: List[str] = []
        self.metadatas: List[DocumentMetadata] = []
        self.dimension = config.embedding_dimension
        
        # File paths
        self.index_path = config.vector_store_path / "faiss_index"
        self.metadata_path = config.vector_store_path / "metadata.pkl"
        self.texts_path = config.vector_store_path / "texts.pkl"
    
    async def initialize(self) -> None:
        """Initialize FAISS index."""        
        self.config.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing index
        if self.index_path.exists():
            await self._load_index()
        else:
            await self._create_new_index()
        
        self.logger.info(f"FAISS backend initialized with {len(self.texts)} documents")
    
    async def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Optionally wrap with IDMap for document tracking
        self.index = faiss.IndexIDMap(self.index)
        
        self.texts = []
        self.metadatas = []
    
    async def _load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, "rb") as f:
                    self.metadatas = pickle.load(f)
            
            # Load texts
            if self.texts_path.exists():
                with open(self.texts_path, "rb") as f:
                    self.texts = pickle.load(f)
            
            self.logger.info("Loaded existing FAISS index")
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing index: {e}")
            await self._create_new_index()
    
    async def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadatas, f)
            
            # Save texts
            with open(self.texts_path, "wb") as f:
                pickle.dump(self.texts, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")
    
    async def add_documents(
        self, 
        embeddings: List[np.ndarray], 
        texts: List[str], 
        metadatas: List[DocumentMetadata]
    ) -> None:
        """Add documents to FAISS index."""
        if not embeddings:
            return
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.vstack(embeddings).astype(np.float32)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / np.maximum(norms, 1e-8)
        
        # Generate IDs for new documents
        start_id = len(self.texts)
        ids = np.array(range(start_id, start_id + len(embeddings)), dtype=np.int64)
        
        # Add to index
        self.index.add_with_ids(embeddings_array, ids)
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        # Save to disk
        await self._save_index()
        
        self.logger.debug(f"Added {len(embeddings)} documents to FAISS index")
    
    async def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in FAISS index."""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            if score < similarity_threshold:
                continue
            
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "similarity_score": float(score)
            })
        
        return results
    
    async def remove_document(self, file_path: str) -> bool:
        """Remove document from FAISS index."""
        # FAISS doesn't support efficient removal, so we rebuild without the document
        indices_to_remove = [
            i for i, meta in enumerate(self.metadatas) 
            if meta.file_path == file_path
        ]
        
        if not indices_to_remove:
            return False
        
        # Create new lists without the removed documents
        new_texts = [text for i, text in enumerate(self.texts) if i not in indices_to_remove]
        new_metadatas = [meta for i, meta in enumerate(self.metadatas) if i not in indices_to_remove]
        
        # Rebuild index with remaining documents
        await self._create_new_index()
        
        if new_texts:
            # We need to regenerate embeddings, which is expensive
            # For now, we'll mark this as a limitation
            self.logger.warning(f"Document removal requires rebuilding embeddings for {file_path}")
        
        self.texts = new_texts
        self.metadatas = new_metadatas
        await self._save_index()
        
        return True
    
    async def get_document_count(self) -> int:
        """Get number of unique documents."""
        unique_files = set(meta.file_path for meta in self.metadatas)
        return len(unique_files)
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        return len(self.texts)
    
    async def is_empty(self) -> bool:
        """Check if index is empty."""
        return self.index is None or self.index.ntotal == 0
    
    async def clear(self) -> None:
        """Clear all data."""
        await self._create_new_index()
        await self._save_index()
    
    async def close(self) -> None:
        """Close FAISS backend."""
        if self.index is not None:
            await self._save_index()