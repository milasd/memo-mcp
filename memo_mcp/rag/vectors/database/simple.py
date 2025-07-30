import pickle
from typing import List, Dict, Any
import numpy as np

from memo_mcp.rag.vectors.database.vector_backend import VectorDatabase
from memo_mcp.rag.config import RAGConfig, DocumentMetadata


class SimpleBackend(VectorDatabase):
    """Simple in-memory vector store backend for development and testing."""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadatas: List[DocumentMetadata] = []
        
        self.data_path = config.vector_store_path / "simple_store.pkl"
    
    async def initialize(self) -> None:
        """Initialize simple backend."""
        self.config.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if available
        if self.data_path.exists():
            await self._load_data()
        
        self.logger.info(f"Simple backend initialized with {len(self.texts)} documents")
    
    async def _load_data(self) -> None:
        """Load data from pickle file."""
        try:
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
                self.embeddings = data.get("embeddings", [])
                self.texts = data.get("texts", [])
                self.metadatas = data.get("metadatas", [])
        except Exception as e:
            self.logger.warning(f"Failed to load simple store data: {e}")
            self.embeddings = []
            self.texts = []
            self.metadatas = []
    
    async def _save_data(self) -> None:
        """Save data to pickle file."""
        try:
            data = {
                "embeddings": self.embeddings,
                "texts": self.texts,
                "metadatas": self.metadatas
            }
            with open(self.data_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Failed to save simple store data: {e}")
    
    async def add_documents(
        self, 
        embeddings: List[np.ndarray], 
        texts: List[str], 
        metadatas: List[DocumentMetadata]
    ) -> None:
        """Add documents to simple store."""
        self.embeddings.extend(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        await self._save_data()
        self.logger.debug(f"Added {len(embeddings)} documents to simple store")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search using cosine similarity."""
        if not self.embeddings:
            return []
        
        # Calculate cosine similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            # Cosine similarity
            dot_product = np.dot(query_embedding, emb)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            similarity = dot_product / max(norm_product, 1e-8)
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Format results
        results = []
        for similarity, idx in similarities[:top_k]:
            if similarity < similarity_threshold:
                continue
            
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "similarity_score": float(similarity)
            })
        
        return results
    
    async def remove_document(self, file_path: str) -> bool:
        """Remove document from simple store."""
        indices_to_remove = [
            i for i, meta in enumerate(self.metadatas) 
            if meta.file_path == file_path
        ]
        
        if not indices_to_remove:
            return False
        
        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            del self.embeddings[idx]
            del self.texts[idx]
            del self.metadatas[idx]
        
        await self._save_data()
        return True
    
    def get_document_count(self) -> int:
        """Get number of unique documents."""
        unique_files = set(meta.file_path for meta in self.metadatas)
        return len(unique_files)
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        return len(self.texts)
    
    def is_empty(self) -> bool:
        """Check if store is empty."""
        return len(self.texts) == 0
    
    async def clear(self) -> None:
        """Clear all data."""
        self.embeddings = []
        self.texts = []
        self.metadatas = []
        await self._save_data()
    
    async def close(self) -> None:
        """Close simple backend."""
        await self._save_data()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform simple backend health check."""
        try:
            return {
                "status": "healthy",
                "backend_type": "SimpleBackend",
                "chunk_count": len(self.texts),
                "document_count": self.get_document_count(),
                "is_empty": len(self.texts) == 0,
                "storage_path": str(self.data_path),
                "storage_exists": self.data_path.exists()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend_type": "SimpleBackend",
                "error": str(e)
            }