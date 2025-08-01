from typing import List, Dict, Any
import numpy as np

from memo_mcp.rag.vectors.database.vector_backend import VectorDatabase
from memo_mcp.rag.config import RAGConfig, DocumentMetadata


class ChromaBackend(VectorDatabase):
    """ChromaDB-based vector store backend."""

    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.client = None
        self.collection = None

    async def initialize(self) -> None:
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

        # Initialize client
        self.client = chromadb.PersistentClient(
            path=str(self.config.vector_store_path), settings=Settings(allow_reset=True)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="memo_documents", metadata={"hnsw:space": "cosine"}
        )

        self.logger.info("ChromaDB backend initialized")

    async def add_documents(
        self,
        embeddings: List[np.ndarray],
        texts: List[str],
        metadatas: List[DocumentMetadata],
    ) -> None:
        """Add documents to ChromaDB."""
        if not embeddings:
            return

        # Convert embeddings to list format
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Convert metadata to dict format
        metadatas_dict = []
        for meta in metadatas:
            meta_dict = {
                "file_path": meta.file_path,
                "file_name": meta.file_name,
                "date_created": meta.date_created,
                "date_modified": meta.date_modified,
                "file_size": meta.file_size,
                "chunk_index": meta.chunk_index,
                "total_chunks": meta.total_chunks,
                "content_preview": meta.content_preview[:100],  # Limit preview length
            }
            metadatas_dict.append(meta_dict)

        # Generate IDs
        ids = [f"{meta.file_path}_{meta.chunk_index}" for meta in metadatas]

        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas_dict,
            ids=ids,
        )

        # Ensure data is persisted (ChromaDB should auto-persist with PersistentClient)
        try:
            self.client.persist()
        except AttributeError:
            # persist() method may not be available in all ChromaDB versions
            pass

        self.logger.debug(f"Added {len(embeddings)} documents to ChromaDB")

    def search(
        self, query_embedding: np.ndarray, top_k: int, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB collection."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )

        formatted_results = []
        if results["documents"]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Convert distance to similarity score (ChromaDB returns distances)
                similarity_score = 1.0 - distance

                if similarity_score < similarity_threshold:
                    continue

                # Reconstruct DocumentMetadata
                doc_metadata = DocumentMetadata(
                    file_path=metadata["file_path"],
                    file_name=metadata["file_name"],
                    date_created=metadata["date_created"],
                    date_modified=metadata["date_modified"],
                    file_size=metadata["file_size"],
                    chunk_index=metadata["chunk_index"],
                    total_chunks=metadata["total_chunks"],
                    content_preview=metadata.get("content_preview", ""),
                )

                formatted_results.append(
                    {
                        "text": doc,
                        "metadata": doc_metadata,
                        "similarity_score": similarity_score,
                    }
                )

        return formatted_results

    async def remove_document(self, file_path: str) -> bool:
        """Remove document from ChromaDB."""
        # Get all chunks for this document
        results = self.collection.get(where={"file_path": file_path})

        if not results["ids"]:
            return False

        # Delete all chunks
        self.collection.delete(ids=results["ids"])
        return True

    def get_document_count(self) -> int:
        """Get number of unique documents."""
        # This is approximate since we need to query all metadata
        all_results = self.collection.get()
        if not all_results["metadatas"]:
            return 0

        unique_files = set(meta["file_path"] for meta in all_results["metadatas"])
        return len(unique_files)

    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        return self.collection.count()

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return self.collection.count() == 0

    async def clear(self) -> None:
        """Clear all data."""
        self.client.delete_collection("memo_documents")
        self.collection = self.client.create_collection(
            name="memo_documents", metadata={"hnsw:space": "cosine"}
        )

    async def close(self) -> None:
        """Close connection and clean up resources."""
        if self.client:
            # ChromaDB's PersistentClient doesn't have an explicit close() method
            # but setting client to None might help with resource cleanup
            # and allow its internal __del__ to run.
            del self.client
            self.client = None
            self.logger.info("ChromaDB client explicitly cleared.")

    def health_check(self) -> Dict[str, Any]:
        """Perform ChromaDB-specific health check."""
        try:
            # Test basic connectivity
            count = self.collection.count()
            doc_count = self.get_document_count()

            return {
                "status": "healthy",
                "backend_type": "ChromaBackend",
                "chunk_count": count,
                "document_count": doc_count,
                "is_empty": count == 0,
                "collection_name": "memo_documents",
                "storage_path": str(self.config.vector_store_path),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend_type": "ChromaBackend",
                "error": str(e),
            }
