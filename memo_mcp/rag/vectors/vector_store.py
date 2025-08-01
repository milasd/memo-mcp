import logging
from typing import List, Dict, Any, Union
import numpy as np

from memo_mcp.rag.vectors.database.vector_backend import VectorDatabase

from memo_mcp.rag.config import RAGConfig, DocumentMetadata


class VectorStore:
    """
    Vector storage interface with multiple backend implementations.

    Supports FAISS, Chroma, Qdrant, and simple in-memory storage.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backend: Union[
            FAISSBackend, ChromaBackend, QdrantBackend, SimpleBackend
        ] = None

        from memo_mcp.rag.vectors.database.faiss import FAISSBackend
        from memo_mcp.rag.vectors.database.chromadb import ChromaBackend
        from memo_mcp.rag.vectors.database.qdrant import QdrantBackend
        from memo_mcp.rag.vectors.database.simple import SimpleBackend

        backend_type = config.vector_store_type.lower()
        if backend_type == "faiss":
            self.backend = FAISSBackend(config)
        elif backend_type == "chroma":
            self.backend = ChromaBackend(config)
        elif backend_type == "qdrant":
            self.backend = QdrantBackend(config)
        else:
            self.backend = SimpleBackend(config)

    async def initialize(self) -> None:
        """Initialize the vector store backend."""
        await self.backend.initialize()

    async def add_documents(
        self,
        embeddings: List[np.ndarray],
        texts: List[str],
        metadatas: List[DocumentMetadata],
    ) -> None:
        """Add document embeddings to the store."""
        await self.backend.add_documents(embeddings, texts, metadatas)

    def search(
        self, query_embedding: np.ndarray, top_k: int, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        return self.backend.search(query_embedding, top_k, similarity_threshold)

    async def remove_document(self, file_path: str) -> bool:
        """Remove all chunks of a document."""
        return await self.backend.remove_document(file_path)

    def get_document_count(self) -> int:
        """Get the number of unique documents."""
        return self.backend.get_document_count()

    def get_chunk_count(self) -> int:
        """Get the total number of chunks."""
        return self.backend.get_chunk_count()

    def is_empty(self) -> bool:
        """Check if the vector store is empty."""
        return self.backend.is_empty()

    async def clear(self) -> None:
        """Clear all data from the vector store."""
        await self.backend.clear()

    async def close(self) -> None:
        """Close the vector store and cleanup resources."""
        if self.backend:
            await self.backend.close()

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the vector store."""
        return self.backend.health_check()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistical information about the vector store."""
        return self.backend.get_stats()

    def get_backend(self) -> VectorDatabase:
        """
        Get direct access to the underlying backend for advanced operations.

        Use this when you need backend-specific features:

        Example:
            store = VectorStore(config)
            if isinstance(store.get_backend(), QdrantBackend):
                results = await store.get_backend().search_with_filter(...)
        """
        return self.backend

    def get_backend_type(self) -> str:
        """Get the type of the current backend."""
        return self.backend.__class__.__name__


def create_vector_store(config: RAGConfig) -> VectorStore:
    """
    Factory function to create vector store instances.

    Args:
        config: RAG configuration

    Returns:
        VectorStore instance
    """
    return VectorStore(config)
