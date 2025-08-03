from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from chromadb import Collection
from chromadb.api.client import ClientAPI

from memo_mcp.rag.config.rag_config import DocumentMetadata, RAGConfig
from memo_mcp.rag.vector.database.vector_backend import VectorDatabase

# Error message constant
_CLIENT_NOT_INITIALIZED_ERROR = (
    "ChromaDB client not initialized. Call initialize() first."
)


class ChromaBackend(VectorDatabase):
    """ChromaDB-based vector store backend."""

    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.client: ClientAPI | None = None
        self.collection: Collection | None = None

    async def initialize(self) -> None:
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            ) from e

        # Initialize client
        self.client = chromadb.PersistentClient(
            path=str(self.config.vector_store_path), settings=Settings(allow_reset=True)
        )

        if self.client:
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="memo_documents", metadata={"hnsw:space": "cosine"}
            )

        self.logger.info("ChromaDB backend initialized")

    async def add_documents(
        self,
        embeddings: list[np.ndarray],
        texts: list[str],
        metadatas: list[DocumentMetadata],
    ) -> None:
        """Add documents to ChromaDB."""
        if not embeddings:
            return

        # Convert embeddings to list format
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Convert metadata to dict format
        metadatas_dict: list[Mapping[str, str | int | float | bool | None]] = []
        for meta in metadatas:
            meta_dict: Mapping[str, str | int | float | bool] = {
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

        if self.collection is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED_ERROR)

        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas_dict,
            ids=ids,
        )

        self.logger.debug(f"Added {len(embeddings)} documents to ChromaDB")

    def search(
        self, query_embedding: np.ndarray, top_k: int, similarity_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """Search ChromaDB collection."""
        if self.collection is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED_ERROR)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )

        formatted_results = []
        if (
            results
            and results["documents"]
            and results["metadatas"]
            and results["distances"]
        ):
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                strict=False,
            ):
                if metadata is None or distance is None:
                    continue  # type: ignore[unreachable]

                # Convert distance to similarity score (ChromaDB returns distances)
                similarity_score = 1.0 - distance

                if similarity_score < similarity_threshold:
                    continue

                # Reconstruct DocumentMetadata
                doc_metadata = DocumentMetadata(
                    file_path=metadata.get("file_path", ""),  # type: ignore
                    file_name=metadata.get("file_name", ""),  # type: ignore
                    date_created=metadata.get("date_created", ""),  # type: ignore
                    date_modified=metadata.get("date_modified", ""),  # type: ignore
                    file_size=metadata.get("file_size", 0),  # type: ignore
                    chunk_index=metadata.get("chunk_index", 0),  # type: ignore
                    total_chunks=metadata.get("total_chunks", 0),  # type: ignore
                    content_preview=metadata.get("content_preview", ""),  # type: ignore
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
        if self.collection is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED_ERROR)

        # Get all chunks for this document
        results = self.collection.get(where={"file_path": file_path})

        if not results["ids"]:
            return False

        # Delete all chunks
        self.collection.delete(ids=results["ids"])
        return True

    def get_document_count(self) -> int:
        """Get number of unique documents."""
        if self.collection is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED_ERROR)

        # This is approximate since we need to query all metadata
        all_results = self.collection.get()
        if not all_results["metadatas"]:
            return 0

        unique_files = {
            meta["file_path"]
            for meta in all_results["metadatas"]
            if meta and "file_path" in meta
        }
        return len(unique_files)

    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        if self.collection is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED_ERROR)
        return self.collection.count()

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        if self.collection is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED_ERROR)
        return self.collection.count() == 0

    async def clear(self) -> None:
        """Clear all data."""
        if self.client is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED_ERROR)
        if self.collection is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED_ERROR)

        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="memo_documents", metadata={"hnsw:space": "cosine"}
        )

    async def close(self) -> None:
        """Close connection and clean up resources."""
        self.client = None
        self.collection = None
        self.logger.info("ChromaDB client cleared.")

    def health_check(self) -> dict[str, Any]:
        """Perform ChromaDB-specific health check."""
        if self.collection is None:
            return {
                "status": "unhealthy",
                "backend_type": "ChromaBackend",
                "error": _CLIENT_NOT_INITIALIZED_ERROR,
            }
        try:
            # Test basic connectivity
            count = self.collection.count()
            all_results = self.collection.get()
            doc_count = 0
            if all_results["metadatas"]:
                unique_files = {
                    meta["file_path"]
                    for meta in all_results["metadatas"]
                    if meta and "file_path" in meta
                }
                doc_count = len(unique_files)

            return {
                "status": "healthy",
                "backend_type": "ChromaBackend",
                "chunk_count": count,
                "document_count": doc_count,
                "is_empty": count == 0,
                "collection_name": self.collection.name,
                "storage_path": str(self.config.vector_store_path),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend_type": "ChromaBackend",
                "error": str(e),
            }
