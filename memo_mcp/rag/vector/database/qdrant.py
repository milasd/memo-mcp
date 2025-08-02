import uuid
from typing import Any

import numpy as np

from memo_mcp.rag.config.rag_config import DocumentMetadata, RAGConfig
from memo_mcp.rag.vector.database.vector_backend import VectorDatabase


class QdrantBackend(VectorDatabase):
    """
    Qdrant-based vector store backend for vector search.

    Features:
    - Local and cloud Qdrant support
    - Advanced filtering with native Qdrant filters
    - Batch operations for efficiency
    - Payload indexing for faster queries
    - GRPC and REST API support
    """

    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.client = None
        self.collection_name = config.qdrant_collection_name

        # Track initialization state
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Qdrant client and collection."""
        if self._initialized:
            return

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import CreateCollection, Distance, VectorParams
        except ImportError:
            raise ImportError(
                "Qdrant client not installed. Install with: pip install qdrant-client"
            )

        self.logger.info(
            f"Initializing Qdrant backend at {self.config.qdrant_host}:{self.config.qdrant_port}"
        )

        # Initialize client based on configuration
        try:
            if self.config.qdrant_api_key:
                # Cloud/remote Qdrant with authentication
                self.client = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                    api_key=self.config.qdrant_api_key,
                    https=self.config.qdrant_use_https,
                    prefer_grpc=self.config.qdrant_prefer_grpc,
                )
                self.logger.info("Connected to Qdrant Cloud with API key")
            else:
                # Local Qdrant
                self.client = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                    prefer_grpc=self.config.qdrant_prefer_grpc,
                )
                self.logger.info("Connected to local Qdrant instance")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(
                f"Cannot connect to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}"
            )

        # Create or verify collection
        await self._setup_collection()

        # Create initial payload indexes for better performance
        await self._create_default_indexes()

        self._initialized = True
        self.logger.info("Qdrant backend initialized successfully")

    async def _setup_collection(self) -> None:
        """Set up the document collection."""
        from qdrant_client.models import Distance, VectorParams

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimension, distance=Distance.COSINE
                    ),
                )
                self.logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                # Verify existing collection configuration
                collection_info = self.client.get_collection(self.collection_name)
                if (
                    collection_info.config.params.vectors.size
                    != self.config.embedding_dimension
                ):
                    raise ValueError(
                        f"Existing collection has dimension {collection_info.config.params.vectors.size}, "
                        f"but config specifies {self.config.embedding_dimension}"
                    )
                self.logger.info(
                    f"Using existing Qdrant collection: {self.collection_name}"
                )

        except Exception as e:
            self.logger.error(f"Failed to setup Qdrant collection: {e}")
            raise

    async def _create_default_indexes(self) -> None:
        """Create default payload indexes for common query patterns."""
        indexes_to_create = [
            "file_path",  # For file-based filtering
            "date_created",  # For date range filtering
            "file_name",  # For filename-based queries
        ]

        for field_name in indexes_to_create:
            try:
                await self.create_payload_index(field_name)
            except Exception as e:
                # Indexes might already exist, log as debug
                self.logger.debug(f"Could not create index for {field_name}: {e}")

    async def add_documents(
        self,
        embeddings: list[np.ndarray],
        texts: list[str],
        metadatas: list[DocumentMetadata],
    ) -> None:
        """Add documents to Qdrant collection."""
        if not embeddings:
            return

        from qdrant_client.models import PointStruct

        points = []
        for embedding, text, metadata in zip(
            embeddings, texts, metadatas, strict=False
        ):
            # Create unique point ID
            point_id = str(uuid.uuid4())

            # Prepare payload with comprehensive metadata
            payload = {
                "text": text,
                "file_path": metadata.file_path,
                "file_name": metadata.file_name,
                "date_created": metadata.date_created,
                "date_modified": metadata.date_modified,
                "file_size": metadata.file_size,
                "chunk_index": metadata.chunk_index,
                "total_chunks": metadata.total_chunks,
                "content_preview": metadata.content_preview[:500],  # Limit preview size
                # Additional searchable fields
                "text_length": len(text),
                "word_count": len(text.split()),
                # Extract year/month for easier filtering
                "year": metadata.date_created[:4]
                if len(metadata.date_created) >= 4
                else "",
                "month": metadata.date_created[:7]
                if len(metadata.date_created) >= 7
                else "",
            }

            points.append(
                PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)
            )

        # Upload points in optimized batches
        batch_size = min(100, max(10, len(points) // 10))  # Adaptive batch size

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
            except Exception as e:
                self.logger.error(f"Failed to upload batch {i // batch_size + 1}: {e}")
                raise

        self.logger.debug(
            f"Added {len(points)} documents to Qdrant in {len(range(0, len(points), batch_size))} batches"
        )

    async def search(
        self, query_embedding: np.ndarray, top_k: int, similarity_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """Search Qdrant collection for similar vectors."""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=similarity_threshold,
            )

            results = []
            for hit in search_result:
                payload = hit.payload

                # Reconstruct DocumentMetadata
                metadata = DocumentMetadata(
                    file_path=payload["file_path"],
                    file_name=payload["file_name"],
                    date_created=payload["date_created"],
                    date_modified=payload["date_modified"],
                    file_size=payload["file_size"],
                    chunk_index=payload["chunk_index"],
                    total_chunks=payload["total_chunks"],
                    content_preview=payload.get("content_preview", ""),
                )

                results.append(
                    {
                        "text": payload["text"],
                        "metadata": metadata,
                        "similarity_score": float(hit.score),
                        "point_id": hit.id,
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    async def remove_document(self, file_path: str) -> bool:
        """Remove all chunks of a document from Qdrant."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            # Create filter for the specific file
            filter_condition = Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path))
                ]
            )

            # Delete points matching the filter
            result = self.client.delete(
                collection_name=self.collection_name, points_selector=filter_condition
            )

            success = result.operation_id is not None
            if success:
                self.logger.debug(f"Removed document: {file_path}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to remove document {file_path}: {e}")
            return False

    async def get_document_count(self) -> int:
        """Get number of unique documents."""
        try:
            # Use scroll to get all unique file paths
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=["file_path"],
            )

            unique_files = set()
            for point in scroll_result[0]:
                if "file_path" in point.payload:
                    unique_files.add(point.payload["file_path"])

            return len(unique_files)

        except Exception as e:
            self.logger.warning(f"Failed to count unique documents: {e}")
            return 0

    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count or 0
        except Exception as e:
            self.logger.warning(f"Failed to get chunk count: {e}")
            return 0

    async def is_empty(self) -> bool:
        """Check if collection is empty."""
        count = await self.get_chunk_count()
        return count == 0

    async def clear(self) -> None:
        """Clear all data from collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)

            # Recreate collection
            await self._setup_collection()
            await self._create_default_indexes()

            self.logger.info(f"Cleared Qdrant collection: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to clear Qdrant collection: {e}")
            raise

    async def close(self) -> None:
        """Close Qdrant client and cleanup resources."""
        if self.client:
            try:
                self.client.close()
                self.logger.debug("Qdrant client closed")
            except Exception as e:
                self.logger.warning(f"Error closing Qdrant client: {e}")

        self._initialized = False

    # Qdrant-specific methods
    async def search_with_filter(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        date_range: tuple[str, str] | None = None,
        file_pattern: str | None = None,
        year_filter: str | None = None,
        month_filter: str | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Advanced search with Qdrant's native filtering capabilities.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            date_range: Date range tuple (start_date, end_date) in ISO format
            file_pattern: File path pattern for filtering
            year_filter: Filter by specific year (e.g., "2025")
            month_filter: Filter by specific month (e.g., "2025-06")
            similarity_threshold: Minimum similarity score

        Returns:
            List of filtered search results
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        conditions = []

        # Date range filter
        if date_range:
            start_date, end_date = date_range
            conditions.append(
                FieldCondition(
                    key="date_created", range=Range(gte=start_date, lte=end_date)
                )
            )

        # Year filter (more efficient than date range for year-based queries)
        if year_filter:
            conditions.append(
                FieldCondition(key="year", match=MatchValue(value=year_filter))
            )

        # Month filter (e.g., "2025-06" for June 2025)
        if month_filter:
            conditions.append(
                FieldCondition(key="month", match=MatchValue(value=month_filter))
            )

        # File pattern filter
        if file_pattern:
            conditions.append(
                FieldCondition(key="file_path", match=MatchValue(value=file_pattern))
            )

        # Create filter
        search_filter = Filter(must=conditions) if conditions else None

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=top_k,
                score_threshold=similarity_threshold,
            )

            results = []
            for hit in search_result:
                payload = hit.payload

                metadata = DocumentMetadata(
                    file_path=payload["file_path"],
                    file_name=payload["file_name"],
                    date_created=payload["date_created"],
                    date_modified=payload["date_modified"],
                    file_size=payload["file_size"],
                    chunk_index=payload["chunk_index"],
                    total_chunks=payload["total_chunks"],
                    content_preview=payload.get("content_preview", ""),
                )

                results.append(
                    {
                        "text": payload["text"],
                        "metadata": metadata,
                        "similarity_score": float(hit.score),
                        "point_id": hit.id,
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Filtered search failed: {e}")
            raise

    async def create_payload_index(
        self, field_name: str, field_type: str = "keyword"
    ) -> None:
        """Create an index on a payload field for faster filtering."""
        try:
            from qdrant_client.models import PayloadSchemaType

            # Map field types
            type_mapping = {
                "keyword": PayloadSchemaType.KEYWORD,
                "integer": PayloadSchemaType.INTEGER,
                "float": PayloadSchemaType.FLOAT,
                "bool": PayloadSchemaType.BOOL,
            }

            schema_type = type_mapping.get(field_type, PayloadSchemaType.KEYWORD)

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

            self.logger.info(
                f"Created payload index on field: {field_name} ({field_type})"
            )

        except Exception as e:
            # Index might already exist
            self.logger.debug(f"Could not create payload index for {field_name}: {e}")

    async def get_collection_info(self) -> dict[str, Any]:
        """Get detailed information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "status": collection_info.status,
            }

        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {}

    async def optimize_collection(self) -> bool:
        """Optimize the collection for better performance."""
        try:
            # Trigger collection optimization
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config={
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000,
                },
            )

            self.logger.info(
                f"Triggered optimization for collection: {self.collection_name}"
            )
            return True

        except Exception as e:
            self.logger.warning(f"Failed to optimize collection: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check of the Qdrant backend."""
        try:
            # Collection-specific checks
            collection_info = await self.get_collection_info()

            return {
                "status": "healthy",
                "connected": True,
                "collection_exists": bool(collection_info),
                "points_count": collection_info.get("points_count", 0),
                "host": self.config.qdrant_host,
                "port": self.config.qdrant_port,
                "collection_name": self.collection_name,
                "backend_type": "QdrantBackend",
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "host": self.config.qdrant_host,
                "port": self.config.qdrant_port,
                "backend_type": "QdrantBackend",
            }
