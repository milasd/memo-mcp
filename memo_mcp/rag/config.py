import os
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""

    # Paths
    data_root: Path = field(default_factory=lambda: Path("data/memo"))
    index_path: Path = field(default_factory=lambda: Path("index"))

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    use_gpu: bool = True
    batch_size: int = 32

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50

    # Retrieval settings
    default_top_k: int = 365
    similarity_threshold: float = 0.3

    # Vector store settings
    vector_store_type: str = "chroma"  # faiss, chroma, qdrant, or simple
    persist_embeddings: bool = True

    # File processing
    supported_extensions: List[str] = field(default_factory=lambda: [".md", ".txt"])
    encoding: str = "utf-8"

    # Qdrant-specific settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "memo_documents"
    qdrant_use_https: bool = False
    qdrant_api_key: Optional[str] = None
    qdrant_prefer_grpc: bool = True

    # Performance settings
    max_concurrent_files: int = 10
    cache_embeddings: bool = True

    # Logging
    log_level: int = logging.INFO

    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure paths are Path objects
        self.data_root = Path(self.data_root)
        self.index_path = Path(f"{self.index_path}/{self.vector_store_type}")

        # Create directories if they don't exist
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Adjust settings based on environment
        self._adjust_for_environment()

        self.device = self.get_device_preference()
        print(self.device)

    def _adjust_for_environment(self):
        """Adjust configuration based on available resources."""
        # Check if we're in a resource-constrained environment
        if os.getenv("MEMO_RAG_LITE", "false").lower() == "true":
            self.batch_size = min(self.batch_size, 8)
            self.max_concurrent_files = min(self.max_concurrent_files, 4)
            self.chunk_size = min(self.chunk_size, 256)

        # Override with environment variables if set
        if model := os.getenv("MEMO_EMBEDDING_MODEL"):
            self.embedding_model = model

        if gpu_str := os.getenv("MEMO_USE_GPU"):
            self.use_gpu = gpu_str.lower() in ("true", "1", "yes")

        if data_root := os.getenv("MEMO_DATA_ROOT"):
            self.data_root = Path(data_root)

        if qdrant_host := os.getenv("MEMO_QDRANT_HOST"):
            self.qdrant_host = qdrant_host

        if qdrant_port := os.getenv("MEMO_QDRANT_PORT"):
            self.qdrant_port = int(qdrant_port)

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        cache_path = self.index_path / "cache"
        cache_path.mkdir(exist_ok=True)
        return cache_path

    @property
    def vector_store_path(self) -> Path:
        """Get the vector store path."""
        return self.index_path / "vectors"

    @property
    def embeddings_cache_path(self) -> Path:
        """Get the embeddings cache path."""
        return self.cache_dir / "embeddings.pkl"

    def get_device_preference(self) -> str:
        """Get the preferred device for computations."""
        if not self.use_gpu:
            return "cpu"

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "data_root": str(self.data_root),
            "index_path": str(self.index_path),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "default_top_k": None,
            # "default_top_k": self.default_top_k,
            "similarity_threshold": self.similarity_threshold,
            "vector_store_type": self.vector_store_type,
            "device": self.device,
        }


@dataclass
class DocumentMetadata:
    """Metadata for a document in the RAG system."""

    file_path: str
    file_name: str
    date_created: str  # ISO format date
    date_modified: str  # ISO format date
    file_size: int
    chunk_index: int
    total_chunks: int
    content_preview: str = ""

    @classmethod
    def from_file_path(
        cls, file_path: Path, chunk_index: int = 0, total_chunks: int = 1
    ) -> "DocumentMetadata":
        """Create metadata from a file path."""
        stat = file_path.stat()

        # Extract date from path structure (data/[...]/YYYY/MM/DD.md)
        parts = file_path.parts
        if len(parts) >= 5:
            year, month, day_file = parts[-3], parts[-2], parts[-1]
            day = day_file.split(".")[0]
            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        else:
            # Fallback to file modification time
            date_str = datetime.fromtimestamp(stat.st_mtime).date().isoformat()

        # Format modification time
        modified_str = datetime.fromtimestamp(stat.st_mtime).isoformat()

        return cls(
            file_path=str(file_path),
            file_name=file_path.name,
            date_created=date_str,
            date_modified=modified_str,
            file_size=stat.st_size,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )


def load_config_from_env() -> RAGConfig:
    """Load configuration with environment variable overrides."""
    return RAGConfig()
