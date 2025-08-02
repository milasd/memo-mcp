from .config.rag_config import DocumentMetadata, RAGConfig, load_config_from_env
from .document.indexer import DocumentIndexer, FileWatcher, TextProcessor
from .document.retriever import DocumentRetriever, QueryExpander, ResultAggregator
from .memo_rag import MemoRAG, create_rag_system, quick_query
from .vector.embeddings import EmbeddingManager
from .vector.vector_store import VectorStore

# Main exports
__all__ = [
    # Main classes
    "MemoRAG",
    "RAGConfig",
    "DocumentMetadata",
    # Core components
    "EmbeddingManager",
    "VectorStore",
    "DocumentIndexer",
    "DocumentRetriever",
    # Utilities
    "TextProcessor",
    "QueryExpander",
    "ResultAggregator",
    "FileWatcher",
    # Convenience functions
    "create_rag_system",
    "quick_query",
    "load_config_from_env",
]

# Package metadata
SUPPORTED_FORMATS = [".md", ".txt"]
SUPPORTED_EMBEDDINGS = [
    "sentence-transformers/all-MiniLM-L6-v2",
]
SUPPORTED_VECTOR_STORES = ["faiss", "chroma", "qdrant", "simple"]
