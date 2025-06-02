from .config import DocumentMetadata, RAGConfig, load_config_from_env
from .vectors.embeddings import EmbeddingManager
from .vectors.vector_store import VectorStore
from .document.indexer import DocumentIndexer, TextProcessor, FileWatcher
from .document.retriever import DocumentRetriever, QueryExpander, ResultAggregator
from .memo_rag import MemoRAG, create_rag_system, quick_query

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