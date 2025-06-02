from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import date

from memo_mcp.rag.vectors.embeddings import EmbeddingManager
from memo_mcp.rag.vectors.vector_store import VectorStore
from memo_mcp.rag.document.indexer import DocumentIndexer
from memo_mcp.rag.document.retriever import DocumentRetriever
from memo_mcp.rag.config import RAGConfig
from memo_mcp.rag.app_logging import setup_logging


class MemoRAG:
    """
    Main RAG system for Memo MCP.
    
    Handles document indexing, embedding generation, and retrieval
    for daily journal entries stored in hierarchical date structure.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the RAG system with configuration."""
        self.config = config or RAGConfig()
        self.logger = setup_logging(self.config.log_level)
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(self.config)
        self.vector_store = VectorStore(self.config)
        self.indexer = DocumentIndexer(self.config, self.embedding_manager, self.vector_store)
        self.retriever = DocumentRetriever(self.config, self.embedding_manager, self.vector_store)
        
        # Track initialization state
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the RAG system components."""
        if self._initialized:
            return
            
        self.logger.info("Initializing Memo RAG system...")
        
        try:
            # Initialize embedding manager (loads model, checks GPU)
            await self.embedding_manager.initialize()
            
            # Initialize vector store
            await self.vector_store.initialize()
            
            self._initialized = True
            self.logger.info("Memo RAG system initialized successfully")
            
            # Note: We don't automatically build index here anymore
            # User should call build_index() explicitly if needed
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    async def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build or rebuild the document index.
        
        Args:
            force_rebuild: Whether to rebuild even if index exists
            
        Returns:
            Dictionary with indexing statistics
        """
        # Ensure system is initialized first
        if not self._initialized:
            await self.initialize()
            
        self.logger.info("Starting document indexing...")
        
        try:
            if force_rebuild:
                await self.vector_store.clear()
            
            stats = await self.indexer.index_documents()
            
            self.logger.info(f"Indexing completed. Stats: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")
            raise
    
    async def query(
        self, 
        query_text: str, 
        top_k: int = None,
        date_filter: Optional[Tuple[date, date]] = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system for relevant documents.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            date_filter: Optional date range filter (start_date, end_date)
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with metadata
        """
        # Ensure system is initialized
        if not self._initialized:
            await self.initialize()
            
        # Check if index exists and build if needed
        if await self.vector_store.is_empty():
            self.logger.info("Vector store is empty, building index...")
            await self.build_index()
            
        top_k = top_k or self.config.default_top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        self.logger.debug(f"Querying with: '{query_text}' (top_k={top_k})")
        
        try:
            results = await self.retriever.retrieve(
                query_text=query_text,
                top_k=top_k,
                date_filter=date_filter,
                similarity_threshold=similarity_threshold
            )
            
            self.logger.debug(f"Retrieved {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise
    
    async def add_document(
        self, 
        file_path: Path, 
        force_reindex: bool = False
    ) -> bool:
        """
        Add or update a single document in the index.
        
        Args:
            file_path: Path to the document
            force_reindex: Whether to reindex even if document exists
            
        Returns:
            True if document was added/updated
        """
        # Ensure system is initialized
        if not self._initialized:
            await self.initialize()
            
        try:
            success = await self.indexer.index_single_document(
                file_path, 
                force_reindex
            )
            
            if success:
                self.logger.info(f"Successfully indexed document: {file_path}")
            else:
                self.logger.debug(f"Document already up-to-date: {file_path}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to add document {file_path}: {e}")
            raise
    
    async def remove_document(self, file_path: Path) -> bool:
        """
        Remove a document from the index.
        
        Args:
            file_path: Path to the document to remove
            
        Returns:
            True if document was removed
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            success = await self.vector_store.remove_document(str(file_path))
            
            if success:
                self.logger.info(f"Successfully removed document: {file_path}")
            else:
                self.logger.debug(f"Document not found in index: {file_path}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to remove document {file_path}: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        # Ensure system is initialized
        if not self._initialized:
            await self.initialize()
            
        return {
            "total_documents": await self.vector_store.get_document_count(),
            "total_chunks": await self.vector_store.get_chunk_count(),
            "embedding_model": self.embedding_manager.model_name,
            "device": self.embedding_manager.device,
            "vector_dimension": self.embedding_manager.embedding_dimension,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the RAG system."""
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # Test embedding generation
            test_embedding = await self.embedding_manager.embed_text("test")
            
            # Test vector store
            stats = await self.get_stats()
            
            return {
                "status": "healthy",
                "healthy": True,
                "embedding_test": len(test_embedding) > 0,
                "stats": stats
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }
    
    async def close(self) -> None:
        """Clean up resources."""
        self.logger.info("Shutting down Memo RAG system...")
        
        if hasattr(self, 'vector_store'):
            await self.vector_store.close()
        
        if hasattr(self, 'embedding_manager'):
            await self.embedding_manager.close()
            
        self._initialized = False
        self.logger.info("Memo RAG system shut down complete")
        import gc
        gc.collect() # Force garbage collection after closing components


# Convenience functions for easy usage
async def create_rag_system(config: Optional[RAGConfig] = None) -> MemoRAG:
    """Create and initialize a RAG system."""
    rag = MemoRAG(config)
    await rag.initialize()
    return rag


async def quick_query(query_text: str, **kwargs) -> List[Dict[str, Any]]:
    """Quick query function for simple use cases."""
    rag = await create_rag_system()
    try:
        return await rag.query(query_text, **kwargs)
    finally:
        await rag.close()
