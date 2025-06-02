import asyncio
from pathlib import Path

from memo_mcp.rag import RAGConfig, create_rag_system

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"


async def run_rag():
    config = RAGConfig(
        vector_store_type="chroma",
        data_root=Path("data/memo"),
        use_gpu=True,
        cache_embeddings=True
    )
    
    # Create RAG system
    rag = await create_rag_system(config)
    
    try:
        stats = await rag.get_stats()
        
        if stats["total_documents"] == 0:
            rag.logger.info("Building index for the first time...")
            await rag.build_index()
        else:
            rag.logger.info(f"Using existing index with {stats['total_documents']} documents")
        
        results = await rag.query("thoughts on LLMs", top_k=3)
        rag.logger.info(f"\nFound {len(results)} results:")
        for result in results:
            rag.logger.info(f"\n\n- {result['metadata'].file_name}: {result['text'][:100]}...\n")
    
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(run_rag())