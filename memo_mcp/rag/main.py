import asyncio
from pathlib import Path

from memo_mcp.rag import RAGConfig, create_rag_system
from memo_mcp.utils.logging_setup import set_logger

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"


"""Sample script to run RAG system locally without MCP."""

SAMPLE_DATA_DIR = Path("data/memo_example")
VECTOR_DB = "chroma"  # you can try "faiss", "chroma", "simple".


async def run_rag() -> None:
    config = RAGConfig(
        vector_store_type=VECTOR_DB,
        data_root=SAMPLE_DATA_DIR,
        use_gpu=True,
        cache_embeddings=True,
    )

    logger = set_logger()
    rag = await create_rag_system(config, logger)
    try:
        stats = await rag.get_stats()

        if stats["total_documents"] == 0:
            rag.logger.info("Building index for the first time...")
            await rag.build_index()
        else:
            rag.logger.info(
                f"Using existing index with {stats['total_documents']} documents"
            )

        results = await rag.query("feelings about work this year")
        rag.logger.info(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            similarity_score = result.get("similarity_score", 0.0)
            combined_score = result.get("combined_score", similarity_score)
            metadata = result["metadata"]
            rag.logger.info(
                f"\n{i}. {metadata.file_name} ({metadata.date_created})"
                f"\n   Similarity Score: {similarity_score:.3f}"
                f"\n   Combined Score: {combined_score:.3f}"
                f"\n   Preview: {result['text'][:100]}...\n"
            )

    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(run_rag())
