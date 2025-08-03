from datetime import date, timedelta
from typing import Any

import mcp.types as types

from memo_mcp.config import DATA_DIR, TOP_K
from memo_mcp.rag import MemoRAG, RAGConfig, create_rag_system
from memo_mcp.utils.logging_setup import set_logger

"""
MCP tool handlers for memo operations.

This module contains the business logic for all MCP tools,
keeping the server.py focused on protocol handling.
"""

# Global RAG system instance
_rag_system: MemoRAG | None


def parse_date_filter_string(date_filter_str: str) -> tuple | None:
    """Convert string date filter to date range tuple."""
    if not date_filter_str:
        return None

    try:
        if len(date_filter_str) == 4:  # "2025"
            start_date = date(int(date_filter_str), 1, 1)
            end_date = date(int(date_filter_str), 12, 31)
        elif len(date_filter_str) == 7:  # "2025-01"
            year, month = map(int, date_filter_str.split("-"))
            start_date = date(year, month, 1)
            if month == 12:
                end_date = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(year, month + 1, 1) - timedelta(days=1)
        elif len(date_filter_str) == 10:  # "2025-01-15"
            year, month, day = map(int, date_filter_str.split("-"))
            start_date = end_date = date(year, month, day)
        else:
            raise ValueError(f"Invalid date filter format: {date_filter_str}")
        return (start_date, end_date)
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid date filter '{date_filter_str}': {e}")
        return None


async def get_rag_system() -> MemoRAG:
    """Get or initialize the global RAG system."""
    global _rag_system

    if _rag_system is None:
        logger = set_logger()
        logger.info("Initializing RAG system for MCP server...")

        # Set RAG parameters - can be customized via environment variables
        config = RAGConfig(
            vector_store_type="chroma",  # or "faiss"
            data_root=DATA_DIR,
            use_gpu=True,
            cache_embeddings=True,
            chunk_size=512,
            default_top_k=TOP_K,
            similarity_threshold=0.3,
        )

        _rag_system = await create_rag_system(config, logger)

        # Ensure index is built
        stats = await _rag_system.get_stats()
        if stats["total_documents"] == 0:
            logger.info("Building RAG index...")
            await _rag_system.build_index()

        logger.info(f"RAG system ready with {stats['total_documents']} documents")

    return _rag_system


async def cleanup_rag_system() -> None:
    """Clean up the RAG system on shutdown."""
    global _rag_system
    if _rag_system:
        await _rag_system.close()
        _rag_system = None


async def handle_add_memo(arguments: dict | None) -> list[types.TextContent]:
    """Handle the add-memo tool."""
    if not arguments:
        raise ValueError("Missing arguments")

    content = arguments.get("content")
    if not content:
        raise ValueError("Missing content")

    # Get date (use today if not provided)
    entry_date = arguments.get("date")
    if entry_date:
        try:
            # Parse the provided date
            year, month, day = entry_date.split("-")
            parsed_date = date(int(year), int(month), int(day))
        except (ValueError, TypeError) as e:
            raise ValueError("Invalid date format. Use YYYY-MM-DD format.") from e
    else:
        # Use today's date
        parsed_date = date.today()

    # Create the directory structure
    year_str = str(parsed_date.year)
    month_str = f"{parsed_date.month:02d}"
    day_str = f"{parsed_date.day:02d}"

    memo_dir = DATA_DIR / year_str / month_str
    memo_dir.mkdir(parents=True, exist_ok=True)

    # Create the file path
    memo_file = memo_dir / f"{day_str}.md"

    # Check if file exists and handle append/overwrite
    if memo_file.exists():
        # Append to existing file
        with open(memo_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n---\n\n{content}")
        action = "appended to"
    else:
        # Create new file
        with open(memo_file, "w", encoding="utf-8") as f:
            f.write(content)
        action = "created"

    # Get RAG system and index the new/updated document
    try:
        rag = await get_rag_system()
        await rag.add_document(memo_file, force_reindex=True)
    except Exception as e:
        # Log error but don't fail the memo creation
        print(f"Warning: Failed to index new memo: {e}")

    return [
        types.TextContent(
            type="text",
            text=f"Successfully {action} memo entry for {parsed_date.isoformat()} at {memo_file}",
        )
    ]


async def handle_search_journal(arguments: dict | None) -> list[types.TextContent]:
    """Handle the search-journal tool."""
    if not arguments:
        raise ValueError("Missing arguments")

    query = arguments.get("query")
    if not query:
        raise ValueError("Missing query parameter")

    top_k: int = arguments.get("top_k", TOP_K)
    date_filter_str: str | None = arguments.get("date_filter")
    date_filter = parse_date_filter_string(date_filter_str) if date_filter_str else None

    # Get RAG system
    rag = await get_rag_system()

    # Perform search with proper date filter
    results = await rag.query(query, top_k=top_k, date_filter=date_filter)

    # Format response
    if not results:
        response_text = f"No results found for query: '{query}'"
        if date_filter_str:
            response_text += f" (filtered by date: {date_filter_str})"
    else:
        response_parts = [
            f"Found {len(results)} results for: '{query}'",
            f"Date filter: {date_filter_str}" if date_filter_str else "",
            "",
            "Results:",
        ]

        for i, result in enumerate(results, 1):
            score = result.get("score", 0.0)
            metadata = result["metadata"]
            text_preview = (
                result["text"][:300] + "..."
                if len(result["text"]) > 300
                else result["text"]
            )

            result_text = f"""
{i}. {metadata.file_name} ({metadata.date_created})
   Score: {score:.3f}
   Preview: {text_preview}
   """
            response_parts.append(result_text.strip())

        response_text = "\n".join(filter(None, response_parts))

    return [
        types.TextContent(
            type="text",
            text=response_text,
        )
    ]


async def handle_get_journal_stats(arguments: dict | None) -> list[types.TextContent]:
    """Handle the get-journal-stats tool."""
    try:
        rag: MemoRAG = await get_rag_system()
        stats: dict[str, Any] = await rag.get_stats()

        response_text = f"""
Memo Data Stats:
• Total Documents: {stats["total_documents"]}
• Total Chunks: {stats["total_chunks"]}
• Embedding Model: {stats["embedding_model"]}
• Device: {stats["device"]}
• Vector Dimension: {stats["vector_dimension"]}
• Chunk Size: {stats["chunk_size"]}
• Vector Store: {stats.get("vector_store_type", "Unknown")}
        """.strip()

    except Exception as e:
        response_text = f"Error getting stats: {str(e)}"

    return [
        types.TextContent(
            type="text",
            text=response_text,
        )
    ]


async def handle_rebuild_journal_index(
    arguments: dict | None,
) -> list[types.TextContent]:
    """Handle the rebuild-journal-index tool."""
    try:
        force = arguments.get("force", False) if arguments else False

        rag = await get_rag_system()

        # Rebuild index
        stats = await rag.build_index(force_rebuild=force)

        response_text = f"""
Index Rebuild Complete:
• Total Files: {stats["total_files"]}
• Processed Files: {stats["processed_files"]}
• Skipped Files: {stats["skipped_files"]}
• Total Chunks: {stats["total_chunks"]}
• Errors: {stats["errors"]}
• Duration: {stats["duration"]:.2f}s
        """.strip()

    except Exception as e:
        response_text = f"Error rebuilding index: {str(e)}"

    return [
        types.TextContent(
            type="text",
            text=response_text,
        )
    ]
