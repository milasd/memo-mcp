import mcp.server.stdio
import mcp.types as types

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from memo_mcp.rag import RAGConfig, create_rag_system, MemoRAG
from memo_mcp.utils.logging_setup import set_logger
from pydantic import AnyUrl
from pathlib import Path
from typing import Optional


# Data dir path
DATA_DIR: Path = Path("data/memo/")

# Default top k
TOP_K: int = 365

notes: dict[str, str] = {}

# Global RAG system instance
_rag_system: Optional[MemoRAG] = None

server = Server("memo-mcp")


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
        stats = _rag_system.get_stats()
        if stats["total_documents"] == 0:
            logger.info("Building RAG index...")
            await _rag_system.build_index()

        logger.info(f"RAG system ready with {stats['total_documents']} documents")

    return _rag_system


async def cleanup_rag_system():
    """Clean up the RAG system on shutdown."""
    global _rag_system
    if _rag_system:
        await _rag_system.close()
        _rag_system = None


def parse_date_filter_string(date_filter_str: str):
    """Convert string date filter to date range tuple."""
    from datetime import date, timedelta

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


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available note resources.
    Each note is exposed as a resource with a custom note:// URI scheme.
    """
    return [
        types.Resource(
            uri=AnyUrl(f"note://internal/{name}"),
            name=f"Note: {name}",
            description=f"A simple note named {name}",
            mimeType="text/plain",
        )
        for name in notes
    ]


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific note's content by its URI.
    The note name is extracted from the URI host component.
    """
    if uri.scheme != "note":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    name = uri.path
    if name is not None:
        name = name.lstrip("/")
        return notes[name]
    raise ValueError(f"Note not found: {name}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="summarize-notes",
            description="Creates a summary of all notes",
            arguments=[
                types.PromptArgument(
                    name="style",
                    description="Style of the summary (brief/detailed)",
                    required=False,
                )
            ],
        )
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    The prompt includes all current notes and can be customized via arguments.
    """
    if name != "summarize-notes":
        raise ValueError(f"Unknown prompt: {name}")

    style = (arguments or {}).get("style", "brief")
    detail_prompt = " Give extensive details." if style == "detailed" else ""

    return types.GetPromptResult(
        description="Summarize the current notes",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Here are the current notes to summarize:{detail_prompt}\n\n"
                    + "\n".join(
                        f"- {name}: {content}" for name, content in notes.items()
                    ),
                ),
            )
        ],
    )


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="add-note",
            description="Add a new note",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
        ),
        types.Tool(
            name="search-journal",
            description="Search through journal entries using RAG. Perfect for questions like 'how did I do at work this year?' or 'what were my thoughts on AI?'",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question about your journal entries",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {TOP_K})",
                        "minimum": 1,
                        "default": TOP_K,
                    },
                    "date_filter": {
                        "type": "string",
                        "description": "Optional date filter (e.g., '2025', '2025-01', '2025-01-15')",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get-journal-stats",
            description="Get statistics about your journal database (document count, chunks, etc.)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="rebuild-journal-index",
            description="Rebuild the journal search index (use if you've added new entries)",
            inputSchema={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Force rebuild even if index exists",
                        "default": False,
                    }
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    try:
        if name == "add-note":
            return await handle_add_note(arguments)
        elif name == "search-journal":
            return await handle_search_journal(arguments)
        elif name == "get-journal-stats":
            return await handle_get_journal_stats(arguments)
        elif name == "rebuild-journal-index":
            return await handle_rebuild_journal_index(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}",
            )
        ]


async def handle_add_note(arguments: dict | None) -> list[types.TextContent]:
    """Handle the add-note tool."""
    if not arguments:
        raise ValueError("Missing arguments")

    note_name = arguments.get("name")
    content = arguments.get("content")

    if not note_name or not content:
        raise ValueError("Missing name or content")

    # Update server state
    notes[note_name] = content

    # Notify clients that resources have changed
    await server.request_context.session.send_resource_list_changed()

    return [
        types.TextContent(
            type="text",
            text=f"Added note '{note_name}' with content: {content}",
        )
    ]


async def handle_search_journal(arguments: dict | None) -> list[types.TextContent]:
    """Handle the search-journal tool."""
    if not arguments:
        raise ValueError("Missing arguments")

    query = arguments.get("query")
    if not query:
        raise ValueError("Missing query parameter")

    top_k = arguments.get("top_k", TOP_K)
    date_filter_str = arguments.get("date_filter")
    date_filter = parse_date_filter_string(date_filter_str)

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
        rag = await get_rag_system()
        stats = rag.get_stats()

        response_text = f"""
Journal Statistics:
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


async def main():
    """Main server function with proper resource cleanup."""
    try:
        # Run the server using stdin/stdout streams
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="memo-mcp",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        # Clean up RAG system on shutdown
        await cleanup_rag_system()
