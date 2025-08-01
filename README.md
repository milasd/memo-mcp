# Memo MCP: LLM journaling w/ RAG

Memo MCP is a local Model Context Protocol (MCP) server that provides flexible search capabilities over personal memo and journal entries using Retrieval-Augmented Generation (RAG) with indexing and GPU support for faster embeddings.

With Memo MCP, you can ask about past events from your journal entries (e.g. How has my mood changed in the past 3 months?), and the LLM can also be aware of past discussed events and point it without being asked if it judges appropriate to the current discussion. You can also add new entries automatically about the current day.

For privacy and safety, I recommend personal journaling with not-so-sensitive data in your personal computer. If you decide to use it with "external" models such as Claude, Gemini etc., be careful about any personal, sensitive data and information. 

## Table of Contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Memo Data](#memo-data)
- [MCP](#mcp)
- [RAG](#rag)
- [Architecture](#️-architecture)
- [Codebase Usage Examples](#codebase-usage-examples)
- [License](#license)


## Highlights

- **Journal processing**: Ask natural language questions such as `"How did I do at work this year?"`, `"How was my mood overall last week? How does it compare to last month?"` to LLMs and get relevant excerpts from your journal
- **GPU Acceleration**: Optional GPU support for faster embedding generation
- **Multiple Vector Stores**: Support for local ChromaDB, FAISS and simple in-memory storage. Will add Qdrant in the future.
- **Date Filtering**: Filter search results by year, month, or specific dates
- **MCP Integration**: Works seamlessly with Claude Desktop, Cline, and other MCP-compatible LLM clients
- **Automatic Indexing**: Automatically builds and maintains search indices for your journal entries

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager


1. Clone repository and install dependencies with uv:
   ```bash
   cd memo-mcp
   uv sync
   ```

2. Prepare your memo data ([Instructions](#memo-data)) structure appropriately. Example to create a sample entry:
   ```bash
   # Create the expected directory structure
   mkdir -p data/memo/2025/01
   
   # Add your journal entries in markdown format
   # Example: data/memo/2025/01/15.md
   echo "# January 15th, 2025
   
   Had a great day at work today. Finished the project ahead of schedule..." > data/memo/2025/01/15.md
   ```

3. Set up **Memo MCP** on your client MCP configuration file ([Instructions](#integration)).

Restart your client and you're ready to use!

 If you wish to customize "advanced" settings, such as `top_k`, `chunk_size`, etc., keep reading the documentation.



## Memo Data 

### Folder Structure

The Memo MCP expects the memo and journal entries to be stored in a folder using the pattern `[year]/[month]/[day].md`:

```
data/memo/
├── 2024/
│   ├── 01/
│   │   ├── 07.md
│   │   ├── 15.md
│   │   ├── 28.md
│   │   └── ...
│   ├── 08/
│   │   └── ...
│   └── ...
├── 2025/
│   ├── 07/
│   │   ├── 01.md
│   │   └── ...
│   └── ...
└── ...
```

It is recommended to store it inside `data/memo`, as it is the default folder being monitored by the MCP server, and it is already added to `.gitignore` for safety. 

A sample data folder can be found in `data/memo_example`. It contains multiple sample journal entries you can use to test the MCP querying.

### Using Custom Data Folder Paths

In case you wish to use a different memo/journal data directory instead of the recommended `data/memo` path, you can modify it:

1. **Update `.gitignore`**: Add your custom path to prevent Git tracking:
   ```bash
   # Add to .gitignore
   path/to/my_custom_memo_repo/
   ```

2. **Modify MCP Configuration**: Update the `data_path` in your MCP server configuration file or environment variables to point to your custom location.

3. **Update `server.py`**: Ensure the RAG indexer points to your new data directory by changing `DATA_DIR` path to your custom folder in `memo_mcp/mcp/server.py`.

**Always ensure your memo/journal data folder is included in `.gitignore` to prevent accidentally committing personal information to version control.**

### Indexing

Memo/journal data is indexed to speed up search and retrieval. A recommendation is to set up your agent to rebuild the index everyday to ensure that new entries are included in the search results -- Rebuilding the index periodically guarantees that all recent journal entries are properly integrated into your searchable personal archive.

## MCP

### Tools

The Memo MCP server connects theprovides these tools for LLM interaction:

1. **`search-journal`**: Search through journal entries
   - `query` (required): Your search question
   - `top_k` (optional): Number of results (1-20, default: 5)
   - `date_filter` (optional): Filter by date (e.g., "2025", "2025-01")

2. **`get-journal-stats`**: Get memo dataset stats

3. **`rebuild-journal-index`**: Rebuild the search index
   - `force` (optional): Force rebuild even if index exists

### Integration

You can integrate the Memo MCP to multiple services, such as Cline in Visual Studio Code, Claude Desktop, or any code editor of your choice.

#### Cline/VSCode
1. **Add the MCP server**: Add to your Cline MCP Config file (`[...]/cline_mcp_settings.json`), changing `[PATH TO BASE DIRECTORY]` to the directory containing the cloned `memo-mcp` repo:
```json
{
  "mcpServers": {
    "memo-mcp": {
      "disabled": false,
      "timeout": 60,
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory",
        "[PATH TO BASE DIRECTORY]/memo-mcp",
        "run",
        "memo-mcp"
      ]
    }
  }
}
```
2. **Restart Cline** and start asking questions about your journal, or ask it to write a new entry data too.


#### Claude Desktop

1. **Add the MCP server**:
   Add to your Claude Desktop config (`~/claude_desktop_config.json`). You can optionally set up envionment variables:
   ```json
   {
     "mcpServers": {
       "memo-mcp": {
         "command": "uv",
         "args": [
            "--directory",
            "[PATH TO BASE DIRECTORY]/memo-mcp",
            "run",
            "memo-mcp"
         ],
         "env": {
           "MEMO_DATA_ROOT": "/path/to/your/journal/data"
         }
       }
     }
   }
   ```

2. **Restart Claude Desktop** and start asking questions about your journal, or ask it to write a new entry data too.

### Queries

Once integrated, you can ask your LLMs things such as:

```
- "How did I feel about work this year?"
- "Show me entries about my vacation in July this year"
- "What goals did I set for myself in 2025?"
- "How was my mental health last month?"
- "How many times did I forget to get groceries last week?"
```

#### Example

Using Gemini 2.5 Pro, with the sample (fictional) data provided in `data/memo_example`:

```
> How did I feel about work this year?


Based on your journal entries from 2025, it seems you've had a dynamic and transformative year at work, marked by significant highs and challenging lows.

**Early Success and Satisfaction (January):**
The year started on a very positive note. You received a long-awaited promotion, which brought a feeling of overwhelming relief and gratitude. You celebrated this achievement and felt a deep sense of things "falling into place." Later that month, you had a highly productive day where you tackled your entire to-do list, leaving you with a satisfying sense of accomplishment.

**Struggles and Burnout (March):**
By March, your momentum seemed to wane. You described feeling stuck, struggling to find your rhythm, and your brain feeling like it was "moving through molasses." This culminated in a difficult team meeting where you felt you had nothing meaningful to contribute, highlighting a period of professional frustration.

**A Major Setback and a Shift in Perspective (June):**
June brought a significant challenge with a "disastrous" job interview. Despite your preparation, the experience left you cringing. However, this failure became a turning point. It prompted you to question whether you were on the right career path and if the "corporate ladder" was what you truly wanted.

**Finding a New Groove (November):**
Towards the end of the year, things took a positive turn. A new workflow system you implemented began to pay off, making your work more seamless and efficient. You adopted new productivity techniques, like time-blocking, which helped you feel more in control and less overwhelmed. This not only improved your own work life but also enabled you to help a colleague who was struggling with similar issues, bringing you a sense of satisfaction.

**Overall Themes:**
Throughout the year, your feelings about work have been a rollercoaster. You've experienced the thrill of success, the weight of burnout, and the sting of failure. However, each of these experiences seems to have contributed to a deeper understanding of what you want from your career. You've learned the importance of not just working hard, but also working smart, and you've begun to question traditional notions of success in favor of finding a more sustainable and fulfilling path.

```


## RAG

The Memo MCP system uses Retrieval Augmented Generation (RAG) to perform queries over your daily entries. 

Multiple vector search DBs are supported: `ChromaDB` and `FAISS` local servers. A simple in-memory vector database can also be used for testing and simple operations. 

The RAG (Retrieval-Augmented Generation) system creates vector embeddings of your memo and journal content, allowing for flexible queries such as "How did I handle stress last month?" or "What were my thoughts on career changes?" without manually scanning through files, surfacing relevant entries based on broader meanings rather than just keyword matches. 

**Memo MCP automatically detects if there's any GPU or MPS (Apple M1/M2...) available for performance optimisation**, significantly improving the speed of embedding generation. 

### Default Settings

The default vector database is `ChromaDB`. You can also choose `FAISS` or a simple in-memory implementation. All of these servers will run locally in your computer. 

Other settings can be configured via environment variables or by modifying the configuration in `memo_mcp/mcp/server.py`:

```python
config = RAGConfig(
    vector_store_type="chroma",        # "chroma", "faiss", or "simple"
    data_root=Path("data/memo"),       # Path to your journal files
    use_gpu=True,                      # Enable GPU acceleration
    cache_embeddings=True,             # Cache embeddings for faster startup
    chunk_size=512,                    # Text chunk size for processing
    default_top_k=5,                   # Default number of search results
    similarity_threshold=0.3           # Minimum similarity for results
)
```

### Environment Variables

- `MEMO_DATA_ROOT`: Override the default data directory
- `MEMO_USE_GPU`: Enable/disable GPU usage ("true"/"false")
- `MEMO_EMBEDDING_MODEL`: Custom embedding model name
- `MEMO_RAG_LITE`: Enable resource-constrained mode ("true"/"false")



## Architecture

[TODO: a comprehensible diagram showing how the RAG provides the "logic" for the retrieval, using the MCP as an interface to connect it as tools for the LLMs.]

## Codebase Usage Examples

If you want to experiment with the RAG and MCP without the aid of LLM plugins, you can try writing scripts to run the implementations.

Examples:

### RAG Usage (Python)

```python
from memo_mcp.rag import create_rag_system, RAGConfig
from pathlib import Path

# Create and initialize RAG system
config = RAGConfig(data_root=Path("data/memo"))
rag = await create_rag_system(config)

# Search your journal
results = await rag.query("productivity tips", top_k=3)
for result in results:
    print(f"File: {result['metadata'].file_name}")
    print(f"Text: {result['text'][:200]}...")
    print("---")

await rag.close()
```

### MCP Client Usage

```python
# With an MCP client library
import mcp

client = mcp.Client()
await client.connect_to_server("memo-mcp")

# Search journal
result = await client.call_tool("search-journal", {
    "query": "How did I handle stress this year?",
    "top_k": 5,
    "date_filter": "2025"
})

print(result.content[0].text)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- `memo_mcp/mcp/server.py` is built on top of the [Model Context Protocol](https://github.com/modelcontextprotocol/)
