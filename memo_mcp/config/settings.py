"""
Path and constant configurations for memo-mcp.

This module defines the core paths and constants used throughout
the memo-mcp project, including data directories and default values.
"""

from pathlib import Path

# Data directory path - where journal entries are stored
# Expected stored file structure: DATA_DIR/YYYY/MM/DD.md
DATA_DIR: Path = Path("data/memo")

# Default max. number of results to return from RAG queries
TOP_K: int = 366
