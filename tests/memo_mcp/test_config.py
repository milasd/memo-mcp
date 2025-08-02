from memo_mcp.config import DATA_DIR, TOP_K
from os.path import exists
from pathlib import Path

"""
test_config.py
--------------
Test file to assess whether variables set for memo_mcp config are valid.
"""

def test_data_dir_type():
    assert isinstance(DATA_DIR, Path)

def test_data_dir_exists():
    assert exists(DATA_DIR)

def test_top_k_type():
    assert isinstance(TOP_K, int)

def test_top_k_positive():
    assert TOP_K > 0
