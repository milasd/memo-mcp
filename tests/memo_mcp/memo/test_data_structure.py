import re
import pytest
from pathlib import Path
from datetime import datetime
from tests.settings import TEST_DATA_DIR

"""
test_data_structure.py
----------------------
Test file to validate the data structure for memo files.

Requirements:
1. Files must be structured as [year]/[month]/[day].md
2. Only .md files are allowed in month folders
3. No files (including .md) are allowed directly in year folders
4. Year must be 4 digits, month must be 2 digits (01-12), day must be 2 digits (01-31)
5. File names must be valid dates
"""

@pytest.fixture
def data_dir() -> Path:
    """Get the data directory path."""
    return TEST_DATA_DIR / "memo_example"


def test_data_directory_exists(data_dir: Path):
    """Test that the data directory exists."""
    assert data_dir.exists(), f"Data directory {data_dir} does not exist"
    assert data_dir.is_dir(), f"Data directory {data_dir} is not a directory"


def test_year_folder_structure(data_dir: Path):
    """Test that year folders follow the correct format (4 digits)."""
    if not data_dir.exists():
        pytest.skip("Data directory does not exist")
    
    year_pattern = re.compile(r'^\d{4}$')
    
    for item in data_dir.iterdir():
        if item.is_dir():
            assert year_pattern.match(item.name), \
                f"Year folder '{item.name}' does not match YYYY format"
            

def test_no_files_in_year_folders(data_dir: Path):
    """Test that no files exist directly in year folders."""
    for year_dir in data_dir.iterdir():
        if year_dir.is_dir() and re.match(r'^\d{4}$', year_dir.name):
            files_in_year = [f for f in year_dir.iterdir() if f.is_file()]
            assert len(files_in_year) == 0, \
                f"Year folder '{year_dir.name}' contains files: {[f.name for f in files_in_year]}. " \
                f"Files should only be in month folders."


def test_month_folder_structure(data_dir: Path):
    """Test that month folders follow the correct format (01-12)."""
    if not data_dir.exists():
        pytest.skip("Data directory does not exist")
    
    month_pattern = re.compile(r'^(0[1-9]|1[0-2])$')
    
    for year_dir in data_dir.iterdir():
        if year_dir.is_dir() and re.match(r'^\d{4}$', year_dir.name):
            for item in year_dir.iterdir():
                if item.is_dir():
                    assert month_pattern.match(item.name), \
                        f"Month folder '{item.name}' in year '{year_dir.name}' " \
                        f"does not match MM format (01-12)"


def test_only_md_files_in_month_folders(data_dir: Path):
    """Test that only .md files exist in month folders."""
    if not data_dir.exists():
        pytest.skip("Data directory does not exist")
    
    for year_dir in data_dir.iterdir():
        if year_dir.is_dir() and re.match(r'^\d{4}$', year_dir.name):
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and re.match(r'^(0[1-9]|1[0-2])$', month_dir.name):
                    for file_item in month_dir.iterdir():
                        if file_item.is_file():
                            assert file_item.suffix == '.md', \
                                f"Non-markdown file '{file_item.name}' found in " \
                                f"{year_dir.name}/{month_dir.name}/. Only .md files are allowed."


def test_day_file_naming_format(data_dir: Path):
    """Test that day files follow DD.md format and are valid dates."""
    if not data_dir.exists():
        pytest.skip("Data directory does not exist")
    
    day_pattern = re.compile(r'^(0[1-9]|[12][0-9]|3[01])\.md$')
    
    for year_dir in data_dir.iterdir():
        if year_dir.is_dir() and re.match(r'^\d{4}$', year_dir.name):
            year = int(year_dir.name)
            
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and re.match(r'^(0[1-9]|1[0-2])$', month_dir.name):
                    month = int(month_dir.name)
                    
                    for file_item in month_dir.iterdir():
                        if file_item.is_file():
                            assert day_pattern.match(file_item.name), \
                                f"File '{file_item.name}' in {year_dir.name}/{month_dir.name}/ " \
                                f"does not match DD.md format"
                            
                            # Extract day and validate it's a valid date
                            day = int(file_item.stem)
                            try:
                                datetime(year, month, day)
                            except ValueError as e:
                                pytest.fail(
                                    f"Invalid date: {year}-{month:02d}-{day:02d} "
                                    f"for file {year_dir.name}/{month_dir.name}/{file_item.name}. "
                                    f"Error: {e}"
                                )


def test_no_duplicate_day_files(data_dir: Path):
    """Test that there are no duplicate day files in the same month."""
    if not data_dir.exists():
        pytest.skip("Data directory does not exist")
    
    for year_dir in data_dir.iterdir():
        if year_dir.is_dir() and re.match(r'^\d{4}$', year_dir.name):
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and re.match(r'^(0[1-9]|1[0-2])$', month_dir.name):
                    day_files = []
                    for file_item in month_dir.iterdir():
                        if file_item.is_file() and file_item.suffix == '.md':
                            day_files.append(file_item.stem)
                    
                    # Check for duplicates
                    unique_days = set(day_files)
                    assert len(day_files) == len(unique_days), \
                        f"Duplicate day files found in {year_dir.name}/{month_dir.name}/: " \
                        f"{[day for day in day_files if day_files.count(day) > 1]}"


def test_no_unexpected_directories(data_dir: Path):
    """Test that there are no unexpected directories in the structure."""
    # Check root level - should only contain year directories and README.md
    for item in data_dir.iterdir():
        if item.is_dir():
            assert re.match(r'^\d{4}$', item.name), \
                f"Unexpected directory '{item.name}' in root. Only YYYY directories allowed."
        elif item.is_file():
            assert item.name in ['README.md'], \
                f"Unexpected file '{item.name}' in root. Only README.md allowed."
    
    # Check year level - should only contain month directories
    for year_dir in data_dir.iterdir():
        if year_dir.is_dir() and re.match(r'^\d{4}$', year_dir.name):
            for item in year_dir.iterdir():
                assert item.is_dir(), \
                    f"Unexpected file '{item.name}' in year directory {year_dir.name}. " \
                    f"Only month directories allowed."
                assert re.match(r'^(0[1-9]|1[0-2])$', item.name), \
                    f"Unexpected directory '{item.name}' in year {year_dir.name}. " \
                    f"Only MM directories allowed."
    
    # Check month level - should only contain .md files
    for year_dir in data_dir.iterdir():
        if year_dir.is_dir() and re.match(r'^\d{4}$', year_dir.name):
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and re.match(r'^(0[1-9]|1[0-2])$', month_dir.name):
                    for item in month_dir.iterdir():
                        assert item.is_file(), \
                            f"Unexpected directory '{item.name}' in month directory " \
                            f"{year_dir.name}/{month_dir.name}/. Only .md files allowed."


def test_file_content_accessibility(data_dir: Path):
    """Test that all .md files are readable as text."""
    if not data_dir.exists():
        pytest.skip("Data directory does not exist")
    
    for year_dir in data_dir.iterdir():
        if year_dir.is_dir() and re.match(r'^\d{4}$', year_dir.name):
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and re.match(r'^(0[1-9]|1[0-2])$', month_dir.name):
                    for file_item in month_dir.iterdir():
                        if file_item.is_file() and file_item.suffix == '.md':
                            try:
                                content = file_item.read_text(encoding='utf-8')
                                # Files can be empty, but should be readable
                                assert isinstance(content, str), \
                                    f"File {file_item} is not readable as text"
                            except Exception as e:
                                pytest.fail(
                                    f"Cannot read file {file_item}: {e}"
                                )