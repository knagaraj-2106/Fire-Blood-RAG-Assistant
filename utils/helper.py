"""
Common Helper Functions
"""

from pathlib import Path

def create_directory(path:Path):
    path.mkdir(parents=True, exist_ok=True)

def file_exists(path:Path):
    return path.exists()