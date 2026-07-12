"""
Retrieved Chunk Model
"""

from dataclasses import dataclass


@dataclass
class RetrievedChunk:

    chunk_id: str

    text: str

    page_number: int

    source: str

    similarity_score: float

    chunk_index: int