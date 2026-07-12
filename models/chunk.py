from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:

    chunk_id: str

    page_number: int

    text: str

    source: str

    token_count: int

    chunk_index: int

    embedding: Optional[list] = None

    metadata: dict = field(default_factory=dict)