"""
Document Model

Represents one page extracted from the PDF.

"""

from dataclasses import dataclass

@dataclass
class Document:
    page_number: int
    text: str
    source: str