"""
Text chunking.

Splits normalized documents into retrievable chunks (e.g., fixed-size,
recursive, semantic, or sentence-window strategies) with configurable
size/overlap.
"""

from typing import Any, List


class Chunker:
    """
    TODO: Implement one or more chunking strategies.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Any) -> List[Any]:
        """
        TODO: Split a loaded document into a list of chunks,
        preserving metadata (source, page number, position, etc.).
        """
        pass
