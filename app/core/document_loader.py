"""
Document loading.

Responsible for reading raw documents (PDF, DOCX, TXT, HTML, etc.)
from various sources (file upload, path, URL) and converting them
into a normalized internal representation for downstream chunking.
"""

from typing import Any


class DocumentLoader:
    """
    TODO: Implement loaders per file type (PDF, DOCX, TXT, HTML, CSV, ...).
    """

    def load_from_path(self, path: str) -> Any:
        """
        TODO: Load a document from a local file path.
        """
        pass

    def load_from_bytes(self, content: bytes, filename: str) -> Any:
        """
        TODO: Load a document from raw bytes (e.g., an uploaded file).
        """
        pass

    def load_from_url(self, url: str) -> Any:
        """
        TODO: Fetch and load a document from a URL.
        """
        pass
