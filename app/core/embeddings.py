"""
Embedding generation.

Wraps an embedding model/provider (local or API-based) to convert
text chunks and queries into vector representations.
"""

from typing import List


class EmbeddingService:
    """
    TODO: Implement embedding generation using the configured model/provider.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        """
        TODO: Generate an embedding vector for a single piece of text.
        """
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        TODO: Generate embedding vectors for a batch of texts.
        """
        pass
