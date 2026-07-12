"""
retriever_service.py

Semantic retrieval service for the RAG application.

Responsibilities:
- Retrieve the most relevant document chunks
- Return similarity scores
- Hide ChromaDB response format from the rest of the application
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from app.core.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    """
    Represents one retrieved chunk.
    """

    document: Document

    similarity_score: float

    distance: float

@staticmethod
def similarity_label(score: float) -> str:
    """
    Convert a similarity score into a human-readable label.
    """

    if score >= 0.90:
        return "Excellent"

    if score >= 0.75:
        return "Good"

    if score >= 0.60:
        return "Fair"

    return "Weak"
@property
def citation(self) -> str:
    return f"{self.document.metadata['source']} (Page {self.document.metadata['page']})"

class RetrieverService:
    """
    Semantic retriever built on top of the vector store.
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
    ) -> None:

        self.vector_store = vector_store

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant document chunks.

        Args:
            query_embedding:
                Embedding of the user's question.

            top_k:
                Maximum number of retrieved chunks.

        Returns:
            List of RetrievalResult objects.
        """

        logger.info(
            "Retrieving top %d document(s).",
            top_k,
        )

        response = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: List[RetrievalResult] = []

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            document = Document(
                page_content=text,
                metadata=metadata,
            )

            similarity_score = max(0.0, 1.0 - (distance / 2.0))

            results.append(
                RetrievalResult(
                    document=document,
                    similarity_score=similarity_score,
                    distance=distance,
                )
            )
        logger.info(
            "Retrieved %d document(s).",
            len(results),
        )

        return results
    @staticmethod
    def similarity_label(score: float) -> str:
        """
        Convert a similarity score into a human-readable label.
        """

        if score >= 0.90:
            return "Excellent"

        if score >= 0.75:
            return "Good"

        if score >= 0.60:
            return "Fair"

        return "Weak"