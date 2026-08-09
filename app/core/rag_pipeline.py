"""
rag_pipeline.py

Main orchestration layer for the RAG application.

Responsibilities:
- Load PDF documents
- Split documents into chunks
- Generate embeddings
- Store embeddings in ChromaDB
- Retrieve relevant chunks for a question
- Send retrieved context to Gemini
- Return the final answer with citations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.chunking import chunk_documents
from app.core.document_loader import load_documents
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStoreService
from app.core.retriever_service import RetrieverService
from app.core.llm_service import LLMService
from app.core.translation import TranslationService
from app.core.contradiction_detection import (
    ContradictionDetectionService,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    High-level orchestration layer for the complete RAG application.
    """

    def __init__(
        self,
        docs_directory: str | Path = "docs",
        top_k: int = 3,
    ) -> None:

        self.docs_directory = Path(docs_directory)
        self.top_k = top_k

        # Embedding service
        self.embedding_service = EmbeddingService()

        # Vector database
        self.vector_store = VectorStoreService()

        # Retriever
        self.retriever = RetrieverService(
            vector_store=self.vector_store
        )

        # Gemini LLM
        self.llm_service = LLMService()

        self.translation_service = TranslationService()

        self.contradiction_service = ContradictionDetectionService()

        logger.info("RAG Pipeline initialized successfully.")

    # ============================================================
    # INGESTION
    # ============================================================

    def ingest_documents(self) -> dict[str, Any]:
        """
        Ingest documents into the vector database.

        Workflow:

        PDF
        ↓
        Document Loader
        ↓
        Chunking
        ↓
        Embeddings
        ↓
        ChromaDB
        """

        logger.info("Starting document ingestion.")

        documents = load_documents(
            str(self.docs_directory)
        )

        if not documents:
            logger.warning(
                "No documents found for ingestion."
            )

            return {
                "documents": 0,
                "chunks": 0,
                "vectors": self.vector_store.count(),
            }

        chunks = chunk_documents(documents)

        if not chunks:
            logger.warning(
                "No chunks generated from documents."
            )

            return {
                "documents": len(documents),
                "chunks": 0,
                "vectors": self.vector_store.count(),
            }

        # IMPORTANT:
        # EmbeddingService expects strings, not Document objects.
        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_service.embed_documents(
                texts
            )
        )

        self.vector_store.add_documents(
            chunks,
            embeddings,
        )

        result = {
            "documents": len(documents),
            "chunks": len(chunks),
            "vectors": self.vector_store.count(),
        }

        logger.info(
            "Document ingestion completed: %s",
            result,
        )

        return result

    # ============================================================
    # RESET
    # ============================================================

    def reset_database(self) -> None:
        """
        Reset the ChromaDB vector collection.
        """

        logger.warning(
            "Resetting vector database."
        )

        self.vector_store.reset()

    # ============================================================
    # STATS
    # ============================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Return current RAG pipeline statistics.
        """

        return {
            "status": "ready",
            "vector_count": self.vector_store.count(),
            "docs_directory": str(
                self.docs_directory
            ),
        }

    # ============================================================
    # QUESTION ANSWERING
    # ============================================================

    def ask(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Answer a user question using the complete RAG pipeline.

        Workflow:

        Question
            ↓
        Language Detection
            ↓
        Translation to English
            ↓
        Query Embedding
            ↓
        Semantic Retrieval
            ↓
        Contradiction Detection
            ↓
        Context Building
            ↓
        Gemini Answer
            ↓
        Translate Answer Back
            ↓
        Final Response
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        original_question = question.strip()

        logger.info(
            "Processing question: %s",
            original_question,
        )

        # --------------------------------------------------------
        # 1. Detect language
        # --------------------------------------------------------

        detected_language = (
            self.translation_service.detect_language(
                original_question
            )
        )

        logger.info(
            "Detected language: %s",
            detected_language,
        )

        # --------------------------------------------------------
        # 2. Translate question to English if required
        # --------------------------------------------------------

        if detected_language.lower() != "english":

            translated_question = (
                self.translation_service.translate(
                    text=original_question,
                    target_language="English",
                    source_language=detected_language,
                )
            )

        else:
            translated_question = original_question

        logger.info(
            "Question used for retrieval: %s",
            translated_question,
        )

        # --------------------------------------------------------
        # 3. Convert question into embedding
        # --------------------------------------------------------

        query_embedding = (
            self.embedding_service.embed_query(
                translated_question
            )
        )

        # --------------------------------------------------------
        # 4. Retrieve relevant chunks
        # --------------------------------------------------------

        retrieved_results = (
            self.retriever.retrieve(
                query_embedding=query_embedding,
                top_k=self.top_k,
            )
        )

        if not retrieved_results:

            logger.warning(
                "No relevant documents found."
            )

            return {
                "question": original_question,
                "language": detected_language,
                "answer": (
                    "I could not find relevant "
                    "information in the provided documents."
                ),
                "sources": [],
                "contradiction": {
                    "has_contradiction": False,
                    "explanation": "No documents were retrieved.",
                },
                "results": [],
            }

        # --------------------------------------------------------
        # 5. Detect contradictions
        # --------------------------------------------------------

        retrieved_texts = [
            result.document.page_content
            for result in retrieved_results
        ]

        contradiction_result = (
            self.contradiction_service.detect(
                retrieved_texts
            )
        )

        # --------------------------------------------------------
        # 6. Build context for Gemini
        # --------------------------------------------------------

        context_parts = []

        for index, result in enumerate(
            retrieved_results,
            start=1,
        ):

            context_parts.append(
                f"""
    SOURCE {index}
    Citation: {result.citation}
    Similarity: {result.similarity_score:.3f}

    Content:
    {result.document.page_content}
    """.strip()
            )

        context = "\n\n".join(
            context_parts
        )

        # --------------------------------------------------------
        # 7. Generate answer using Gemini
        # --------------------------------------------------------

        answer = self.llm_service.generate(
            question=translated_question,
            context=context,
        )

        # --------------------------------------------------------
        # 8. Translate answer back to original language
        # --------------------------------------------------------

        if detected_language.lower() != "english":

            answer = (
                self.translation_service.translate(
                    text=answer,
                    target_language=detected_language,
                    source_language="English",
                )
            )

        # --------------------------------------------------------
        # 9. Build citations
        # --------------------------------------------------------

        sources = []

        for result in retrieved_results:

            sources.append(
                {
                    "citation": result.citation,
                    "similarity_score": round(
                        result.similarity_score,
                        3,
                    ),
                    "quality": (
                        result.similarity_label(
                            result.similarity_score
                        )
                    ),
                }
            )

        # --------------------------------------------------------
        # 10. Return complete response
        # --------------------------------------------------------

        return {
            "question": original_question,
            "language": detected_language,
            "answer": answer,
            "sources": sources,
            "contradiction": contradiction_result,
            "results": retrieved_results,
            }
       