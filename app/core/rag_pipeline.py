"""
rag_pipeline.py

Orchestrates the complete RAG ingestion workflow.

Responsibilities:
- Load PDF documents
- Chunk documents
- Generate embeddings
- Store embeddings in ChromaDB

Retrieval and LLM generation are intentionally implemented
in separate services.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.chunking import chunk_documents
from app.core.document_loader import load_documents
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    High-level orchestration layer for the RAG application.

    This class coordinates the ingestion workflow by connecting
    the document loader, chunker, embedding service, and vector store.
    """

    def __init__(
        self,
        docs_directory: str | Path = "docs",
    ) -> None:

        self.docs_directory = Path(docs_directory)

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStoreService()

        logger.info("RAG Pipeline initialized successfully.")

    
    def ingest_documents(self) -> dict[str, Any]:

        logger.info("Starting document ingestion.")

        if not self.docs_directory.exists():
            raise FileNotFoundError(
                f"Documents directory '{self.docs_directory}' does not exist."
            )
        
        documents = load_documents(str(self.docs_directory))
        if not documents:
            logger.warning("No documents found for ingestion.")

            return {
                "documents": 0,
                "chunks": 0,
                "vectors": self.vector_store.count(),
            }
        logger.info("Chunking %d document(s).", len(documents))
        chunks = chunk_documents(documents)
        logger.info("Generating embeddings.")
        texts = [chunk.page_content for chunk in chunks]

        embeddings = self.embedding_service.embed_documents(texts)
        logger.info("Persisting embeddings to vector store.")
        self.vector_store.add_documents(
            chunks,
            embeddings,
        )
        logger.info("Document ingestion completed successfully.")
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "vectors": self.vector_store.count(),
    }
    def reset_database(self) -> None:
        """
        Reset the vector database.
        """

        self.vector_store.reset()

    def get_stats(self) -> dict[str, Any]:

        return {
            "status": "ready",
            "vector_count": self.vector_store.count(),
            "docs_directory": str(self.docs_directory),
        }
    def ask(self, question: str) -> str:

        """
        Answer a user query.

        This method will be implemented after the RetrieverService
        and LLM integration are completed.
        """

        raise NotImplementedError(
            "Retriever has not been implemented yet."
        )