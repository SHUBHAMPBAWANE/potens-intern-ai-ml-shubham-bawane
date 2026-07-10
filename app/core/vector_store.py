"""
vector_store.py

Persistent vector storage layer for the RAG application.

Responsibilities:
- Create or load a persistent ChromaDB collection.
- Store chunked LangChain Document objects and their embeddings.
- Count stored vectors.
- Reset the collection.

This module does NOT perform retrieval or LLM inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStoreInitializationError(Exception):
    """Raised when the Chroma vector store cannot be initialized."""


class VectorStoreOperationError(Exception):
    """Raised when an operation on the vector store fails."""


class VectorStoreService:
    """
    Persistent storage service built on top of ChromaDB.

    Responsibilities
    ----------------
    - Create/load persistent Chroma collection
    - Store embeddings
    - Store metadata
    - Store original document chunks
    - Count stored vectors
    - Reset collection

    This class intentionally does NOT implement retrieval.
    """

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "policy_documents",
        collection_metadata: Optional[dict] = None,
    ) -> None:

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name

        logger.info(
            "Initializing ChromaDB collection '%s'",
            self.collection_name,
        )

        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
        except Exception as exc:
            logger.exception("Unable to initialize ChromaDB.")
            raise VectorStoreInitializationError(
                f"Failed to initialize ChromaDB: {exc}"
            ) from exc

        try:
            self._collection: Collection = (
                self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata=collection_metadata,
                )
            )
        except Exception as exc:
            logger.exception(
                "Unable to load collection '%s'.",
                self.collection_name,
            )
            raise VectorStoreInitializationError(
                f"Failed to create/load collection '{self.collection_name}'."
            ) from exc

        logger.info(
            "Collection '%s' loaded successfully (%d vectors).",
            self.collection_name,
            self.count(),
        )
    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> List[str]:
        """
        Store LangChain documents and their embeddings.

        Args:
            documents:
                Chunked LangChain Document objects.

            embeddings:
                Embedding vectors corresponding to each document.

        Returns:
            List of generated document IDs.

        Raises:
            VectorStoreOperationError:
                If storing documents fails.
        """

        if not documents:
            raise VectorStoreOperationError(
                "No documents provided for storage."
            )

        if len(documents) != len(embeddings):
            raise VectorStoreOperationError(
                "Documents and embeddings must have the same length."
            )

        ids = []

        texts = []
        metadatas = []
        vectors = []

        for index, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):

            source = doc.metadata.get("source", "document")
            page = doc.metadata.get("page", 0)
            chunk = doc.metadata.get("chunk", index + 1)

            document_id = (
                f"{Path(source).stem}_p{page}_c{chunk}"
            )

            ids.append(document_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            vectors.append(list(embedding))

        logger.info(
            "Adding %d vectors into '%s'.",
            len(ids),
            self.collection_name,
        )

        try:
            self._collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=vectors,
            )

        except Exception as exc:
            logger.exception("Failed to store vectors.")

            raise VectorStoreOperationError(
                f"Unable to store vectors: {exc}"
            ) from exc

        logger.info(
            "Collection '%s' now contains %d vectors.",
            self.collection_name,
            self.count(),
        )

        return ids

    def count(self) -> int:
        """
        Return number of stored vectors.

        Returns:
            Total vector count.
        """

        try:
            return self._collection.count()

        except Exception as exc:
            raise VectorStoreOperationError(
                f"Unable to count vectors: {exc}"
            ) from exc

    def reset(self) -> None:
        """
        Delete and recreate the collection.

        Useful during development when rebuilding the
        vector database from scratch.
        """

        logger.warning(
            "Resetting collection '%s'.",
            self.collection_name,
        )

        try:
            self._client.delete_collection(
                self.collection_name
            )

            self._collection = (
                self._client.get_or_create_collection(
                    name=self.collection_name
                )
            )

        except Exception as exc:
            logger.exception("Failed to reset collection.")

            raise VectorStoreOperationError(
                f"Unable to reset collection: {exc}"
            ) from exc

        logger.info(
            "Collection '%s' reset successfully.",
            self.collection_name,
        )