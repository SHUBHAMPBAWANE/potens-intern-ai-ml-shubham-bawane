"""
chunking.py

Splits LangChain Document objects into semantic chunks
using RecursiveCharacterTextSplitter while preserving metadata.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into smaller semantic chunks.

    Args:
        documents:
            List of LangChain Document objects.

        chunk_size:
            Maximum characters per chunk.

        chunk_overlap:
            Number of overlapping characters.

    Returns
    -------
    List[Document]
        Chunked documents with preserved metadata.
    """

    if not documents:
        logger.warning("No documents supplied for chunking.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index + 1

    logger.info(
        "Generated %d chunks from %d documents.",
        len(chunks),
        len(documents),
    )

    return chunks