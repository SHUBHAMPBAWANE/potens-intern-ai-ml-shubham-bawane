"""
document_loader.py

Loads PDF documents from a directory using PyMuPDF and returns
LangChain Document objects with cleaned text and metadata.

Each page of a PDF becomes one Document object.
"""

from pathlib import Path
import logging
import re
from typing import List

import fitz
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_single_pdf(file_path: Path) -> List[Document]:
    """
    Load a single PDF and return one Document per page.

    Metadata stored with each document:
        - source : PDF filename
        - page   : Page number (1-based)
        - path   : Absolute file path

    Args:
        file_path: Path object pointing to a PDF.

    Returns:
        List of LangChain Document objects.
    """
    documents: List[Document] = []

    try:
        pdf = fitz.open(file_path)
    except Exception as e:
        logger.warning(
            f"Skipping unreadable PDF '{file_path.name}': {e}"
        )
        return documents

    try:
        for page_number in range(len(pdf)):
            try:
                page = pdf.load_page(page_number)

                text = _clean_text(page.get_text("text"))

                if not text:
                    logger.info(
                        f"No text found on page {page_number + 1} of '{file_path.name}'"
                    )
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page": page_number + 1,
                            "path": str(file_path.resolve()),
                        },
                    )
                )

            except Exception as page_error:
                logger.warning(
                    f"Skipping page {page_number + 1} in '{file_path.name}': {page_error}"
                )

    finally:
        pdf.close()

    logger.info(
        f"Loaded {len(documents)} page(s) from '{file_path.name}'"
    )

    return documents


def load_documents(docs_dir: str | Path) -> List[Document]:
    """
    Load all PDF files from a directory.

    Each PDF page is converted into one LangChain Document.

    Returns:
        List[Document]

    Each Document contains:
        page_content

    metadata:
        source : filename
        page   : page number
        path   : absolute path
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        logger.error(f"Directory not found: {docs_path}")
        return []

    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s).")

    all_documents: List[Document] = []

    for pdf_file in pdf_files:
        all_documents.extend(_load_single_pdf(pdf_file))

    logger.info(
        f"Loaded {len(all_documents)} document page(s) in total."
    )

    return all_documents