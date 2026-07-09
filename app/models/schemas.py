"""
Pydantic request/response models used by the API layer.
"""

from typing import List, Optional

from pydantic import BaseModel


# --- Ingestion -------------------------------------------------------------

class IngestResponse(BaseModel):
    """
    TODO: Fields such as document_id, num_chunks, status.
    """
    pass


# --- Query / RAG -------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    TODO: Fields such as query, top_k, filters, language.
    """
    pass


class RetrievedChunk(BaseModel):
    """
    TODO: Fields such as chunk_id, content, source, score.
    """
    pass


class QueryResponse(BaseModel):
    """
    TODO: Fields such as answer, sources: List[RetrievedChunk].
    """
    pass


# --- Translation -------------------------------------------------------------

class TranslationRequest(BaseModel):
    """
    TODO: Fields such as text, target_language, source_language.
    """
    pass


class TranslationResponse(BaseModel):
    """
    TODO: Fields such as translated_text, detected_source_language.
    """
    pass


# --- Contradiction detection --------------------------------------------------

class ContradictionCheckRequest(BaseModel):
    """
    TODO: Fields such as passages: List[str] or answer + sources.
    """
    pass


class ContradictionResult(BaseModel):
    """
    TODO: Fields such as statement_a, statement_b, explanation, confidence.
    """
    pass


class ContradictionCheckResponse(BaseModel):
    """
    TODO: Fields such as contradictions: List[ContradictionResult].
    """
    pass
