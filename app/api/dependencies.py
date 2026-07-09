"""
FastAPI dependency providers.

Wires together core services (retrieval, embeddings, translation,
contradiction detection) for injection into route handlers.
"""


def get_document_loader():
    """
    TODO: Return a configured document loader instance.
    """
    pass


def get_chunker():
    """
    TODO: Return a configured chunking service instance.
    """
    pass


def get_embedding_service():
    """
    TODO: Return a configured embedding service instance.
    """
    pass


def get_retrieval_service():
    """
    TODO: Return a configured retrieval service instance,
    wired with embeddings and vector store.
    """
    pass


def get_translation_service():
    """
    TODO: Return a configured translation service instance.
    """
    pass


def get_contradiction_service():
    """
    TODO: Return a configured contradiction detection service instance.
    """
    pass
