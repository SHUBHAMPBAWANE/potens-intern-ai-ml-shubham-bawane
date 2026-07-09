"""
embeddings.py

Production-ready embedding module for a Retrieval-Augmented Generation (RAG)
application. Wraps the `sentence-transformers` library around the
`BAAI/bge-small-en-v1.5` model to provide document and query embeddings.

This module is intentionally scoped to embedding generation only. It does
NOT implement vector storage, indexing, or retrieval logic — those concerns
belong in separate modules (e.g. a vector store / retriever module).

Typical usage:

    from embeddings import EmbeddingService

    service = EmbeddingService()  # model is not loaded yet
    doc_vectors = service.embed_documents(["Paris is the capital of France."])
    query_vector = service.embed_query("What is the capital of France?")
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

if not logger.handlers:
    # Library-friendly default: attach a NullHandler so this module never
    # emits log records unless the host application configures logging.
    logger.addHandler(logging.NullHandler())


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models are trained with a specific instruction prefix for queries to
# improve retrieval quality (asymmetric search). Document embeddings do not
# use a prefix. See: https://huggingface.co/BAAI/bge-small-en-v1.5
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingServiceError(Exception):
    """Base exception for all errors raised by EmbeddingService."""


class ModelLoadError(EmbeddingServiceError):
    """Raised when the underlying sentence-transformers model fails to load."""


class EmbeddingGenerationError(EmbeddingServiceError):
    """Raised when embedding generation fails for given input text(s)."""


class EmbeddingService:
    """
    A reusable service for generating text embeddings using a
    sentence-transformers model (default: BAAI/bge-small-en-v1.5).

    The underlying model is loaded lazily: it is only instantiated the
    first time an embedding method is called (or `load_model()` is
    explicitly invoked), and is cached for the lifetime of the
    EmbeddingService instance. This avoids the cost of loading the model
    (which can take a noticeable amount of time and memory) until it is
    actually needed, while guaranteeing it is loaded at most once per
    instance.

    Thread safety: model loading is guarded by a lock, so it is safe to
    share a single EmbeddingService instance across multiple threads.

    Attributes:
        model_name: The HuggingFace model identifier used to load the
            SentenceTransformer model.
        device: The device the model should be loaded on (e.g. "cpu",
            "cuda", "mps"). If None, sentence-transformers will
            auto-select an available device.
        normalize_embeddings: Whether to L2-normalize output embeddings.
            BGE models are designed to be used with cosine similarity on
            normalized vectors, so this defaults to True.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize the EmbeddingService configuration.

        Note: this constructor does NOT load the model. The model is
        loaded lazily on first use (see class docstring).

        Args:
            model_name: HuggingFace model identifier to load via
                sentence-transformers. Defaults to
                "BAAI/bge-small-en-v1.5".
            device: Target device string (e.g. "cpu", "cuda", "cuda:0",
                "mps"). If None, sentence-transformers chooses
                automatically based on availability.
            normalize_embeddings: Whether embeddings should be
                L2-normalized before being returned. Recommended True
                for BGE models when using cosine similarity.
        """
        self.model_name: str = model_name
        self.device: Optional[str] = device
        self.normalize_embeddings: bool = normalize_embeddings

        self._model = None  # type: Optional["SentenceTransformer"]
        self._lock = threading.Lock()

        logger.debug(
            "EmbeddingService configured (model_name=%s, device=%s, "
            "normalize_embeddings=%s). Model will be loaded lazily on first use.",
            self.model_name,
            self.device,
            self.normalize_embeddings,
        )

    def load_model(self) -> None:
        """
        Explicitly load the sentence-transformers model if it has not
        already been loaded.

        This method is idempotent and thread-safe: calling it multiple
        times, including concurrently from multiple threads, will result
        in the model being loaded exactly once. Most callers do not need
        to call this directly, since `embed_documents` and `embed_query`
        call it automatically, but it is useful for eagerly "warming up"
        the service (e.g. at application startup) so the first real
        request does not pay the model-loading latency cost.

        Raises:
            ModelLoadError: If the model fails to load, for example due
                to a missing dependency, an invalid model name, or a
                network/disk error while fetching model weights.
        """
        if self._model is not None:
            return

        with self._lock:
            # Re-check inside the lock in case another thread loaded the
            # model while we were waiting to acquire it.
            if self._model is not None:
                return

            logger.info("Loading embedding model '%s'...", self.model_name)
            try:
                # Imported here (rather than at module level) so that
                # importing this module does not require sentence-transformers
                # to be installed unless embeddings are actually used, and so
                # that model loading truly only happens on first use.
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                logger.error(
                    "sentence-transformers is not installed. Install it with "
                    "'pip install sentence-transformers'."
                )
                raise ModelLoadError(
                    "The 'sentence-transformers' package is required but not "
                    "installed. Install it with: pip install sentence-transformers"
                ) from exc

            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                logger.exception(
                    "Failed to load embedding model '%s'.", self.model_name
                )
                raise ModelLoadError(
                    f"Failed to load sentence-transformers model "
                    f"'{self.model_name}': {exc}"
                ) from exc

            logger.info(
                "Embedding model '%s' loaded successfully on device '%s'.",
                self.model_name,
                getattr(self._model, "device", self.device),
            )

    @property
    def is_loaded(self) -> bool:
        """
        Whether the underlying model has already been loaded into memory.

        Returns:
            True if the model has been loaded, False if it is still
            pending lazy initialization.
        """
        return self._model is not None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of documents (passages).

        Use this method for the "corpus" side of a RAG pipeline — i.e.
        the chunks of text you intend to index and later retrieve. For
        embedding a user's search query, use `embed_query` instead, since
        BGE models use an asymmetric embedding scheme with a different
        instruction prefix for queries.

        Args:
            texts: A list of document strings to embed. Must be
                non-empty, and each element must be a non-empty string.

        Returns:
            A list of embedding vectors (one per input text), where each
            vector is a list of floats. The order of the returned
            embeddings matches the order of `texts`.

        Raises:
            ValueError: If `texts` is empty, is not a list, or contains
                any non-string or empty/whitespace-only elements.
            ModelLoadError: If the embedding model fails to load.
            EmbeddingGenerationError: If embedding generation fails for
                the given input.
        """
        self._validate_texts(texts)
        self.load_model()

        logger.debug("Embedding %d document(s).", len(texts))
        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Failed to generate document embeddings.")
            raise EmbeddingGenerationError(
                f"Failed to generate document embeddings: {exc}"
            ) from exc

        logger.debug("Successfully embedded %d document(s).", len(texts))
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single search query.

        BGE models are trained for asymmetric retrieval: queries are
        prefixed with a specific instruction string
        ("Represent this sentence for searching relevant passages: ")
        while documents are not. This method applies that prefix
        automatically, so callers should pass the raw, unprefixed query
        text.

        Args:
            text: The query string to embed. Must be a non-empty,
                non-whitespace-only string.

        Returns:
            The embedding vector for the query, as a list of floats.

        Raises:
            ValueError: If `text` is not a string or is empty/whitespace-only.
            ModelLoadError: If the embedding model fails to load.
            EmbeddingGenerationError: If embedding generation fails for
                the given input.
        """
        self._validate_texts([text], context="query")
        self.load_model()

        prefixed_text = f"{BGE_QUERY_INSTRUCTION}{text}"
        logger.debug("Embedding query text.")
        try:
            embedding = self._model.encode(
                prefixed_text,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Failed to generate query embedding.")
            raise EmbeddingGenerationError(
                f"Failed to generate query embedding: {exc}"
            ) from exc

        logger.debug("Successfully embedded query.")
        return embedding.tolist()

    @staticmethod
    def _validate_texts(texts: List[str], context: str = "document") -> None:
        """
        Validate input text(s) prior to embedding.

        Args:
            texts: List of strings to validate.
            context: A short label ("document" or "query") used to
                produce clearer error messages.

        Raises:
            ValueError: If `texts` is not a non-empty list, or if any
                element is not a non-empty, non-whitespace-only string.
        """
        if not isinstance(texts, list):
            raise ValueError(
                f"Expected a list of {context} strings, got {type(texts).__name__}."
            )

        if len(texts) == 0:
            raise ValueError(f"Cannot embed an empty list of {context}s.")

        for index, item in enumerate(texts):
            if not isinstance(item, str):
                raise ValueError(
                    f"{context.capitalize()} at index {index} is not a string "
                    f"(got {type(item).__name__})."
                )
            if not item.strip():
                raise ValueError(
                    f"{context.capitalize()} at index {index} is empty or "
                    "whitespace-only."
                )