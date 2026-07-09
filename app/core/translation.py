"""
Translation.

Provides translation of user queries and/or retrieved content between
languages, enabling multilingual RAG (e.g., query in one language,
documents in another).
"""


class TranslationService:
    """
    TODO: Implement translation using the configured provider
    (e.g., a hosted API or local model).
    """

    def translate(self, text: str, target_language: str, source_language: str = "auto") -> str:
        """
        TODO: Translate the given text into the target language.
        """
        pass

    def detect_language(self, text: str) -> str:
        """
        TODO: Detect the language of the given text.
        """
        pass
