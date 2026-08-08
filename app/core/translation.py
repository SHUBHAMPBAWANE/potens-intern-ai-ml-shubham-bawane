"""
Translation Service.

Provides translation of user queries and/or retrieved content between
languages using Gemini.
"""

from __future__ import annotations
from langdetect import detect, LangDetectException
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from google import genai

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Translation service powered by Gemini.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = os.getenv(
            "GEMINI_MODEL_NAME",
            "gemini-3.5-flash"
        )

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str = "auto",
    ) -> str:
        """
        Translate text into the target language.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        prompt = f"""
Translate the following text into {target_language}.

Source language: {source_language}

Rules:
- Return only the translated text.
- Do not explain the translation.
- Preserve numbers, names, prices, and important formatting.
- Do not add extra information.

Text:
{text}
""".strip()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            translated_text = response.text

            if not translated_text:
                raise ValueError(
                    "Gemini returned an empty translation."
                )

            return translated_text.strip()

        except Exception as exc:
            logger.exception(
                "Translation request failed."
            )

            raise RuntimeError(
                f"Translation failed: {exc}"
            ) from exc

    def detect_language(self, text: str) -> str:
     """
    Detect the language of the given text locally.

    This avoids an unnecessary Gemini API request.
    """

     if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

     try:
            language_code = detect(text)

            language_map = {
                "en": "English",
                "hi": "Hindi",
                "mr": "Marathi",
                "gu": "Gujarati",
                "bn": "Bengali",
                "ta": "Tamil",
                "te": "Telugu",
                "kn": "Kannada",
                "ml": "Malayalam",
                "pa": "Punjabi",
                "ur": "Urdu",
                "fr": "French",
                "de": "German",
                "es": "Spanish",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "zh-cn": "Chinese",
                "zh-tw": "Chinese",
            }

            return language_map.get(
                language_code,
                language_code,
            )

     except LangDetectException as exc:
            logger.exception(
                "Language detection failed."
            )

            raise RuntimeError(
                f"Language detection failed: {exc}"
            ) from exc