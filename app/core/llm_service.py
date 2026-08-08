"""
llm_service.py

Gemini LLM service for the RAG application.

Responsibilities:
- Load Gemini API key securely from .env
- Initialize the Gemini client
- Generate answers from supplied context
- Prevent the model from answering using outside knowledge
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Raised when the Gemini LLM service encounters an error."""


class LLMService:
    """
    Service responsible for communication with Google's Gemini API.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash",
    ) -> None:

        # Load variables from .env
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise LLMServiceError(
                "GEMINI_API_KEY was not found. "
                "Please add it to your .env file."
            )

        self.model_name = model_name

        try:
            self.client = genai.Client(api_key=api_key)

        except Exception as exc:
            logger.exception("Failed to initialize Gemini client.")

            raise LLMServiceError(
                f"Failed to initialize Gemini client: {exc}"
            ) from exc

        logger.info(
            "Gemini LLM initialized with model '%s'.",
            self.model_name,
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Generate an answer using the supplied retrieved context.

        Args:
            question:
                User's question.

            context:
                Text retrieved from the vector database.

        Returns:
            Gemini-generated answer.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context or not context.strip():
            return (
                "I could not find relevant information in the "
                "provided documents."
            )

        prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the CONTEXT below.

Do not use outside knowledge.

If the answer cannot be found in the context, clearly say:
"I could not find the answer in the provided documents."

Do not invent facts or make assumptions.

CONTEXT:
--------------------
{context}
--------------------

QUESTION:
{question}

ANSWER:
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

        except Exception as exc:
            logger.exception("Gemini API request failed.")

            raise LLMServiceError(
                f"Gemini API request failed: {exc}"
            ) from exc

        answer = response.text

        if not answer:
            raise LLMServiceError(
                "Gemini returned an empty response."
            )

        return answer.strip()