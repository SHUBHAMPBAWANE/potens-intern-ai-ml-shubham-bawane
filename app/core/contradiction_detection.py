"""
Contradiction Detection Service.

Uses Gemini to analyze retrieved document chunks and determine
whether they contain conflicting information.
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)


class ContradictionDetectionService:
    """
    Detects contradictions between retrieved document chunks using Gemini.
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
            "gemini-3.5-flash",
        )

    def detect(
        self,
        documents: list[str],
    ) -> dict:
        """
        Analyze documents and detect whether they contradict each other.

        Args:
            documents:
                List of retrieved document chunks.

        Returns:
            Dictionary containing contradiction status and explanation.
        """

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        formatted_documents = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            formatted_documents.append(
                f"DOCUMENT {index}:\n{document}"
            )

        context = "\n\n".join(
            formatted_documents
        )

        prompt = f"""
Analyze the following documents and determine whether
they contain contradictory information.

Rules:

- Compare factual claims between the documents.
- Ignore differences in wording or formatting.
- Do not consider missing information a contradiction.
- Return only valid JSON.
- Do not add markdown or explanations outside the JSON.

Return exactly this structure:

{{
    "has_contradiction": true,
    "explanation": "Short explanation of the contradiction."
}}

If there is no contradiction, return:

{{
    "has_contradiction": false,
    "explanation": "No contradiction found."
}}

Documents:

{context}
""".strip()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            result = response.text

            if not result:
                raise ValueError(
                    "Gemini returned an empty contradiction detection result."
                )

            result = result.strip()

            # Remove Markdown JSON code fences if Gemini adds them.
            if result.startswith("```json"):
                result = result[len("```json"):].strip()

            elif result.startswith("```"):
                result = result[3:].strip()

            if result.endswith("```"):
                result = result[:-3].strip()

            # Convert Gemini's JSON string into a Python dictionary.
            try:
                contradiction_result = json.loads(result)

            except json.JSONDecodeError as exc:
                logger.error(
                    "Invalid JSON returned by Gemini: %s",
                    result,
                )

                raise ValueError(
                    "Gemini returned invalid JSON for contradiction detection."
                ) from exc

            # Make sure the result is actually a dictionary.
            if not isinstance(
                contradiction_result,
                dict,
            ):
                raise ValueError(
                    "Contradiction detection result must be a dictionary."
                )

            # Validate expected fields.
            if "has_contradiction" not in contradiction_result:
                raise ValueError(
                    "Contradiction result is missing 'has_contradiction'."
                )

            if "explanation" not in contradiction_result:
                raise ValueError(
                    "Contradiction result is missing 'explanation'."
                )

            return contradiction_result

        except Exception as exc:
            logger.exception(
                "Contradiction detection request failed."
            )

            raise RuntimeError(
                f"Contradiction detection failed: {exc}"
            ) from exc