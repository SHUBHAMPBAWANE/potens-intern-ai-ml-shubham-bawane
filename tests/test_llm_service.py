"""
Test Gemini LLM service.
"""

from app.core.llm_service import LLMService


def main() -> None:

    print("=" * 60)
    print("Testing Gemini LLM Service")
    print("=" * 60)

    llm = LLMService()

    context = """
    The company provides 20 days of annual paid leave to
    eligible full-time employees.
    """

    question = "How many days of annual paid leave are provided?"

    print("\nSending question to Gemini...\n")

    answer = llm.generate(
        question=question,
        context=context,
    )

    print("Gemini Response")
    print("-" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("LLM Test Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()