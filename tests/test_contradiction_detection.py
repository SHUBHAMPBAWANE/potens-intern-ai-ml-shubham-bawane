"""
Test the Contradiction Detection Service.
"""

from app.core.contradiction_detection import (
    ContradictionDetectionService,
)


def main():
    print("=" * 60)
    print("Testing Contradiction Detection Service")
    print("=" * 60)

    detector = ContradictionDetectionService()

    documents_without_contradiction = [
        """
        The Tactile Push Button Switch has an amount of Rs 11.86
        and a total amount of Rs 14.00 including tax.
        """,

        """
        The Tactile Push Button Switch has a total price of Rs 14.00
        after including tax.
        """,
    ]

    print("\nTesting documents without contradiction...\n")

    result = detector.detect(
        documents_without_contradiction
    )

    print("Result:")
    print(result)

    documents_with_contradiction = [
        """
        The Tactile Push Button Switch has a total amount of Rs 14.00.
        """,

        """
        The Tactile Push Button Switch has a total amount of Rs 19.00.
        """,
    ]

    print("\nTesting documents with contradiction...\n")

    result = detector.detect(
        documents_with_contradiction
    )

    print("Result:")
    print(result)

    print("\n" + "=" * 60)
    print("Test completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()