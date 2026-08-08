from app.core.translation import TranslationService


def main() -> None:
    print("=" * 60)
    print("Testing Translation Service")
    print("=" * 60)

    translator = TranslationService()

    # --------------------------------------------------
    # Test 1: Language Detection
    # --------------------------------------------------

    print("\nTesting Language Detection...")

    texts = [
        "What is the total amount of the tactile push button switch?",
        "टैक्टाइल पुश बटन स्विच की कुल राशि कितनी है?",
        "टॅक्टाइल पुश बटन स्विचची एकूण रक्कम किती आहे?"
    ]

    for text in texts:
        language = translator.detect_language(text)

        print(f"\nText     : {text}")
        print(f"Language : {language}")

    # --------------------------------------------------
    # Test 2: Translation
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Testing Translation")
    print("=" * 60)

    text = "What is the total amount of the tactile push button switch?"

    translated = translator.translate(
        text=text,
        target_language="Marathi",
        source_language="English",
    )

    print(f"\nOriginal   : {text}")
    print(f"Translated : {translated}")

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
