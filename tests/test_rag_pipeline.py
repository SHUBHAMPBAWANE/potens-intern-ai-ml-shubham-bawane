"""
Test the complete RAG Pipeline.
"""

from app.core.rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("Testing RAG Pipeline")
    print("=" * 60)

    pipeline = RAGPipeline()

    print("\nPipeline initialized successfully.\n")

    # --------------------------------------------------
    # 1. Current Stats
    # --------------------------------------------------

    print("Current Stats:")
    print(pipeline.get_stats())

    # --------------------------------------------------
    # 2. Ingestion
    # --------------------------------------------------

    print("\nStarting document ingestion...\n")

    result = pipeline.ingest_documents()

    print("Ingestion Result:")
    print(result)

    print("\nUpdated Stats:")
    print(pipeline.get_stats())

    # --------------------------------------------------
    # 3. English Question
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Testing English Question")
    print("=" * 60)

    english_result = pipeline.ask(
        "What is the total amount of the tactile push button switch?"
    )

    print("\nQuestion:")
    print(english_result["question"])

    print("\nLanguage:")
    print(english_result["language"])

    print("\nAnswer:")
    print(english_result["answer"])

    print("\nSources:")

    for source in english_result["sources"]:
        print(
            f"- {source['citation']} | "
            f"Similarity: {source['similarity_score']} | "
            f"Quality: {source['quality']}"
        )

    # --------------------------------------------------
    # 4. Marathi Question
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Testing Marathi Question")
    print("=" * 60)

    marathi_result = pipeline.ask(
        "टॅक्टाइल पुश बटन स्विचची एकूण रक्कम किती आहे?"
    )

    print("\nQuestion:")
    print(marathi_result["question"])

    print("\nLanguage:")
    print(marathi_result["language"])

    print("\nAnswer:")
    print(marathi_result["answer"])

    print("\nSources:")

    for source in marathi_result["sources"]:
        print(
            f"- {source['citation']} | "
            f"Similarity: {source['similarity_score']} | "
            f"Quality: {source['quality']}"
        )

    # --------------------------------------------------
    # 5. Hindi Question
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Testing Hindi Question")
    print("=" * 60)

    hindi_result = pipeline.ask(
        "टैक्टाइल पुश बटन स्विच की कुल राशि कितनी है?"
    )

    print("\nQuestion:")
    print(hindi_result["question"])

    print("\nLanguage:")
    print(hindi_result["language"])

    print("\nAnswer:")
    print(hindi_result["answer"])

    print("\nSources:")

    for source in hindi_result["sources"]:
        print(
            f"- {source['citation']} | "
            f"Similarity: {source['similarity_score']} | "
            f"Quality: {source['quality']}"
        )

    # --------------------------------------------------
    # 6. Reset
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Resetting Vector Store")
    print("=" * 60)

    pipeline.reset_database()

    print("\nStats After Reset:")
    print(pipeline.get_stats())

    print("\n" + "=" * 60)
    print("RAG Pipeline Test Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()