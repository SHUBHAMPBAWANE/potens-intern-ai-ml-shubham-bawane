from app.core.rag_pipeline import RAGPipeline


def main() -> None:

    print("=" * 60)
    print("Testing End-to-End RAG")
    print("=" * 60)

    pipeline = RAGPipeline(
        docs_directory="docs",
        top_k=3,
    )

    print("\nIngesting documents...\n")

    stats = pipeline.ingest_documents()

    print("Ingestion Result:")
    print(stats)

    question = (
        "What is the total amount of the tactile push button switch?"
    )

    print("\nQuestion:")
    print(question)

    print("\nAsking Gemini...\n")

    result = pipeline.ask(question)

    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:
        print(
            f"- {source['citation']} "
            f"| Similarity: {source['similarity_score']} "
            f"| Quality: {source['quality']}"
        )

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()