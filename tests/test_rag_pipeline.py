"""
Test the RAG Pipeline ingestion workflow.
"""

from app.core.rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("Testing RAG Pipeline")
    print("=" * 60)

    pipeline = RAGPipeline()

    print("\nPipeline initialized successfully.\n")

    print("Current Stats:")
    print(pipeline.get_stats())

    print("\nStarting document ingestion...\n")

    result = pipeline.ingest_documents()

    print("Ingestion Result:")
    print(result)

    print("\nUpdated Stats:")
    print(pipeline.get_stats())

    print("\nResetting Vector Store...\n")

    pipeline.reset_database()

    print("Stats After Reset:")
    print(pipeline.get_stats())

    print("\n" + "=" * 60)
    print("RAG Pipeline Test Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()