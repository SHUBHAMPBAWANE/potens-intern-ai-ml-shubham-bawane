"""
Test RetrieverService.
"""

from app.core.embeddings import EmbeddingService
from app.core.rag_pipeline import RAGPipeline
from app.core.retriever_service import RetrieverService


def main():

    print("=" * 60)
    print("Testing Retriever Service")
    print("=" * 60)

    pipeline = RAGPipeline()

    print("\nIngesting documents...\n")

    stats = pipeline.ingest_documents()

    print(stats)

    embedding_service = EmbeddingService()

    retriever = RetrieverService(
        pipeline.vector_store
    )

    question = "What is this document about?"

    query_embedding = embedding_service.embed_query(
        question
    )

    results = retriever.retrieve(
        query_embedding=query_embedding,
        top_k=3,
    )

    print("\nRetrieved Results")
    print("-" * 60)

    if not results:
        print("No documents retrieved.")
        return

    for index, result in enumerate(results, start=1):

        print(f"\nResult #{index}")

        print(
            f"Similarity : {result.similarity_score:.3f}"
        )

        print(
            f"Quality    : {retriever.similarity_label(result.similarity_score)}"
        )

        print(
            f"Source: {result.document.metadata.get('source')}"
        )

        print(
            f"Page: {result.document.metadata.get('page')}"
        )

        print("\nPreview:")

        print(
            result.document.page_content[:250]
        )

        print("-" * 60)


if __name__ == "__main__":
    main()