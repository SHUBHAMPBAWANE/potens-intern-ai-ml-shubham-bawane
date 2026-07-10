from langchain_core.documents import Document

from app.core.vector_store import VectorStoreService

print("=" * 50)
print("Initializing Vector Store...")
print("=" * 50)

db = VectorStoreService()

print("Vector count:", db.count())

doc = Document(
    page_content="This is a sample HR policy.",
    metadata={
        "source": "policy.pdf",
        "page": 1,
        "chunk": 1
    }
)

embedding = [[0.1] * 384]

ids = db.add_documents(
    documents=[doc],
    embeddings=embedding
)

print("Stored IDs:", ids)

print("Updated Count:", db.count())

print("=" * 50)
print("Test Successful!")
print("=" * 50)