from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="RAG Document Question Answering API",
    description="A RAG API using ChromaDB, BGE embeddings and Gemini.",
    version="1.0.0",
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "RAG API is running successfully.",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }