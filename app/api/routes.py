from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    QuestionRequest,
    QuestionResponse,
)
from app.core.rag_pipeline import RAGPipeline


router = APIRouter(
    prefix="/api",
    tags=["RAG"],
)


pipeline = RAGPipeline(
    docs_directory="docs",
    top_k=3,
)


@router.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest,
) -> QuestionResponse:

    try:
        result = pipeline.ask(
            request.question
        )

        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc