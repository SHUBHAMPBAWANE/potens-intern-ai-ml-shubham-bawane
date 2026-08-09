from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    QuestionRequest,
    AskResponse,
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
    response_model=AskResponse,
)
def ask_question(
    request: QuestionRequest,
) -> AskResponse:

    try:
        result = pipeline.ask(
            request.question
        )

        return AskResponse(
            question=result["question"],
            language=result["language"],
            answer=result["answer"],
            sources=result["sources"],
            contradiction=result["contradiction"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc