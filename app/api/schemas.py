from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system",
    )


class SourceResponse(BaseModel):
    citation: str
    similarity_score: float
    quality: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]