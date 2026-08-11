from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

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


# ============================================================
# Helper: Detect Gemini/API Errors
# ============================================================

def _get_api_error_status(exc: Exception) -> int | None:
    """
    Detect common Gemini/API failures and return
    the appropriate HTTP status code.
    """

    error_text = str(exc).lower()

    # --------------------------------------------------------
    # API quota / rate limit
    # --------------------------------------------------------

    quota_keywords = [
        "quota",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
    ]

    if any(
        keyword in error_text
        for keyword in quota_keywords
    ):
        return 429

    # --------------------------------------------------------
    # Temporary upstream/API service failure
    # --------------------------------------------------------

    service_keywords = [
        "503",
        "service unavailable",
        "temporarily unavailable",
        "unavailable",
    ]

    if any(
        keyword in error_text
        for keyword in service_keywords
    ):
        return 503

    return None


# ============================================================
# Ask Question
# ============================================================

@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    request: QuestionRequest,
) -> AskResponse:

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

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

        # ----------------------------------------------------
        # Check for Gemini quota / API errors
        # ----------------------------------------------------

        api_error_status = _get_api_error_status(exc)

        if api_error_status == 429:

            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota or rate limit exceeded. "
                    "Please wait and try again later."
                ),
            ) from exc

        if api_error_status == 503:

            raise HTTPException(
                status_code=503,
                detail=(
                    "The Gemini API is temporarily unavailable. "
                    "Please try again later."
                ),
            ) from exc

        # ----------------------------------------------------
        # General server error
        # ----------------------------------------------------

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while processing "
                "your question."
            ),
        ) from exc


# ============================================================
# Upload Document
# ============================================================

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a document and add it to the RAG knowledge base.
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    allowed_extensions = {
        ".pdf",
        ".txt",
        ".docx",
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload PDF, TXT, or DOCX files."
            ),
        )

    # --------------------------------------------------------
    # Save file
    # --------------------------------------------------------

    docs_directory = Path("docs")

    docs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(
        file.filename
    ).name

    file_path = (
        docs_directory / safe_filename
    )

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        with open(
            file_path,
            "wb",
        ) as destination:

            destination.write(
                contents
            )

        # ----------------------------------------------------
        # Ingest documents
        # ----------------------------------------------------

        ingestion_result = (
            pipeline.ingest_documents()
        )

        return {
            "status": "success",
            "message": (
                f"Document '{safe_filename}' "
                "uploaded and processed successfully."
            ),
            "filename": safe_filename,
            "ingestion": ingestion_result,
        }

    except HTTPException:
        # Do not convert our own 400 errors into 500 errors.
        if file_path.exists():
            file_path.unlink()

        raise

    except Exception as exc:

        # ----------------------------------------------------
        # Check API quota/service errors
        # ----------------------------------------------------

        api_error_status = _get_api_error_status(exc)

        # Remove partially processed file
        if file_path.exists():
            file_path.unlink()

        if api_error_status == 429:

            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota or rate limit exceeded "
                    "while processing the document. "
                    "Please wait and try again later."
                ),
            ) from exc

        if api_error_status == 503:

            raise HTTPException(
                status_code=503,
                detail=(
                    "The Gemini API is temporarily unavailable. "
                    "Please try again later."
                ),
            ) from exc

        # ----------------------------------------------------
        # General document processing error
        # ----------------------------------------------------

        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed. "
                "Please check the file and try again."
            ),
        ) from exc