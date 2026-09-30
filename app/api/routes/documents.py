from fastapi import APIRouter, File, HTTPException, UploadFile, Form

from app.models.schemas import DocumentListItem, DocumentResponse
from app.services.document_service import (
    DocumentNotFoundError,
    get_document,
    list_documents,
    upload_and_process_document,
)

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_MIME_PREFIXES = ("image/", "application/pdf")


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    owner_email: str | None = Form(default=None),
    language: str | None = Form(default=None),
) -> DocumentResponse:
    mime_type = file.content_type or "application/octet-stream"
    if not mime_type.startswith(_ALLOWED_MIME_PREFIXES):
        raise HTTPException(status_code=400, detail="Only image and PDF documents are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        return await upload_and_process_document(
            file_bytes=file_bytes,
            file_name=file.filename or "document",
            mime_type=mime_type,
            owner_email=owner_email,
            language=language,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentListItem])
async def list_all_documents() -> list[DocumentListItem]:
    return await list_documents()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_by_id(document_id: str) -> DocumentResponse:
    try:
        return await get_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
