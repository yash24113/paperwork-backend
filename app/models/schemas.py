from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ExtractedField(BaseModel):
    field_name: str
    field_value: str | None = None
    confidence: float | None = None


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    mime_type: str
    document_type: str | None = None
    summary: str | None = None
    status: DocumentStatus
    storage_path: str
    language: str = "en"
    created_at: datetime
    updated_at: datetime
    extracted_fields: list[ExtractedField] = Field(default_factory=list)


class DocumentListItem(BaseModel):
    id: str
    file_name: str
    document_type: str | None = None
    status: DocumentStatus
    created_at: datetime


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime | None = None


class ChatRequest(BaseModel):
    document_id: str
    question: str
    language: str | None = None


class ChatResponse(BaseModel):
    answer: str
    history: list[ChatMessage] = Field(default_factory=list)
