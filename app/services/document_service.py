import uuid
from datetime import datetime, timezone

from app.agents.graph import run_ingest, run_question
from app.core.config import get_settings
from app.core.languages import DEFAULT_LANGUAGE, language_name
from app.core.supabase_client import get_supabase_client
from app.models.schemas import (
    ChatMessage,
    ChatResponse,
    DocumentListItem,
    DocumentResponse,
    DocumentStatus,
    ExtractedField,
)


class DocumentNotFoundError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def upload_and_process_document(
    file_bytes: bytes,
    file_name: str,
    mime_type: str,
    owner_email: str | None,
    language: str | None = None,
) -> DocumentResponse:
    """Uploads the file to Supabase Storage, runs the agent graph, and persists results."""
    supabase = get_supabase_client()
    settings = get_settings()

    language_code = (language or DEFAULT_LANGUAGE).lower()

    document_id = str(uuid.uuid4())
    storage_path = f"{document_id}/{file_name}"

    supabase.storage.from_(settings.supabase_storage_bucket).upload(
        storage_path,
        file_bytes,
        {"content-type": mime_type},
    )

    insert_result = (
        supabase.table("documents")
        .insert(
            {
                "id": document_id,
                "owner_email": owner_email,
                "file_name": file_name,
                "storage_path": storage_path,
                "mime_type": mime_type,
                "status": DocumentStatus.PROCESSING.value,
                "language": language_code,
            }
        )
        .execute()
    )
    if not insert_result.data:
        raise RuntimeError("Failed to create document record")

    try:
        result_state = run_ingest(
            file_bytes=file_bytes,
            mime_type=mime_type,
            file_name=file_name,
            document_id=document_id,
            language=language_name(language_code),
        )
    except Exception as exc:  # noqa: BLE001
        supabase.table("documents").update(
            {"status": DocumentStatus.FAILED.value, "updated_at": _now()}
        ).eq("id", document_id).execute()
        raise RuntimeError(f"Agent pipeline failed: {exc}") from exc

    if result_state.get("error"):
        supabase.table("documents").update(
            {"status": DocumentStatus.FAILED.value, "updated_at": _now()}
        ).eq("id", document_id).execute()
        raise RuntimeError(result_state["error"])

    document_type = result_state.get("document_type", "Unknown Document")
    summary = result_state.get("summary", "")
    fields = result_state.get("extracted_fields", [])

    supabase.table("documents").update(
        {
            "document_type": document_type,
            "summary": summary,
            "status": DocumentStatus.READY.value,
            "updated_at": _now(),
        }
    ).eq("id", document_id).execute()

    if fields:
        supabase.table("extracted_fields").insert(
            [
                {
                    "document_id": document_id,
                    "field_name": field["field_name"],
                    "field_value": field.get("field_value"),
                    "confidence": field.get("confidence"),
                }
                for field in fields
            ]
        ).execute()

    return await get_document(document_id)


async def get_document(document_id: str) -> DocumentResponse:
    supabase = get_supabase_client()

    doc_result = supabase.table("documents").select("*").eq("id", document_id).execute()
    if not doc_result.data:
        raise DocumentNotFoundError(document_id)
    doc = doc_result.data[0]

    fields_result = (
        supabase.table("extracted_fields").select("*").eq("document_id", document_id).execute()
    )

    return DocumentResponse(
        id=doc["id"],
        file_name=doc["file_name"],
        mime_type=doc["mime_type"],
        document_type=doc.get("document_type"),
        summary=doc.get("summary"),
        status=DocumentStatus(doc["status"]),
        storage_path=doc["storage_path"],
        language=doc.get("language") or DEFAULT_LANGUAGE,
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        extracted_fields=[
            ExtractedField(
                field_name=f["field_name"],
                field_value=f.get("field_value"),
                confidence=f.get("confidence"),
            )
            for f in fields_result.data or []
        ],
    )


async def list_documents() -> list[DocumentListItem]:
    supabase = get_supabase_client()
    result = supabase.table("documents").select("*").order("created_at", desc=True).execute()
    return [
        DocumentListItem(
            id=doc["id"],
            file_name=doc["file_name"],
            document_type=doc.get("document_type"),
            status=DocumentStatus(doc["status"]),
            created_at=doc["created_at"],
        )
        for doc in result.data or []
    ]


async def ask_question(document_id: str, question: str, language: str | None = None) -> ChatResponse:
    supabase = get_supabase_client()
    settings = get_settings()

    document = await get_document(document_id)
    language_code = (language or document.language or DEFAULT_LANGUAGE).lower()

    file_bytes = supabase.storage.from_(settings.supabase_storage_bucket).download(
        document.storage_path
    )

    history_result = (
        supabase.table("chat_messages")
        .select("*")
        .eq("document_id", document_id)
        .order("created_at")
        .execute()
    )
    chat_history = [
        {"role": row["role"], "content": row["content"]} for row in history_result.data or []
    ]

    supabase.table("chat_messages").insert(
        {"document_id": document_id, "role": "user", "content": question}
    ).execute()

    result_state = run_question(
        file_bytes=file_bytes,
        mime_type=document.mime_type,
        document_id=document_id,
        document_type=document.document_type or "Unknown Document",
        extracted_fields=[f.model_dump() for f in document.extracted_fields],
        chat_history=chat_history,
        question=question,
        language=language_name(language_code),
    )

    if result_state.get("error"):
        raise RuntimeError(result_state["error"])

    answer = result_state.get("answer") or "I could not generate an answer for that question."

    supabase.table("chat_messages").insert(
        {"document_id": document_id, "role": "assistant", "content": answer}
    ).execute()

    full_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    return ChatResponse(
        answer=answer,
        history=[ChatMessage(role=turn["role"], content=turn["content"]) for turn in full_history],
    )
