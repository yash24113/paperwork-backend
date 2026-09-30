from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state threaded through every node of the LangGraph document graph."""

    document_id: str
    file_bytes: bytes
    mime_type: str
    file_name: str
    language: str

    document_type: str
    extracted_fields: list[dict[str, str | float | None]]
    summary: str

    question: str | None
    chat_history: list[dict[str, str]]
    answer: str | None

    error: str | None
