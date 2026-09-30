from app.agents.gemini_client import answer_question_about_document
from app.agents.state import AgentState


def answer_question(state: AgentState) -> AgentState:
    """Answers a user's question about the document, grounded in the file + extracted fields."""
    if state.get("error"):
        return state

    question = state.get("question")
    if not question:
        return {**state, "answer": None}

    try:
        answer = answer_question_about_document(
            file_bytes=state["file_bytes"],
            mime_type=state["mime_type"],
            document_type=state.get("document_type", "Unknown Document"),
            extracted_fields=state.get("extracted_fields", []),
            chat_history=state.get("chat_history", []),
            question=question,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller via state
        return {**state, "error": f"answer generation failed: {exc}"}

    return {**state, "answer": answer}
