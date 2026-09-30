from app.agents.gemini_client import classify_and_extract
from app.agents.state import AgentState


def classify_document(state: AgentState) -> AgentState:
    """Runs the multimodal Gemini call and stashes the raw result for downstream nodes.

    Classification and field extraction happen in one Gemini call for latency/cost
    reasons; this node owns document_type, extract_fields owns shaping the fields list.
    """
    try:
        result = classify_and_extract(state["file_bytes"], state["mime_type"])
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller via state
        return {**state, "error": f"classification failed: {exc}"}

    return {
        **state,
        "document_type": result["document_type"],
        "summary": result["summary"],
        "extracted_fields": result["fields"],
    }
