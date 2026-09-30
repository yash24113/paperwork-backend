from app.agents.state import AgentState


def summarize(state: AgentState) -> AgentState:
    """Ensures a human-readable summary always exists, even if Gemini omitted one."""
    if state.get("error"):
        return state

    summary = state.get("summary") or ""
    if not summary:
        document_type = state.get("document_type", "document")
        summary = f"This appears to be a {document_type.lower()}. No further summary was available."

    return {**state, "summary": summary}
