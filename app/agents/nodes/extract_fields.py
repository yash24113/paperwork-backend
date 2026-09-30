from app.agents.state import AgentState


def extract_fields(state: AgentState) -> AgentState:
    """Normalizes the raw fields produced by classify_document into a clean shape."""
    if state.get("error"):
        return state

    raw_fields = state.get("extracted_fields", [])
    normalized: list[dict[str, str | float | None]] = []
    for field in raw_fields:
        name = str(field.get("field_name", "")).strip()
        if not name:
            continue
        value = field.get("field_value")
        confidence = field.get("confidence")
        normalized.append(
            {
                "field_name": name,
                "field_value": str(value).strip() if value is not None else None,
                "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            }
        )

    return {**state, "extracted_fields": normalized}
