from functools import lru_cache

from langgraph.graph import END, StateGraph

from app.agents.nodes.answer_question import answer_question
from app.agents.nodes.classify_document import classify_document
from app.agents.nodes.extract_fields import extract_fields
from app.agents.nodes.summarize import summarize
from app.agents.state import AgentState


def _respond(state: AgentState) -> AgentState:
    """Terminal node: the state at this point already holds everything the API needs."""
    return state


def _route_after_extract(state: AgentState) -> str:
    """A question present in state means this run is a Q&A turn, not an initial ingest."""
    if state.get("error"):
        return "respond"
    return "answer_question" if state.get("question") else "summarize"


def _route_entry(state: AgentState) -> str:
    """Documents already classified (Q&A turns) skip straight to answering the question."""
    if state.get("document_type") and state.get("question"):
        return "answer_question"
    return "classify_document"


@lru_cache
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_document", classify_document)
    graph.add_node("extract_fields", extract_fields)
    graph.add_node("summarize", summarize)
    graph.add_node("answer_question", answer_question)
    graph.add_node("respond", _respond)

    graph.set_conditional_entry_point(
        _route_entry,
        {
            "classify_document": "classify_document",
            "answer_question": "answer_question",
        },
    )
    graph.add_edge("classify_document", "extract_fields")
    graph.add_conditional_edges(
        "extract_fields",
        _route_after_extract,
        {
            "summarize": "summarize",
            "answer_question": "answer_question",
            "respond": "respond",
        },
    )
    graph.add_edge("summarize", "respond")
    graph.add_edge("answer_question", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def run_ingest(
    file_bytes: bytes,
    mime_type: str,
    file_name: str,
    document_id: str,
    language: str = "English",
) -> AgentState:
    """Runs the full pipeline for a freshly uploaded document."""
    app_graph = build_graph()
    initial_state: AgentState = {
        "document_id": document_id,
        "file_bytes": file_bytes,
        "mime_type": mime_type,
        "file_name": file_name,
        "chat_history": [],
        "language": language,
    }
    return app_graph.invoke(initial_state)


def run_question(
    file_bytes: bytes,
    mime_type: str,
    document_id: str,
    document_type: str,
    extracted_fields: list[dict],
    chat_history: list[dict[str, str]],
    question: str,
    language: str = "English",
) -> AgentState:
    """Runs the pipeline in Q&A mode, skipping straight past summarize to answer_question."""
    app_graph = build_graph()
    initial_state: AgentState = {
        "document_id": document_id,
        "file_bytes": file_bytes,
        "mime_type": mime_type,
        "document_type": document_type,
        "extracted_fields": extracted_fields,
        "chat_history": chat_history,
        "question": question,
        "language": language,
    }
    return app_graph.invoke(initial_state)
