import json
import logging
import random
import time
from functools import lru_cache

from google import genai
from google.genai import errors, types

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_JSON_BLOCK_MARKERS = ("```json", "```")
_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 2
_MAX_DELAY_SECONDS = 30


@lru_cache
def get_gemini_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


def _generate_content_with_retry(client: genai.Client, **kwargs):
    """Retries transient Gemini server errors (e.g. 503 overloaded) with capped, jittered backoff."""
    for attempt in range(_MAX_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = min(_BASE_DELAY_SECONDS * (2**attempt), _MAX_DELAY_SECONDS)
            delay += random.uniform(0, delay * 0.25)
            logger.warning(
                "Gemini server error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                _MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    for marker in _JSON_BLOCK_MARKERS:
        if cleaned.startswith(marker):
            cleaned = cleaned[len(marker) :]
        if cleaned.endswith(marker):
            cleaned = cleaned[: -len(marker)]
    return cleaned.strip()


def classify_and_extract(file_bytes: bytes, mime_type: str, language: str = "English") -> dict:
    """Single multimodal call: classifies the document and pulls structured fields.

    Returns a dict with keys: document_type, summary, fields (list of
    {field_name, field_value, confidence}).
    """
    client = get_gemini_client()
    settings = get_settings()

    prompt = (
        "You are Paperwork Buddy, an assistant that reads real-world documents "
        "(forms, bills, letters, IDs, receipts). Analyze the attached document and "
        "respond ONLY with strict JSON matching this schema:\n"
        "{\n"
        '  "document_type": string (e.g. "Utility Bill", "Passport", "Lease Agreement"),\n'
        '  "summary": string (2-3 plain-language sentences explaining what this document is '
        "and what, if anything, the reader needs to do),\n"
        '  "fields": [ {"field_name": string, "field_value": string, "confidence": number 0-1} ]\n'
        "}\n"
        "Extract every meaningful field a human would care about (names, dates, amounts, "
        "account numbers, addresses, due dates, statuses). Field values that are proper nouns, "
        "numbers, dates, or IDs should stay as written in the original document. Do not include "
        "markdown fences or any text outside the JSON object.\n"
        f"Write the document_type and summary in {language}, regardless of the document's own "
        "language."
    )

    response = _generate_content_with_retry(
        client,
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt,
        ],
    )

    raw_text = _clean_json_text(response.text or "{}")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {
            "document_type": "Unknown Document",
            "summary": raw_text[:500] if raw_text else "Could not analyze this document.",
            "fields": [],
        }

    parsed.setdefault("document_type", "Unknown Document")
    parsed.setdefault("summary", "")
    parsed.setdefault("fields", [])
    return parsed


def answer_question_about_document(
    file_bytes: bytes,
    mime_type: str,
    document_type: str,
    extracted_fields: list[dict],
    chat_history: list[dict[str, str]],
    question: str,
    language: str = "English",
) -> str:
    """Answers a natural-language question grounded in the original document + extracted fields."""
    client = get_gemini_client()
    settings = get_settings()

    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in chat_history[-10:])
    fields_text = "\n".join(
        f"- {f.get('field_name')}: {f.get('field_value')}" for f in extracted_fields
    )

    prompt = (
        "You are Paperwork Buddy, a helpful assistant answering questions about a specific "
        f"document (type: {document_type}). Use the attached document image/PDF as the "
        "source of truth, along with these pre-extracted fields:\n"
        f"{fields_text}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"User question: {question}\n\n"
        "Answer concisely, in plain language, in 1-4 sentences. If the document does not "
        "contain the answer, say so honestly instead of guessing. "
        f"Reply in {language}, regardless of what language the question or document is in."
    )

    response = _generate_content_with_retry(
        client,
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    return (response.text or "").strip() or "I could not find that information in this document."
