SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ru": "Russian",
}

DEFAULT_LANGUAGE = "en"


def language_name(code: str | None) -> str:
    return SUPPORTED_LANGUAGES.get((code or DEFAULT_LANGUAGE).lower(), SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])
