from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_LANGUAGE_NAME_BY_CODE = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "tr": "Turkish",
    "sv": "Swedish",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "hi": "Hindi",
    "ar": "Arabic",
}


def map_language(lang_code: Optional[str]) -> Optional[str]:
    """Map ISO language code to the language names expected by Qwen-ASR."""
    if lang_code is None:
        return None
    return _LANGUAGE_NAME_BY_CODE.get(lang_code.lower(), lang_code)


def validate_stream_start(client_sr: Optional[int], fmt: Optional[str], expected_sr: int) -> Optional[str]:
    """Return a websocket error message if the stream start message is invalid."""
    if client_sr != expected_sr or fmt not in (None, "pcm_s16le"):
        return f"Only pcm_s16le @ {expected_sr}Hz supported"
    return None


def should_emit_partial(*, last_partial_ts: float, now: float, partial_interval_ms: int) -> bool:
    """Return True when enough time elapsed to emit a partial transcript event."""
    return (now - last_partial_ts) * 1000.0 >= partial_interval_ms


def parse_json_object(raw_text: str) -> Optional[dict]:
    """Parse JSON text and return only object values; otherwise return None."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def resolve_model_source(
    *,
    local_path: Optional[str],
    model_name: str,
    local_files_only: bool,
    model_label: str,
) -> str:
    """
    Return local model path when configured; otherwise fallback to model name.
    In local_files_only mode, a missing local path is treated as fatal.
    """
    if local_path:
        normalized = str(Path(local_path).expanduser())
        if not Path(normalized).exists():
            raise RuntimeError(f"{model_label} model path does not exist: {normalized}")
        return normalized

    if local_files_only:
        raise RuntimeError(
            f"{model_label} local model path is required when local_files_only=true"
        )
    return model_name
