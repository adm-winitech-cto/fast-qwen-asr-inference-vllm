from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence, TypeVar

T = TypeVar("T")


def split_into_batches(items: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    """Yield input items in contiguous chunks of size <= batch_size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def _normalized_aliases(model_name: str, model_path: Optional[str]) -> set[str]:
    aliases: set[str] = set()

    normalized_name = model_name.strip()
    if normalized_name:
        aliases.add(normalized_name.lower())
        aliases.add(normalized_name.split("/")[-1].lower())

    if model_path:
        path = Path(model_path).expanduser()
        aliases.add(str(path).lower())
        aliases.add(path.name.lower())

    return aliases


def is_supported_model_name(
    requested_model_name: str,
    configured_model_name: str,
    configured_model_path: Optional[str],
) -> bool:
    """
    Return True when the requested model is empty or points to the configured ASR model.
    Accepts full repo name, local full path, and the local path basename.
    """
    requested = requested_model_name.strip()
    if not requested:
        return True

    aliases = _normalized_aliases(configured_model_name, configured_model_path)
    return requested.lower() in aliases
