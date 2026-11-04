from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def build_log_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_logging(
    *,
    level: str,
    log_file_path: Path,
    max_bytes: int,
    backup_count: int,
) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = build_log_formatter()

    log_dir = log_file_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)

    try:
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    except OSError as exc:
        root_logger.warning("File logging disabled for '%s': %s", log_file_path, exc)
        return

    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
