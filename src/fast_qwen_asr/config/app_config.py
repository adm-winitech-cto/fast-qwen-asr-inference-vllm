from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# repo_root/config/default.json
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.json"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    log_level: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    directory: str
    file_name: str
    max_bytes: int
    backup_count: int


@dataclass(frozen=True)
class ConcurrencyConfig:
    max_concurrent_decode: int
    max_concurrent_infer: int
    threadpool_workers: Optional[int]
    batch_size: int


@dataclass(frozen=True)
class StreamingConfig:
    stream_min_samples: int
    partial_interval_ms: int
    stream_expect_sr: int
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float


@dataclass(frozen=True)
class ModelsConfig:
    enable_asr_model: bool
    enable_aligner_model: bool
    asr_model_name: str
    aligner_model_name: str
    asr_model_path: Optional[str]
    aligner_model_path: Optional[str]
    local_files_only: bool
    force_offline: bool
    gpu_memory_utilization: float
    max_new_tokens: int
    aligner_dtype: str
    aligner_device_map: str
    warmup_enabled: bool
    warmup_dummy_samples: int
    warmup_language: str
    warmup_initial_chunks: list[int]
    warmup_repeat_chunk: int
    warmup_repeat_count: int
    # vLLM context length for the ASR engine. None keeps the model default
    # (65536), whose single-request KV floor (~7 GiB) can fail engine start
    # when a GPU co-tenant grows memory during vLLM's startup profiling.
    max_model_len: Optional[int] = None


@dataclass(frozen=True)
class CorsConfig:
    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]


@dataclass(frozen=True)
class GrpcConfig:
    enabled: bool
    host: str
    port: int
    batch_wait_ms: int
    request_model_name: str
    infer_batching_mode: str


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig
    logging: LoggingConfig
    concurrency: ConcurrencyConfig
    streaming: StreamingConfig
    models: ModelsConfig
    cors: CorsConfig
    grpc: GrpcConfig


_LEGACY_ENV_PATHS = {
    "SERVER_HOST": ["server", "host"],
    "SERVER_PORT": ["server", "port"],
    "SERVER_LOG_LEVEL": ["server", "log_level"],
    "LOG_LEVEL": ["logging", "level"],
    "LOG_DIR": ["logging", "directory"],
    "LOG_FILE_NAME": ["logging", "file_name"],
    "LOG_MAX_BYTES": ["logging", "max_bytes"],
    "LOG_BACKUP_COUNT": ["logging", "backup_count"],
    "MAX_CONCURRENT_DECODE": ["concurrency", "max_concurrent_decode"],
    "MAX_CONCURRENT_INFER": ["concurrency", "max_concurrent_infer"],
    "THREADPOOL_WORKERS": ["concurrency", "threadpool_workers"],
    "BATCH_SIZE": ["concurrency", "batch_size"],
    "STREAM_MIN_SAMPLES": ["streaming", "stream_min_samples"],
    "PARTIAL_INTERVAL_MS": ["streaming", "partial_interval_ms"],
    "STREAM_EXPECT_SR": ["streaming", "stream_expect_sr"],
    "ENABLE_ASR_MODEL": ["models", "enable_asr_model"],
    "ENABLE_ALIGNER_MODEL": ["models", "enable_aligner_model"],
    "ASR_MODEL_NAME": ["models", "asr_model_name"],
    "ALIGNER_MODEL_NAME": ["models", "aligner_model_name"],
    "ASR_MODEL_PATH": ["models", "asr_model_path"],
    "ALIGNER_MODEL_PATH": ["models", "aligner_model_path"],
    "LOCAL_FILES_ONLY": ["models", "local_files_only"],
    "FORCE_OFFLINE": ["models", "force_offline"],
    "GPU_MEMORY_UTILIZATION": ["models", "gpu_memory_utilization"],
    "MAX_NEW_TOKENS": ["models", "max_new_tokens"],
    "GRPC_ENABLED": ["grpc", "enabled"],
    "GRPC_HOST": ["grpc", "host"],
    "GRPC_PORT": ["grpc", "port"],
    "GRPC_BATCH_WAIT_MS": ["grpc", "batch_wait_ms"],
    "GRPC_REQUEST_MODEL_NAME": ["grpc", "request_model_name"],
    "GRPC_INFER_BATCHING_MODE": ["grpc", "infer_batching_mode"],
}


_MISSING = object()


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read config file '{path}': {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config file '{path}' must contain a JSON object at top-level.")

    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def _get_nested(data: dict[str, Any], path: list[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return _MISSING
        cur = cur[key]
    return cur


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    cur: dict[str, Any] = data
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Invalid boolean value '{raw}'.")


def _coerce_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return _parse_bool(raw)
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"Expected integer value, got '{raw}'.") from exc
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"Expected float value, got '{raw}'.") from exc
    if isinstance(current, list):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Expected JSON array value, got '{raw}'.") from exc
        if not isinstance(loaded, list):
            raise ConfigError(f"Expected JSON array value, got '{raw}'.")
        return loaded
    if isinstance(current, dict):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Expected JSON object value, got '{raw}'.") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"Expected JSON object value, got '{raw}'.")
        return loaded

    stripped = raw.strip()
    if stripped.lower() in {"true", "false"}:
        return _parse_bool(stripped)
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return raw


def _apply_prefixed_env_overrides(data: dict[str, Any]) -> None:
    prefix = "APP_CFG__"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        segments = env_key[len(prefix):].split("__")
        path = [segment.strip().lower() for segment in segments if segment.strip()]
        if not path:
            continue

        current = _get_nested(data, path)
        if current is _MISSING:
            joined = ".".join(path)
            raise ConfigError(f"Unknown config path for override '{env_key}' ({joined}).")

        _set_nested(data, path, _coerce_value(env_value, current))


def _apply_legacy_env_overrides(data: dict[str, Any]) -> None:
    for env_key, path in _LEGACY_ENV_PATHS.items():
        if env_key not in os.environ:
            continue

        current = _get_nested(data, path)
        if current is _MISSING:
            joined = ".".join(path)
            raise ConfigError(f"Legacy override mapping points to unknown config path: {joined}")

        _set_nested(data, path, _coerce_value(os.environ[env_key], current))


def _require_int(name: str, value: Any, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    if not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {type(value).__name__}.")
    if min_value is not None and value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}.")
    if max_value is not None and value > max_value:
        raise ConfigError(f"{name} must be <= {max_value}.")
    return value


def _require_float(name: str, value: Any, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    if not isinstance(value, (float, int)):
        raise ConfigError(f"{name} must be a number, got {type(value).__name__}.")
    val = float(value)
    if min_value is not None and val < min_value:
        raise ConfigError(f"{name} must be >= {min_value}.")
    if max_value is not None and val > max_value:
        raise ConfigError(f"{name} must be <= {max_value}.")
    return val


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string.")
    return value


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    raise ConfigError("Optional path values must be string or null.")


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean.")
    return value


def _require_list_of_str(name: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{name} must be a list of strings.")
    return value


def _build_config(data: dict[str, Any]) -> AppConfig:
    try:
        server_raw = data["server"]
        concurrency_raw = data["concurrency"]
        streaming_raw = data["streaming"]
        models_raw = data["models"]
        cors_raw = data["cors"]
    except KeyError as exc:
        raise ConfigError(f"Missing required config section: {exc}") from exc

    grpc_raw = data.get("grpc", {})
    if not isinstance(grpc_raw, dict):
        raise ConfigError("grpc section must be an object when provided.")

    logging_raw = data.get("logging", {})
    if not isinstance(logging_raw, dict):
        raise ConfigError("logging section must be an object when provided.")

    threadpool_workers = concurrency_raw.get("threadpool_workers")
    if threadpool_workers is not None:
        threadpool_workers = _require_int("concurrency.threadpool_workers", threadpool_workers, min_value=1)

    aligner_dtype = _require_str("models.aligner_dtype", models_raw["aligner_dtype"]).lower()
    if aligner_dtype not in {"bfloat16", "float16", "float32"}:
        raise ConfigError("models.aligner_dtype must be one of: bfloat16, float16, float32")

    warmup_initial_chunks = models_raw["warmup_initial_chunks"]
    if not isinstance(warmup_initial_chunks, list) or not all(isinstance(v, int) and v > 0 for v in warmup_initial_chunks):
        raise ConfigError("models.warmup_initial_chunks must be a list of positive integers")

    server_log_level = _require_str("server.log_level", server_raw["log_level"])

    logging_cfg = LoggingConfig(
        level=_require_str("logging.level", logging_raw.get("level", server_log_level)),
        directory=_require_str("logging.directory", logging_raw.get("directory", "logs")),
        file_name=_require_str("logging.file_name", logging_raw.get("file_name", "app.log")),
        max_bytes=_require_int("logging.max_bytes", logging_raw.get("max_bytes", 10 * 1024 * 1024), min_value=1024),
        backup_count=_require_int("logging.backup_count", logging_raw.get("backup_count", 3), min_value=1),
    )

    cfg = AppConfig(
        server=ServerConfig(
            host=_require_str("server.host", server_raw["host"]),
            port=_require_int("server.port", server_raw["port"], min_value=1, max_value=65535),
            log_level=server_log_level,
        ),
        logging=logging_cfg,
        concurrency=ConcurrencyConfig(
            max_concurrent_decode=_require_int("concurrency.max_concurrent_decode", concurrency_raw["max_concurrent_decode"], min_value=1),
            max_concurrent_infer=_require_int("concurrency.max_concurrent_infer", concurrency_raw["max_concurrent_infer"], min_value=1),
            threadpool_workers=threadpool_workers,
            batch_size=_require_int("concurrency.batch_size", concurrency_raw.get("batch_size", 1), min_value=1),
        ),
        streaming=StreamingConfig(
            stream_min_samples=_require_int("streaming.stream_min_samples", streaming_raw["stream_min_samples"], min_value=1),
            partial_interval_ms=_require_int("streaming.partial_interval_ms", streaming_raw["partial_interval_ms"], min_value=1),
            stream_expect_sr=_require_int("streaming.stream_expect_sr", streaming_raw["stream_expect_sr"], min_value=1),
            unfixed_chunk_num=_require_int("streaming.unfixed_chunk_num", streaming_raw["unfixed_chunk_num"], min_value=1),
            unfixed_token_num=_require_int("streaming.unfixed_token_num", streaming_raw["unfixed_token_num"], min_value=1),
            chunk_size_sec=_require_float("streaming.chunk_size_sec", streaming_raw["chunk_size_sec"], min_value=0.1),
        ),
        models=ModelsConfig(
            enable_asr_model=_require_bool("models.enable_asr_model", models_raw["enable_asr_model"]),
            enable_aligner_model=_require_bool("models.enable_aligner_model", models_raw["enable_aligner_model"]),
            asr_model_name=_require_str("models.asr_model_name", models_raw["asr_model_name"]),
            aligner_model_name=_require_str("models.aligner_model_name", models_raw["aligner_model_name"]),
            asr_model_path=_optional_str(models_raw.get("asr_model_path")),
            aligner_model_path=_optional_str(models_raw.get("aligner_model_path")),
            local_files_only=_require_bool("models.local_files_only", models_raw.get("local_files_only", False)),
            force_offline=_require_bool("models.force_offline", models_raw.get("force_offline", False)),
            gpu_memory_utilization=_require_float("models.gpu_memory_utilization", models_raw["gpu_memory_utilization"], min_value=0.01, max_value=1.0),
            max_new_tokens=_require_int("models.max_new_tokens", models_raw["max_new_tokens"], min_value=1),
            aligner_dtype=aligner_dtype,
            aligner_device_map=_require_str("models.aligner_device_map", models_raw["aligner_device_map"]),
            warmup_enabled=_require_bool("models.warmup_enabled", models_raw["warmup_enabled"]),
            warmup_dummy_samples=_require_int("models.warmup_dummy_samples", models_raw["warmup_dummy_samples"], min_value=1),
            warmup_language=_require_str("models.warmup_language", models_raw["warmup_language"]),
            warmup_initial_chunks=warmup_initial_chunks,
            warmup_repeat_chunk=_require_int("models.warmup_repeat_chunk", models_raw["warmup_repeat_chunk"], min_value=1),
            warmup_repeat_count=_require_int("models.warmup_repeat_count", models_raw["warmup_repeat_count"], min_value=0),
            max_model_len=(
                None
                if models_raw.get("max_model_len") is None
                else _require_int("models.max_model_len", models_raw["max_model_len"], min_value=1024)
            ),
        ),
        cors=CorsConfig(
            allow_origins=_require_list_of_str("cors.allow_origins", cors_raw["allow_origins"]),
            allow_credentials=_require_bool("cors.allow_credentials", cors_raw["allow_credentials"]),
            allow_methods=_require_list_of_str("cors.allow_methods", cors_raw["allow_methods"]),
            allow_headers=_require_list_of_str("cors.allow_headers", cors_raw["allow_headers"]),
        ),
        grpc=GrpcConfig(
            enabled=_require_bool("grpc.enabled", grpc_raw.get("enabled", False)),
            host=_require_str("grpc.host", grpc_raw.get("host", "0.0.0.0")),
            port=_require_int("grpc.port", grpc_raw.get("port", 50051), min_value=1, max_value=65535),
            batch_wait_ms=_require_int("grpc.batch_wait_ms", grpc_raw.get("batch_wait_ms", 3), min_value=0, max_value=100),
            request_model_name=_require_str("grpc.request_model_name", grpc_raw.get("request_model_name", "asr")),
            infer_batching_mode=_require_str("grpc.infer_batching_mode", grpc_raw.get("infer_batching_mode", "dynamic")).lower(),
        ),
    )

    if cfg.grpc.infer_batching_mode not in {"dynamic", "inflight"}:
        raise ConfigError("grpc.infer_batching_mode must be one of: dynamic, inflight")

    return cfg


def _resolve_external_path(config_path: Optional[str]) -> Optional[Path]:
    if config_path:
        return Path(config_path).expanduser()

    env_path = os.getenv("APP_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()

    return None


def load_app_config(config_path: Optional[str] = None) -> AppConfig:
    base = _load_json_file(DEFAULT_CONFIG_PATH)

    external_path = _resolve_external_path(config_path)
    if external_path is not None:
        external_abs = external_path.resolve()
        default_abs = DEFAULT_CONFIG_PATH.resolve()
        if external_abs != default_abs:
            if external_path.exists():
                override = _load_json_file(external_path)
                base = _deep_merge(base, override)
            else:
                logger.warning(
                    "APP_CONFIG_PATH points to '%s', but file does not exist. Falling back to defaults.",
                    external_path,
                )

    _apply_legacy_env_overrides(base)
    _apply_prefixed_env_overrides(base)

    return _build_config(base)
