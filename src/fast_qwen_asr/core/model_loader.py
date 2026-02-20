from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import numpy as np
import torch

from fast_qwen_asr.config.app_config import AppConfig
from fast_qwen_asr.core.runtime_state import RuntimeState
from fast_qwen_asr.utils.server import resolve_model_source

logger = logging.getLogger(__name__)

ALIGNER_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


try:
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
except ImportError:
    Qwen3ASRModel = None
    Qwen3ForcedAligner = None


def apply_offline_env(config: AppConfig) -> None:
    if config.models.force_offline or config.models.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


async def load_models_background(
    *,
    runtime: RuntimeState,
    config: AppConfig,
    stream_sample_rate: int,
) -> None:
    logger.info("Background task: Loading models")
    runtime.model_status = "loading_models"

    async def _load_asr() -> None:
        if not config.models.enable_asr_model:
            logger.info("ASR model disabled via config")
            return
        if Qwen3ASRModel is None:
            raise RuntimeError("qwen_asr not installed (Qwen3ASRModel missing).")

        model_source = resolve_model_source(
            local_path=config.models.asr_model_path,
            model_name=config.models.asr_model_name,
            local_files_only=config.models.local_files_only,
            model_label="ASR",
        )
        logger.info("Loading ASR model: %s", model_source)

        runtime.models["asr"] = await asyncio.to_thread(
            Qwen3ASRModel.LLM,
            model=model_source,
            gpu_memory_utilization=config.models.gpu_memory_utilization,
            max_new_tokens=config.models.max_new_tokens,
        )
        logger.info("ASR model loaded")

    async def _load_aligner() -> None:
        if not config.models.enable_aligner_model:
            logger.info("Aligner model disabled via config")
            return
        if Qwen3ForcedAligner is None:
            raise RuntimeError("qwen_asr not installed (Qwen3ForcedAligner missing).")

        aligner_source = resolve_model_source(
            local_path=config.models.aligner_model_path,
            model_name=config.models.aligner_model_name,
            local_files_only=config.models.local_files_only,
            model_label="Aligner",
        )
        logger.info("Loading aligner model: %s", aligner_source)

        runtime.models["aligner"] = await asyncio.to_thread(
            Qwen3ForcedAligner.from_pretrained,
            aligner_source,
            local_files_only=config.models.local_files_only,
            dtype=ALIGNER_DTYPE_MAP[config.models.aligner_dtype],
            device_map=config.models.aligner_device_map,
        )
        logger.info("Aligner model loaded")

    try:
        await asyncio.gather(_load_asr(), _load_aligner())
    except Exception:
        logger.exception("Model load failure")
        runtime.model_status = "error"
        runtime.model_ready_event.set()
        return

    if runtime.has_model("asr") and config.models.warmup_enabled:
        logger.info("Starting ASR warmup")
        runtime.model_status = "warming_up"
        try:
            await _warmup_asr(runtime=runtime, config=config, stream_sample_rate=stream_sample_rate)
            logger.info("Warmup complete")
        except Exception as exc:
            logger.warning("Warmup failed (non-critical): %s", exc)

    runtime.model_status = "ready"
    runtime.model_ready_event.set()
    logger.info("Server is ready to accept requests")


async def _warmup_asr(*, runtime: RuntimeState, config: AppConfig, stream_sample_rate: int) -> None:
    dummy_wav = np.zeros(config.models.warmup_dummy_samples, dtype=np.float32)

    async with runtime.infer_sem:
        await asyncio.to_thread(
            runtime.models["asr"].transcribe,
            audio=[(dummy_wav, stream_sample_rate)],
            language=[config.models.warmup_language],
            return_time_stamps=False,
        )

    async with runtime.infer_sem:
        state = await asyncio.to_thread(
            runtime.models["asr"].init_streaming_state,
            unfixed_chunk_num=config.streaming.unfixed_chunk_num,
            unfixed_token_num=config.streaming.unfixed_token_num,
            chunk_size_sec=config.streaming.chunk_size_sec,
        )

    warmup_chunks: list[int] = list(config.models.warmup_initial_chunks)
    warmup_chunks.extend([config.models.warmup_repeat_chunk] * config.models.warmup_repeat_count)

    for n_samples in warmup_chunks:
        async with runtime.infer_sem:
            await asyncio.to_thread(runtime.models["asr"].streaming_transcribe, dummy_wav[:n_samples], state)

    async with runtime.infer_sem:
        await asyncio.to_thread(runtime.models["asr"].finish_streaming_transcribe, state)
