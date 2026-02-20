from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from fast_qwen_asr.config.app_config import ConfigError, load_app_config
from fast_qwen_asr.core.model_loader import apply_offline_env, load_models_background
from fast_qwen_asr.core.runtime_state import RuntimeState
from fast_qwen_asr.logging.setup import configure_logging
from fast_qwen_asr.services.audio_decoder import read_audio_file
from fast_qwen_asr.utils.inference import split_into_batches
from fast_qwen_asr.utils.server import (
    map_language,
    parse_json_object,
    should_emit_partial,
    validate_stream_start,
)

try:
    APP_CONFIG = load_app_config()
except ConfigError as exc:
    raise RuntimeError(f"Invalid application config: {exc}") from exc

configure_logging(
    level=APP_CONFIG.logging.level,
    log_file_path=Path(APP_CONFIG.logging.directory) / APP_CONFIG.logging.file_name,
    max_bytes=APP_CONFIG.logging.max_bytes,
    backup_count=APP_CONFIG.logging.backup_count,
)
logger = logging.getLogger(__name__)

apply_offline_env(APP_CONFIG)

THREADPOOL_WORKERS = APP_CONFIG.concurrency.threadpool_workers or ((os.cpu_count() or 4) * 5)
runtime = RuntimeState(
    max_concurrent_decode=APP_CONFIG.concurrency.max_concurrent_decode,
    max_concurrent_infer=APP_CONFIG.concurrency.max_concurrent_infer,
)


async def to_thread_limited(sem: asyncio.Semaphore, fn, *args, **kwargs):
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Qwen3-ASR server")

    executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)
    app.state.executor = executor
    asyncio.get_running_loop().set_default_executor(executor)

    task = asyncio.create_task(
        load_models_background(
            runtime=runtime,
            config=APP_CONFIG,
            stream_sample_rate=APP_CONFIG.streaming.stream_expect_sr,
        )
    )

    grpc_server = None
    if APP_CONFIG.grpc.enabled:
        from fast_qwen_asr.api.grpc_runtime import start_grpc_server

        grpc_server = await start_grpc_server(
            runtime_state=runtime,
            config=APP_CONFIG,
            stream_sample_rate=APP_CONFIG.streaming.stream_expect_sr,
        )

    try:
        yield
    finally:
        task.cancel()
        if grpc_server is not None:
            grpc_servicer = getattr(grpc_server, "_stt_servicer", None)
            if grpc_servicer is not None and hasattr(grpc_servicer, "aclose"):
                await grpc_servicer.aclose()
            await grpc_server.stop(grace=3)

        runtime.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=APP_CONFIG.cors.allow_origins,
        allow_credentials=APP_CONFIG.cors.allow_credentials,
        allow_methods=APP_CONFIG.cors.allow_methods,
        allow_headers=APP_CONFIG.cors.allow_headers,
    )

    @app.get("/health")
    async def health():
        mem = psutil.virtual_memory()
        info = {
            "status": runtime.model_status,
            "limits": {
                "max_concurrent_decode": APP_CONFIG.concurrency.max_concurrent_decode,
                "max_concurrent_infer": APP_CONFIG.concurrency.max_concurrent_infer,
                "batch_size": APP_CONFIG.concurrency.batch_size,
                "threadpool_workers": THREADPOOL_WORKERS,
            },
            "grpc": {
                "enabled": APP_CONFIG.grpc.enabled,
                "host": APP_CONFIG.grpc.host,
                "port": APP_CONFIG.grpc.port,
            },
            "memory": {
                "ram_total_mb": mem.total // (1024 * 1024),
                "ram_available_mb": mem.available // (1024 * 1024),
                "ram_percent": mem.percent,
            },
        }
        if torch.cuda.is_available():
            info["memory"]["gpu_allocated_mb"] = torch.cuda.memory_allocated() // (1024 * 1024)
            info["memory"]["gpu_reserved_mb"] = torch.cuda.memory_reserved() // (1024 * 1024)
        return info

    @app.post("/transcribe")
    async def transcribe(
        files: list[UploadFile] = File(...),
        language: Optional[str] = Query(None, description="Language code, e.g. en/de/fr"),
        forced_alignment: bool = Query(False, description="Enable forced alignment"),
    ):
        await runtime.model_ready_event.wait()

        if runtime.model_status != "ready":
            raise HTTPException(status_code=503, detail=f"Server not ready: {runtime.model_status}")
        if not runtime.has_model("asr"):
            raise HTTPException(status_code=503, detail="ASR model is not enabled or failed to load.")

        full_lang = map_language(language)

        async def decode_one(upload: UploadFile):
            content = await upload.read()
            return await to_thread_limited(runtime.decode_sem, read_audio_file, content)

        try:
            audio_batch = await asyncio.gather(*(decode_one(item) for item in files))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {exc}") from exc

        try:
            results = []
            for chunk in split_into_batches(audio_batch, APP_CONFIG.concurrency.batch_size):
                async with runtime.infer_sem:
                    chunk_results = await asyncio.to_thread(
                        runtime.models["asr"].transcribe,
                        audio=chunk,
                        language=[full_lang] * len(chunk),
                        return_time_stamps=False,
                    )
                results.extend(chunk_results)

            if not forced_alignment:
                return [{"text": res.text, "language": res.language} for res in results]

            if not runtime.has_model("aligner"):
                raise HTTPException(status_code=503, detail="Aligner model is not enabled or failed to load.")

            texts = [result.text for result in results]
            alignment_results = []
            for audio_chunk, text_chunk in zip(
                split_into_batches(audio_batch, APP_CONFIG.concurrency.batch_size),
                split_into_batches(texts, APP_CONFIG.concurrency.batch_size),
            ):
                async with runtime.infer_sem:
                    aligned_chunk = await asyncio.to_thread(
                        runtime.models["aligner"].align,
                        audio=audio_chunk,
                        text=text_chunk,
                        language=[full_lang] * len(audio_chunk),
                    )
                alignment_results.extend(aligned_chunk)

            response = []
            for idx, res in enumerate(results):
                response.append(
                    {
                        "text": res.text,
                        "language": res.language,
                        "timestamps": alignment_results[idx],
                    }
                )
            return response
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Inference failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.websocket("/transcribe-streaming")
    async def websocket_endpoint(
        ws: WebSocket,
        language: Optional[str] = Query(None),
        forced_alignment: bool = Query(False),
    ):
        _ = forced_alignment
        await ws.accept()

        await runtime.model_ready_event.wait()
        if runtime.model_status != "ready" or not runtime.has_model("asr"):
            await ws.close(code=1011, reason=f"Server not ready: {runtime.model_status}")
            return

        full_lang = map_language(language)
        started = False

        try:
            async with runtime.infer_sem:
                state = await asyncio.to_thread(
                    runtime.models["asr"].init_streaming_state,
                    unfixed_chunk_num=APP_CONFIG.streaming.unfixed_chunk_num,
                    unfixed_token_num=APP_CONFIG.streaming.unfixed_token_num,
                    chunk_size_sec=APP_CONFIG.streaming.chunk_size_sec,
                )
        except Exception as exc:
            logger.exception("Failed to init streaming state: %s", exc)
            await ws.close(code=1011, reason="init_streaming_state failed")
            return

        try:
            await ws.send_json({"type": "ready"})
        except Exception:
            return

        buf_parts: list[np.ndarray] = []
        buf_n = 0
        last_partial_ts = 0.0

        async def flush_and_infer(send_partial: bool):
            nonlocal buf_n, last_partial_ts
            if buf_n <= 0:
                return

            chunk = np.concatenate(buf_parts, axis=0) if len(buf_parts) > 1 else buf_parts[0]
            async with runtime.infer_sem:
                await asyncio.to_thread(runtime.models["asr"].streaming_transcribe, chunk, state)

            if send_partial:
                now = time.monotonic()
                if should_emit_partial(
                    last_partial_ts=last_partial_ts,
                    now=now,
                    partial_interval_ms=APP_CONFIG.streaming.partial_interval_ms,
                ):
                    await ws.send_json({"type": "partial", "text": state.text, "language": state.language})
                    last_partial_ts = now

        try:
            while True:
                msg = await ws.receive()

                if msg["type"] == "websocket.disconnect":
                    break
                if msg["type"] != "websocket.receive":
                    continue

                if msg.get("text"):
                    data = parse_json_object(msg["text"])
                    if isinstance(data, dict):
                        msg_type = data.get("type")

                        if msg_type == "start":
                            started = True
                            client_sr = int(data.get("sample_rate_hz", 0)) if data.get("sample_rate_hz") else None
                            fmt = data.get("format")

                            validation_error = validate_stream_start(
                                client_sr,
                                fmt,
                                APP_CONFIG.streaming.stream_expect_sr,
                            )
                            if validation_error is not None:
                                await ws.send_json({"type": "error", "message": validation_error})
                                await ws.close(code=1003)
                                return

                            if full_lang is not None:
                                await ws.send_json({"type": "info", "message": f"language={full_lang}"})
                            continue

                        if msg_type == "stop":
                            await flush_and_infer(send_partial=False)
                            async with runtime.infer_sem:
                                await asyncio.to_thread(runtime.models["asr"].finish_streaming_transcribe, state)

                            await ws.send_json({"type": "final", "text": state.text, "language": state.language})
                            await ws.close(code=1000)
                            return

                if msg.get("bytes"):
                    if not started:
                        await ws.send_json(
                            {
                                "type": "error",
                                "message": "Send {type:'start', format:'pcm_s16le', sample_rate_hz:16000} first",
                            }
                        )
                        await ws.close(code=1002)
                        return

                    chunk_bytes = msg["bytes"]
                    audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                    if audio_int16.size == 0:
                        continue

                    audio_f32 = audio_int16.astype(np.float32) / 32768.0
                    buf_parts.append(audio_f32)
                    buf_n += audio_f32.size

                    if buf_n >= APP_CONFIG.streaming.stream_min_samples:
                        await flush_and_infer(send_partial=True)

        except WebSocketDisconnect:
            return
        except Exception as exc:
            logger.exception("WebSocket error: %s", exc)
            try:
                await ws.close(code=1011, reason="internal error")
            except Exception:
                return

    return app


app = create_app()
