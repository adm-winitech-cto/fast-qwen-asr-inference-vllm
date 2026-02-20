from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Optional

from fast_qwen_asr.config.app_config import AppConfig
from fast_qwen_asr.core.runtime_state import RuntimeState
from fast_qwen_asr.utils.inference import is_supported_model_name, split_into_batches
from fast_qwen_asr.utils.server import map_language
from stt_grpc_pb2 import BatchInferItem, BatchInferResponse, HealthCheckResponse, InferResponse
import stt_grpc_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass
class _PendingInferJob:
    audio: "np.ndarray"
    language: Optional[str]
    future: asyncio.Future[BatchInferItem]


class _GrpcInferBatcherBase:
    def __init__(
        self,
        *,
        runtime_state: RuntimeState,
        stream_sample_rate: int,
        max_batch_size: int,
    ):
        self._runtime = runtime_state
        self._stream_sample_rate = stream_sample_rate
        self._max_batch_size = max_batch_size
        self._pending: deque[_PendingInferJob] = deque()
        self._cv = asyncio.Condition()
        self._closed = False
        self._task = asyncio.create_task(self._run_loop(), name="grpc-infer-batcher")

    async def submit(self, *, audio: "np.ndarray", language: Optional[str]) -> BatchInferItem:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[BatchInferItem] = loop.create_future()
        job = _PendingInferJob(audio=audio, language=language, future=future)
        async with self._cv:
            self._pending.append(job)
            self._cv.notify()
        return await future

    async def aclose(self) -> None:
        async with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task

    async def _run_loop(self) -> None:
        raise NotImplementedError

    async def _process_jobs(self, jobs: list[_PendingInferJob]) -> None:
        if not jobs:
            return

        if not self._runtime.has_model("asr"):
            for job in jobs:
                if not job.future.done():
                    job.future.set_result(_error_item("ASR model is not enabled or failed to load."))
            return

        audio_batch = [(job.audio, self._stream_sample_rate) for job in jobs]
        language_batch = [job.language for job in jobs]

        try:
            async with self._runtime.infer_sem:
                results = await asyncio.to_thread(
                    self._runtime.models["asr"].transcribe,
                    audio=audio_batch,
                    language=language_batch,
                    return_time_stamps=False,
                )
        except Exception as exc:
            logger.exception("gRPC batch worker failed: %s", exc)
            for job in jobs:
                if not job.future.done():
                    job.future.set_result(_error_item(str(exc)))
            return

        if len(results) != len(jobs):
            msg = f"Unexpected result size: expected={len(jobs)} got={len(results)}"
            for job in jobs:
                if not job.future.done():
                    job.future.set_result(_error_item(msg))
            return

        for job, result in zip(jobs, results):
            if not job.future.done():
                job.future.set_result(BatchInferItem(text=result.text, success=True, error=""))


class _DynamicGrpcInferBatcher(_GrpcInferBatcherBase):
    def __init__(
        self,
        *,
        runtime_state: RuntimeState,
        stream_sample_rate: int,
        max_batch_size: int,
        batch_wait_ms: int,
    ):
        self._batch_wait_ms = batch_wait_ms
        super().__init__(
            runtime_state=runtime_state,
            stream_sample_rate=stream_sample_rate,
            max_batch_size=max_batch_size,
        )

    async def _run_loop(self) -> None:
        while True:
            async with self._cv:
                while not self._pending and not self._closed:
                    await self._cv.wait()
                if self._closed and not self._pending:
                    return

            if self._batch_wait_ms > 0:
                await asyncio.sleep(self._batch_wait_ms / 1000.0)

            async with self._cv:
                if not self._pending:
                    continue
                batch_n = min(len(self._pending), self._max_batch_size)
                jobs = [self._pending.popleft() for _ in range(batch_n)]

            await self._process_jobs(jobs)


class _InFlightGrpcInferBatcher(_GrpcInferBatcherBase):
    async def _run_loop(self) -> None:
        while True:
            async with self._cv:
                while not self._pending and not self._closed:
                    await self._cv.wait()
                if self._closed and not self._pending:
                    return

                batch_n = min(len(self._pending), self._max_batch_size)
                jobs = [self._pending.popleft() for _ in range(batch_n)]

            await self._process_jobs(jobs)


class STTServiceServicer(stt_grpc_pb2_grpc.STTServiceServicer):
    def __init__(
        self,
        *,
        runtime_state: RuntimeState,
        config: AppConfig,
        stream_sample_rate: int,
    ):
        self._runtime = runtime_state
        self._config = config
        self._stream_sample_rate = stream_sample_rate
        batching_mode = config.grpc.infer_batching_mode.strip().lower()
        if batching_mode == "dynamic":
            self._batcher: _GrpcInferBatcherBase = _DynamicGrpcInferBatcher(
                runtime_state=runtime_state,
                stream_sample_rate=stream_sample_rate,
                max_batch_size=config.concurrency.batch_size,
                batch_wait_ms=config.grpc.batch_wait_ms,
            )
        elif batching_mode == "inflight":
            self._batcher = _InFlightGrpcInferBatcher(
                runtime_state=runtime_state,
                stream_sample_rate=stream_sample_rate,
                max_batch_size=config.concurrency.batch_size,
            )
        else:
            raise ValueError(f"Unsupported grpc infer batching mode: {batching_mode}")

        self._request_model_name = config.grpc.request_model_name.strip().lower()
        logger.info("gRPC Infer batching mode: %s", batching_mode)

    async def aclose(self) -> None:
        await self._batcher.aclose()

    async def Infer(self, request, context) -> InferResponse:
        await self._runtime.model_ready_event.wait()

        readiness_error = self._readiness_error()
        if readiness_error is not None:
            return InferResponse(text="", success=False, error=readiness_error)

        if not self._is_supported_request_model_name(request.model_name):
            return InferResponse(
                text="",
                success=False,
                error=f"Unsupported model_name: '{request.model_name}'",
            )

        audio, audio_error = self._decode_audio_payload(request.audio)
        if audio_error is not None:
            return InferResponse(text="", success=False, error=audio_error)
        assert audio is not None

        language, lang_error = self._resolve_language(_empty_to_none(request.lang))
        if lang_error is not None:
            return InferResponse(text="", success=False, error=lang_error)

        try:
            item = await self._batcher.submit(audio=audio, language=language)
        except Exception as exc:
            logger.exception("gRPC infer failed: %s", exc)
            return InferResponse(text="", success=False, error=str(exc))

        return InferResponse(text=item.text, success=item.success, error=item.error)

    async def BatchInfer(self, request, context) -> BatchInferResponse:
        await self._runtime.model_ready_event.wait()

        audio_payloads = list(request.audios)
        request_size = len(audio_payloads)
        if request_size == 0:
            return BatchInferResponse(items=[_error_item("audios is empty")])

        readiness_error = self._readiness_error()
        if readiness_error is not None:
            return BatchInferResponse(items=[_error_item(readiness_error) for _ in range(request_size)])

        if not self._is_supported_request_model_name(request.model_name):
            msg = f"Unsupported model_name: '{request.model_name}'"
            return BatchInferResponse(items=[_error_item(msg) for _ in range(request_size)])

        languages, lang_error = self._resolve_batch_languages(list(request.langs), request_size)
        if lang_error is not None:
            return BatchInferResponse(items=[_error_item(lang_error) for _ in range(request_size)])
        assert languages is not None

        items: list[Optional[BatchInferItem]] = [None] * request_size
        infer_inputs: list[tuple[int, "np.ndarray"]] = []

        for idx, payload in enumerate(audio_payloads):
            audio, decode_error = self._decode_audio_payload(payload)
            if decode_error is not None:
                items[idx] = _error_item(decode_error)
                continue

            infer_inputs.append((idx, audio))

        if not infer_inputs:
            return BatchInferResponse(items=_filled_items(items))

        model_batch_size = self._config.concurrency.batch_size
        for chunk in split_into_batches(infer_inputs, model_batch_size):
            tasks = [
                self._batcher.submit(audio=audio, language=languages[idx])
                for idx, audio in chunk
            ]
            batch_items = await asyncio.gather(*tasks)
            for (idx, _), item in zip(chunk, batch_items):
                items[idx] = item

        return BatchInferResponse(items=_filled_items(items))

    async def HealthCheck(self, request, context) -> HealthCheckResponse:
        requested_names = list(request.model_names)
        if not requested_names:
            requested_names = ["asr", "aligner"]

        model_status = {}
        for name in requested_names:
            model_status[name] = self._is_model_ready(name)

        return HealthCheckResponse(
            ready=self._runtime.model_status == "ready",
            model_status=model_status,
        )

    def _is_model_ready(self, requested_name: str) -> bool:
        normalized = requested_name.strip().lower()
        if normalized == "asr":
            return self._runtime.has_model("asr")
        if normalized == "aligner":
            return self._runtime.has_model("aligner")
        if is_supported_model_name(
            requested_name,
            self._config.models.asr_model_name,
            self._config.models.asr_model_path,
        ):
            return self._runtime.has_model("asr")
        return False

    def _is_supported_request_model_name(self, request_model_name: str) -> bool:
        requested = request_model_name.strip().lower()
        if not requested:
            return True
        return requested == self._request_model_name

    def _readiness_error(self) -> Optional[str]:
        if self._runtime.model_status != "ready":
            return f"Server not ready: {self._runtime.model_status}"
        if not self._runtime.has_model("asr"):
            return "ASR model is not enabled or failed to load."
        return None

    def _resolve_batch_languages(
        self,
        langs: list[str],
        batch_count: int,
    ) -> tuple[Optional[list[Optional[str]]], Optional[str]]:
        if not langs:
            return [None] * batch_count, None
        if len(langs) == 1:
            mapped, err = self._resolve_language(_empty_to_none(langs[0]))
            if err is not None:
                return None, err
            return [mapped] * batch_count, None
        if len(langs) == batch_count:
            resolved: list[Optional[str]] = []
            for raw_lang in langs:
                mapped, err = self._resolve_language(_empty_to_none(raw_lang))
                if err is not None:
                    return None, err
                resolved.append(mapped)
            return resolved, None
        return None, "langs must be empty, length 1, or same length as audios"

    @staticmethod
    def _resolve_language(raw_lang: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        mapped = map_language(raw_lang)
        if mapped == "unsupported":
            return None, f"Unsupported lang: {raw_lang}"
        return mapped, None

    @staticmethod
    def _decode_audio_payload(payload: bytes) -> tuple[Optional["np.ndarray"], Optional[str]]:
        if not payload:
            return None, "audio is empty"
        try:
            import numpy as np
        except ImportError as exc:
            return None, f"numpy is required: {exc}"
        try:
            audio = np.frombuffer(payload, dtype=np.float32)
        except ValueError as exc:
            return None, f"Invalid audio bytes: {exc}"
        if audio.size == 0:
            return None, "audio is empty"
        return audio, None


def _empty_to_none(value: str) -> Optional[str]:
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _error_item(message: str) -> BatchInferItem:
    return BatchInferItem(text="", success=False, error=message)


def _filled_items(items: list[Optional[BatchInferItem]]) -> list[BatchInferItem]:
    return [item if item is not None else _error_item("No inference result") for item in items]
