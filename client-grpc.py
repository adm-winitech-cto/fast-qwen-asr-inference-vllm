#!/usr/bin/env python3
"""
Simple gRPC client for STTService defined in stt_grpc.proto.
Supports configurable concurrent request batch_size.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import subprocess
import time
from typing import TYPE_CHECKING

import grpc

import stt_grpc_pb2
import stt_grpc_pb2_grpc

DEFAULT_SAMPLE_RATE = 16000
GRPC_CHANNEL_OPTIONS = (
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
)

if TYPE_CHECKING:
    import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STT gRPC client")
    parser.add_argument("--endpoint", default="127.0.0.1:50051", help="gRPC endpoint host:port")
    parser.add_argument("--file", help="Audio file path")
    parser.add_argument("--lang", default="ko", help="Language code for InferRequest.lang")
    parser.add_argument("--pretext", default="", help="InferRequest.pretext")
    parser.add_argument("--model-name", default="", help="InferRequest.model_name")
    parser.add_argument("--requests", type=int, default=1, help="Total infer requests to send")
    parser.add_argument("--batch-size", type=int, default=1, help="Concurrent requests per round")
    parser.add_argument("--health", action="store_true", help="Call HealthCheck instead of Infer")
    parser.add_argument(
        "--health-model",
        action="append",
        default=[],
        help="model_names entry for HealthCheck (repeat option allowed)",
    )
    return parser.parse_args()


def decode_audio_to_float32(path: str) -> tuple["np.ndarray", int]:
    import numpy as np

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            path,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg decoding failed: {err.decode(errors='ignore')}")
    audio = np.frombuffer(out, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("No audio samples decoded from input file")
    return audio, DEFAULT_SAMPLE_RATE


def ensure_mono_16k(audio: "np.ndarray", sr: int) -> "np.ndarray":
    if sr != DEFAULT_SAMPLE_RATE:
        raise ValueError(f"Expected {DEFAULT_SAMPLE_RATE}Hz audio, got {sr}Hz")
    return audio


async def infer_once(
    stub: stt_grpc_pb2_grpc.STTServiceStub,
    *,
    audio_bytes: bytes,
    lang: str,
    pretext: str,
    model_name: str,
) -> tuple[float, stt_grpc_pb2.InferResponse]:
    started = time.perf_counter()
    response = await stub.Infer(
        stt_grpc_pb2.InferRequest(
            audio=audio_bytes,
            lang=lang,
            pretext=pretext,
            model_name=model_name,
        )
    )
    latency = time.perf_counter() - started
    return latency, response


async def run_health(stub: stt_grpc_pb2_grpc.STTServiceStub, model_names: list[str]) -> None:
    response = await stub.HealthCheck(stt_grpc_pb2.HealthCheckRequest(model_names=model_names))
    print(f"ready={response.ready}")
    print(f"model_status={dict(response.model_status)}")


async def run_infer(args: argparse.Namespace) -> None:
    if not args.file:
        raise ValueError("--file is required unless --health is used")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.requests <= 0:
        raise ValueError("--requests must be >= 1")

    audio, sr = decode_audio_to_float32(args.file)
    audio = ensure_mono_16k(audio, sr)
    audio_duration_sec = audio.size / float(DEFAULT_SAMPLE_RATE)
    audio_bytes = audio.tobytes()

    async with grpc.aio.insecure_channel(args.endpoint, options=GRPC_CHANNEL_OPTIONS) as channel:
        stub = stt_grpc_pb2_grpc.STTServiceStub(channel)
        latencies: list[float] = []
        rtfs: list[float] = []
        success = 0
        errors = 0
        final_text = ""

        remaining = args.requests
        while remaining > 0:
            current = min(args.batch_size, remaining)
            tasks = [
                infer_once(
                    stub,
                    audio_bytes=audio_bytes,
                    lang=args.lang,
                    pretext=args.pretext,
                    model_name=args.model_name,
                )
                for _ in range(current)
            ]
            for latency, response in await asyncio.gather(*tasks):
                latencies.append(latency)
                if response.success:
                    success += 1
                    final_text = response.text
                    if audio_duration_sec > 0:
                        rtfs.append(latency / audio_duration_sec)
                else:
                    errors += 1
                    print(f"error={response.error}")
            remaining -= current

    print(f"requests={args.requests} success={success} errors={errors} batch_size={args.batch_size}")
    if final_text:
        print(f"text={final_text}")
    if latencies:
        print(f"latency_avg={statistics.mean(latencies):.4f}s latency_p95={_p95(latencies):.4f}s")
    if rtfs:
        print(f"rtf_avg={statistics.mean(rtfs):.4f} rtf_p95={_p95(rtfs):.4f}")


def _p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20)[-1]


async def main() -> None:
    args = parse_args()
    async with grpc.aio.insecure_channel(args.endpoint, options=GRPC_CHANNEL_OPTIONS) as channel:
        stub = stt_grpc_pb2_grpc.STTServiceStub(channel)
        if args.health:
            await run_health(stub, args.health_model)
            return

    await run_infer(args)


if __name__ == "__main__":
    asyncio.run(main())
