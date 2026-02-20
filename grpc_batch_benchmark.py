#!/usr/bin/env python3
"""
Benchmark gRPC BatchInfer(repeated bytes) throughput/RTF.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import subprocess
import time
from typing import Iterator

import grpc

import stt_grpc_pb2
import stt_grpc_pb2_grpc

PCM_SAMPLE_RATE = 16000
PCM_BYTES_PER_SAMPLE = 4  # float32
DEFAULT_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".pcm")
GRPC_CHANNEL_OPTIONS = (
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gRPC BatchInfer benchmark")
    parser.add_argument("--endpoint", default="127.0.0.1:50051", help="gRPC endpoint")
    parser.add_argument("--audio-dir", required=True, help="Directory containing input audio files")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of files to benchmark")
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=16,
        help="Number of audios per BatchInfer request",
    )
    parser.add_argument("--lang", default="ko", help="Language code")
    parser.add_argument("--model-name", default="1.7b-v2-cp47728", help="ASR model alias/name")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated audio extensions to include",
    )
    return parser.parse_args()


def iter_audio_files(root_dir: str, extensions: tuple[str, ...]) -> Iterator[str]:
    for root, _, files in os.walk(root_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(extensions):
                yield os.path.join(root, name)


def decode_audio_to_f32_bytes(path: str) -> bytes:
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            path,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(PCM_SAMPLE_RATE),
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err.decode(errors='ignore')}")
    if not out:
        raise RuntimeError(f"Decoded audio is empty: {path}")
    return out


def split_batches(items: list[str], batch_size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20)[-1]


async def run(args: argparse.Namespace) -> None:
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be >= 1")
    if args.request_batch_size <= 0:
        raise ValueError("--request-batch-size must be >= 1")

    exts = tuple(ext.strip().lower() for ext in args.extensions.split(",") if ext.strip())
    files = sorted(iter_audio_files(args.audio_dir, exts))
    if not files:
        raise RuntimeError(f"No audio files found in {args.audio_dir}")

    files = files[: args.sample_size]
    print(f"files_selected={len(files)}")
    print(f"request_batch_size={args.request_batch_size}")

    total_audio_sec = 0.0
    rpc_sec_total = 0.0
    rpc_latencies: list[float] = []
    success = 0
    errors = 0
    first_text = ""

    benchmark_start = time.perf_counter()
    async with grpc.aio.insecure_channel(args.endpoint, options=GRPC_CHANNEL_OPTIONS) as channel:
        stub = stt_grpc_pb2_grpc.STTServiceStub(channel)

        for file_batch in split_batches(files, args.request_batch_size):
            audios: list[bytes] = []
            for path in file_batch:
                try:
                    audio_bytes = decode_audio_to_f32_bytes(path)
                except Exception as exc:
                    errors += 1
                    print(f"decode_error path={path} error={exc}")
                    continue
                audios.append(audio_bytes)
                total_audio_sec += len(audio_bytes) / float(PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE)

            if not audios:
                continue

            started = time.perf_counter()
            response = await stub.BatchInfer(
                stt_grpc_pb2.BatchInferRequest(
                    audios=audios,
                    langs=[args.lang],
                    model_name=args.model_name,
                )
            )
            rpc_elapsed = time.perf_counter() - started
            rpc_sec_total += rpc_elapsed
            rpc_latencies.append(rpc_elapsed)

            for item in response.items:
                if item.success:
                    success += 1
                    if not first_text and item.text:
                        first_text = item.text
                else:
                    errors += 1

    wall_sec = time.perf_counter() - benchmark_start

    rpc_rtf = rpc_sec_total / total_audio_sec if total_audio_sec > 0 else float("inf")
    wall_rtf = wall_sec / total_audio_sec if total_audio_sec > 0 else float("inf")
    rpc_x_rt = total_audio_sec / rpc_sec_total if rpc_sec_total > 0 else 0.0
    wall_x_rt = total_audio_sec / wall_sec if wall_sec > 0 else 0.0

    print(f"success={success} errors={errors}")
    print(f"total_audio_sec={total_audio_sec:.3f}")
    print(f"rpc_sec_total={rpc_sec_total:.3f}")
    print(f"wall_sec_total={wall_sec:.3f}")
    print(f"rpc_rtf={rpc_rtf:.4f} rpc_throughput_x={rpc_x_rt:.2f}")
    print(f"wall_rtf={wall_rtf:.4f} wall_throughput_x={wall_x_rt:.2f}")
    if rpc_latencies:
        print(
            f"batch_rpc_latency_avg={statistics.mean(rpc_latencies):.4f}s "
            f"batch_rpc_latency_p95={_p95(rpc_latencies):.4f}s"
        )
    if first_text:
        print(f"sample_text={first_text}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
