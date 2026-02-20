# Client for streaming ASR - this is for TESTING PURPOSES

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

import websockets

CHUNK_BYTES = 640  # 20ms at 16kHz 16-bit mono
PCM_SAMPLE_RATE_HZ = 16000
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE_HZ * 2


def build_start_message() -> str:
    return json.dumps(
        {
            "type": "start",
            "format": "pcm_s16le",
            "sample_rate_hz": PCM_SAMPLE_RATE_HZ,
            "channels": 1,
        }
    )


async def sender(ws, pcm_path: str) -> None:
    await ws.send(build_start_message())

    print(f"Streaming {pcm_path}...")
    with open(pcm_path, "rb") as pcm_file:
        while True:
            chunk = pcm_file.read(CHUNK_BYTES)
            if not chunk:
                break
            await ws.send(chunk)
            # Yield so receiver can flush partial lines in near real-time.
            await asyncio.sleep(0)

    await ws.send(json.dumps({"type": "stop"}))
    print("Finished sending audio.")


def print_server_event(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    try:
        evt = json.loads(message)
    except json.JSONDecodeError:
        print(f"\n[Raw]: {message}")
        return

    msg_type = evt.get("type")
    text = evt.get("text", "")
    lang = evt.get("language", "")

    if msg_type == "ready":
        print(f"[{timestamp}] [Server Ready]")
    elif msg_type == "partial":
        # Overwrite line for partial updates to keep clean output
        sys.stdout.write(f"\r[{timestamp}] [Partial] ({lang}): {text}")
        sys.stdout.flush()
    elif msg_type == "final":
        print(f"\n[{timestamp}] [Final] ({lang}): {text}")
    elif msg_type == "error":
        print(f"\n[{timestamp}] [Error]: {evt.get('message')}")
    else:
        print(f"\n[{timestamp}] [Unknown]: {evt}")


async def receiver(ws) -> None:
    async for message in ws:
        print_server_event(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-ASR Streaming Client")
    parser.add_argument(
        "-e",
        "--endpoint",
        required=True,
        help="WebSocket Endpoint URL (e.g. ws://localhost:8907/transcribe-streaming)",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to raw PCM 16k 16-bit mono file (or WAV with correct format)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    print(f"Connecting to {args.endpoint}...")

    file_size = os.path.getsize(args.file)
    duration = file_size / float(PCM_BYTES_PER_SECOND)
    print(f"Audio Duration: {duration:.2f}s")

    start_time = time.time()
    try:
        async with websockets.connect(args.endpoint, max_size=None) as ws:
            await asyncio.gather(sender(ws, args.file), receiver(ws))

        process_time = time.time() - start_time
        rtf = process_time / duration if duration > 0 else 0.0
        print(f"\nProcessing Time: {process_time:.2f}s")
        print(f"Real-Time Factor (RTF): {rtf:.4f}")

    except Exception as exc:
        print(f"Connection failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
