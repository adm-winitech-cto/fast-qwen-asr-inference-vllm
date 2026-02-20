import argparse
import asyncio
import time
from asyncio import create_subprocess_exec, subprocess

from benchmark_core import BenchmarkStats


async def run_command(cmd: list[str]) -> tuple[float, str, str, int]:
    started_at = time.time()
    proc = await create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    finished_at = time.time()
    return finished_at - started_at, stdout.decode(), stderr.decode(), proc.returncode


async def run_streaming_request(url: str, file_path: str, stats: BenchmarkStats) -> None:
    cmd = ["python", "client-streaming.py", "-e", url, "-f", file_path]
    _, stdout, stderr, returncode = await run_command(cmd)
    if returncode == 0:
        stats.record_streaming_stdout(stdout)
    else:
        print(f"Error: {stderr}")
        stats.errors += 1


async def run_batch_request(url: str, file_path: str, stats: BenchmarkStats) -> None:
    cmd = ["curl", "-s", "-X", "POST", url, "-F", f"files=@{file_path}"]
    duration, stdout, _, returncode = await run_command(cmd)
    if returncode == 0 and "Error" not in stdout:
        stats.record_batch_result(duration, returncode)
    else:
        stats.errors += 1


async def worker(mode: str, url: str, file_path: str, requests: int, stats: BenchmarkStats) -> None:
    for _ in range(requests):
        if mode == "streaming":
            await run_streaming_request(url, file_path, stats)
        else:
            await run_batch_request(url, file_path, stats)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["streaming", "batch"], required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    args = parser.parse_args()

    print(f"Starting Benchmark: {args.clients} clients, {args.requests} requests each.")

    stats = BenchmarkStats()
    stats.start_time = time.time()

    await asyncio.gather(
        *(worker(args.mode, args.url, args.file, args.requests, stats) for _ in range(args.clients))
    )

    stats.end_time = time.time()
    stats.report()


if __name__ == "__main__":
    asyncio.run(main())
