from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

_STREAMING_RTF_PATTERN = re.compile(r"Real-Time Factor \(RTF\): ([0-9.]+)")
_STREAMING_TIME_PATTERN = re.compile(r"Processing Time: ([0-9.]+)s")


@dataclass(frozen=True)
class StreamingMetrics:
    processing_time: float
    rtf: float


def parse_streaming_metrics(stdout: str) -> Optional[StreamingMetrics]:
    rtf_match = _STREAMING_RTF_PATTERN.search(stdout)
    time_match = _STREAMING_TIME_PATTERN.search(stdout)
    if not rtf_match or not time_match:
        return None

    try:
        return StreamingMetrics(
            processing_time=float(time_match.group(1)),
            rtf=float(rtf_match.group(1)),
        )
    except ValueError:
        return None


def _p95(values: list[float]) -> float:
    if len(values) <= 1:
        return values[0]
    return statistics.quantiles(values, n=20)[-1]


@dataclass
class BenchmarkStats:
    rtfs: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    errors: int = 0
    success: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def record_streaming_stdout(self, stdout: str) -> None:
        metrics = parse_streaming_metrics(stdout)
        if metrics is None:
            self.errors += 1
            return

        self.rtfs.append(metrics.rtf)
        self.latencies.append(metrics.processing_time)
        self.success += 1

    def record_batch_result(self, duration: float, returncode: int) -> None:
        if returncode == 0:
            self.latencies.append(duration)
            self.success += 1
        else:
            self.errors += 1

    def report(self) -> None:
        total_time = self.end_time - self.start_time
        total_requests = self.success + self.errors
        avg_qps = (self.success / total_time) if total_time > 0 else 0.0

        print("\n--- Benchmark Results ---")
        print(f"Total Requests:     {total_requests}")
        print(f"Successful:         {self.success}")
        print(f"Errors:             {self.errors}")
        print(f"Total Wall Time:    {total_time:.2f}s")
        print(f"Avg QPS:            {avg_qps:.2f}")

        if self.latencies:
            print("\n-- Latency (sec) --")
            print(f"Avg:  {statistics.mean(self.latencies):.3f}")
            print(f"P50:  {statistics.median(self.latencies):.3f}")
            print(f"P95:  {_p95(self.latencies):.3f}")

        if self.rtfs:
            print("\n-- RTF --")
            print(f"Avg:  {statistics.mean(self.rtfs):.4f}")
            print(f"P50:  {statistics.median(self.rtfs):.4f}")
            print(f"P95:  {_p95(self.rtfs):.4f}")
