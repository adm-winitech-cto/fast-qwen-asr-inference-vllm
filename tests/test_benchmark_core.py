import unittest

from benchmark_core import BenchmarkStats, parse_streaming_metrics


class BenchmarkCoreTests(unittest.TestCase):
    def test_parse_streaming_metrics(self):
        text = "Processing Time: 0.17s\nReal-Time Factor (RTF): 0.0847"
        metrics = parse_streaming_metrics(text)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertAlmostEqual(metrics.processing_time, 0.17)
        self.assertAlmostEqual(metrics.rtf, 0.0847)

    def test_parse_streaming_metrics_returns_none(self):
        self.assertIsNone(parse_streaming_metrics("invalid"))

    def test_stats_collect_success_and_error(self):
        stats = BenchmarkStats()
        stats.record_streaming_stdout("Processing Time: 0.10s\nReal-Time Factor (RTF): 0.0500")
        stats.record_streaming_stdout("invalid")
        self.assertEqual(stats.success, 1)
        self.assertEqual(stats.errors, 1)
        self.assertEqual(len(stats.latencies), 1)
        self.assertEqual(len(stats.rtfs), 1)


if __name__ == "__main__":
    unittest.main()
