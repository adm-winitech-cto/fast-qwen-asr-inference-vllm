import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests import _test_path  # noqa: F401
from fast_qwen_asr.core import model_loader


def _config(max_model_len=16384):
    models = SimpleNamespace(
        enable_asr_model=True, enable_aligner_model=False, asr_model_path="/m", asr_model_name="m",
        local_files_only=True, gpu_memory_utilization=0.09, max_new_tokens=324,
        max_model_len=max_model_len, warmup_enabled=False,
    )
    return SimpleNamespace(models=models)


class _Runtime:
    def __init__(self):
        self.models = {}
        self.model_status = "starting"
        self.model_ready_event = asyncio.Event()

    def has_model(self, name):
        return name in self.models


class AsrLoadRetryTests(unittest.TestCase):
    def _load(self, side_effect, max_model_len=16384):
        calls = []

        def fake_llm(**kwargs):
            calls.append(kwargs)
            outcome = side_effect.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        runtime = _Runtime()
        fake_model = SimpleNamespace(LLM=fake_llm)
        with patch.object(model_loader, "Qwen3ASRModel", fake_model), \
                patch.object(model_loader, "resolve_model_source", return_value="/m"), \
                patch.object(model_loader, "ASR_LOAD_RETRY_SECONDS", 0):
            asyncio.run(model_loader.load_models_background(runtime=runtime, config=_config(max_model_len), stream_sample_rate=16000))
        return runtime, calls

    def test_retries_a_failed_engine_start(self):
        runtime, calls = self._load([ValueError("KV cache too small"), "engine"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(runtime.models["asr"], "engine")
        self.assertNotEqual(runtime.model_status, "error")

    def test_gives_up_after_the_last_attempt(self):
        runtime, calls = self._load([ValueError("a"), ValueError("b"), ValueError("c")])
        self.assertEqual(len(calls), model_loader.ASR_LOAD_ATTEMPTS)
        self.assertEqual(runtime.model_status, "error")

    def test_passes_max_model_len_only_when_set(self):
        _, calls = self._load(["engine"])
        self.assertEqual(calls[0]["max_model_len"], 16384)
        _, calls = self._load(["engine"], max_model_len=None)
        self.assertNotIn("max_model_len", calls[0])


if __name__ == "__main__":
    unittest.main()
