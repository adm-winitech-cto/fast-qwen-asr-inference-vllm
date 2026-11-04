import asyncio
from dataclasses import replace
import importlib.util
import struct
import unittest
from types import SimpleNamespace

from tests import _test_path  # noqa: F401
from fast_qwen_asr.config.app_config import load_app_config
from fast_qwen_asr.api.grpc_service import STTServiceServicer
from fast_qwen_asr.core.runtime_state import RuntimeState
import stt_grpc_pb2


class _FakeASRModel:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def transcribe(self, *, audio, language, return_time_stamps):
        assert return_time_stamps is False
        self.batch_sizes.append(len(audio))
        results = []
        for idx, lang in enumerate(language):
            results.append(SimpleNamespace(text=f"ok-{idx}", language=lang or "auto"))
        return results


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is required for grpc infer path tests")
class GrpcServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = RuntimeState(max_concurrent_decode=1, max_concurrent_infer=1)
        self.fake_model = _FakeASRModel()
        self.runtime.models["asr"] = self.fake_model
        self.runtime.model_status = "ready"
        self.runtime.model_ready_event.set()
        base_config = load_app_config()
        self.config = replace(
            base_config,
            concurrency=replace(base_config.concurrency, batch_size=8),
            grpc=replace(base_config.grpc, batch_wait_ms=15, infer_batching_mode="dynamic"),
        )
        self.servicer = STTServiceServicer(
            runtime_state=self.runtime,
            config=self.config,
            stream_sample_rate=16000,
        )

    async def asyncTearDown(self):
        await self.servicer.aclose()

    async def test_infer_success(self):
        audio_bytes = struct.pack("<ff", 0.1, -0.2)
        response = await self.servicer.Infer(
            stt_grpc_pb2.InferRequest(
                audio=audio_bytes,
                lang="ko",
                pretext="",
                model_name="",
            ),
            context=None,
        )
        self.assertTrue(response.success)
        self.assertEqual(response.error, "")
        self.assertEqual(response.text, "ok-0")

    async def test_infer_accepts_configured_request_model_name(self):
        audio_bytes = struct.pack("<ff", 0.1, -0.2)
        response = await self.servicer.Infer(
            stt_grpc_pb2.InferRequest(
                audio=audio_bytes,
                lang="ko",
                pretext="",
                model_name="asr",
            ),
            context=None,
        )
        self.assertTrue(response.success)
        self.assertEqual(response.error, "")

    async def test_infer_rejects_unknown_model_name(self):
        audio_bytes = struct.pack("<ff", 0.1, -0.2)
        response = await self.servicer.Infer(
            stt_grpc_pb2.InferRequest(
                audio=audio_bytes,
                lang="ko",
                pretext="",
                model_name="whisper_r_ko",
            ),
            context=None,
        )
        self.assertFalse(response.success)
        self.assertIn("Unsupported model_name", response.error)

    async def test_health_check(self):
        response = await self.servicer.HealthCheck(
            stt_grpc_pb2.HealthCheckRequest(model_names=["asr", "aligner"]),
            context=None,
        )
        self.assertTrue(response.ready)
        self.assertTrue(response.model_status["asr"])
        self.assertFalse(response.model_status["aligner"])

    async def test_batch_infer_success(self):
        audio_a = struct.pack("<ff", 0.1, -0.2)
        audio_b = struct.pack("<ff", 0.2, -0.3)
        response = await self.servicer.BatchInfer(
            stt_grpc_pb2.BatchInferRequest(
                audios=[audio_a, audio_b],
                langs=["ko"],
                model_name="",
            ),
            context=None,
        )
        self.assertEqual(len(response.items), 2)
        self.assertTrue(response.items[0].success)
        self.assertTrue(response.items[1].success)
        self.assertTrue(response.items[0].text.startswith("ok-"))
        self.assertTrue(response.items[1].text.startswith("ok-"))

    async def test_batch_infer_rejects_empty_audio_list(self):
        response = await self.servicer.BatchInfer(
            stt_grpc_pb2.BatchInferRequest(
                audios=[],
                langs=["ko"],
                model_name="",
            ),
            context=None,
        )
        self.assertEqual(len(response.items), 1)
        self.assertFalse(response.items[0].success)
        self.assertIn("audios is empty", response.items[0].error)

    async def test_infer_concurrent_requests_are_microbatched(self):
        audio_a = struct.pack("<ff", 0.1, -0.2)
        audio_b = struct.pack("<ff", 0.2, -0.3)

        first, second = await asyncio.gather(
            self.servicer.Infer(
                stt_grpc_pb2.InferRequest(audio=audio_a, lang="ko", pretext="", model_name=""),
                context=None,
            ),
            self.servicer.Infer(
                stt_grpc_pb2.InferRequest(audio=audio_b, lang="ko", pretext="", model_name=""),
                context=None,
            ),
        )

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(self.fake_model.batch_sizes, [2])

    async def test_infer_concurrent_requests_are_microbatched_inflight_mode(self):
        await self.servicer.aclose()
        self.fake_model = _FakeASRModel()
        self.runtime.models["asr"] = self.fake_model
        inflight_config = replace(
            self.config,
            grpc=replace(self.config.grpc, infer_batching_mode="inflight", batch_wait_ms=50),
        )
        self.servicer = STTServiceServicer(
            runtime_state=self.runtime,
            config=inflight_config,
            stream_sample_rate=16000,
        )

        audio_a = struct.pack("<ff", 0.1, -0.2)
        audio_b = struct.pack("<ff", 0.2, -0.3)

        first, second = await asyncio.gather(
            self.servicer.Infer(
                stt_grpc_pb2.InferRequest(audio=audio_a, lang="ko", pretext="", model_name=""),
                context=None,
            ),
            self.servicer.Infer(
                stt_grpc_pb2.InferRequest(audio=audio_b, lang="ko", pretext="", model_name=""),
                context=None,
            ),
        )

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(self.fake_model.batch_sizes, [2])


if __name__ == "__main__":
    unittest.main()
