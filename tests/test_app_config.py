import json
import os
import tempfile
import unittest
from pathlib import Path

from tests import _test_path  # noqa: F401
from fast_qwen_asr.config.app_config import ConfigError, load_app_config


class AppConfigTests(unittest.TestCase):
    def setUp(self):
        self._old_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_loads_defaults_without_external_config(self):
        os.environ.pop("APP_CONFIG_PATH", None)
        cfg = load_app_config()

        self.assertEqual(cfg.server.host, "0.0.0.0")
        self.assertEqual(cfg.server.port, 8000)
        self.assertEqual(cfg.logging.level, "INFO")
        self.assertEqual(cfg.logging.directory, "logs")
        self.assertEqual(cfg.logging.file_name, "app.log")
        self.assertEqual(cfg.grpc.infer_batching_mode, "dynamic")
        self.assertEqual(cfg.concurrency.batch_size, 1)
        self.assertEqual(cfg.models.asr_model_name, "Qwen/Qwen3-ASR-1.7B")
        self.assertIsNone(cfg.models.asr_model_path)
        self.assertIsNone(cfg.models.aligner_model_path)
        self.assertFalse(cfg.models.local_files_only)
        self.assertFalse(cfg.models.force_offline)
        self.assertEqual(cfg.streaming.stream_min_samples, 1600)

    def test_applies_external_file_override(self):
        with tempfile.TemporaryDirectory() as td:
            override_path = Path(td) / "override.json"
            override_path.write_text(
                json.dumps(
                    {
                        "server": {"port": 9010},
                        "concurrency": {"batch_size": 3},
                        "models": {
                            "gpu_memory_utilization": 0.35,
                            "asr_model_path": "/tmp/local-asr",
                            "aligner_model_path": "/tmp/local-aligner",
                            "local_files_only": True,
                            "force_offline": True,
                        },
                        "streaming": {"partial_interval_ms": 200},
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_app_config(str(override_path))

        self.assertEqual(cfg.server.port, 9010)
        self.assertEqual(cfg.concurrency.batch_size, 3)
        self.assertAlmostEqual(cfg.models.gpu_memory_utilization, 0.35)
        self.assertEqual(cfg.models.asr_model_path, "/tmp/local-asr")
        self.assertEqual(cfg.models.aligner_model_path, "/tmp/local-aligner")
        self.assertTrue(cfg.models.local_files_only)
        self.assertTrue(cfg.models.force_offline)
        self.assertEqual(cfg.streaming.partial_interval_ms, 200)

    def test_applies_prefixed_env_override(self):
        os.environ["APP_CFG__SERVER__PORT"] = "9100"
        os.environ["APP_CFG__MODELS__ENABLE_ALIGNER_MODEL"] = "false"
        os.environ["APP_CFG__STREAMING__STREAM_MIN_SAMPLES"] = "4096"
        os.environ["APP_CFG__CONCURRENCY__BATCH_SIZE"] = "6"

        cfg = load_app_config()

        self.assertEqual(cfg.server.port, 9100)
        self.assertFalse(cfg.models.enable_aligner_model)
        self.assertEqual(cfg.streaming.stream_min_samples, 4096)
        self.assertEqual(cfg.concurrency.batch_size, 6)

    def test_applies_legacy_env_override(self):
        os.environ["ASR_MODEL_NAME"] = "Qwen/custom-asr"
        os.environ["ASR_MODEL_PATH"] = "/models/custom-asr"
        os.environ["LOCAL_FILES_ONLY"] = "true"
        os.environ["MAX_CONCURRENT_DECODE"] = "12"
        os.environ["BATCH_SIZE"] = "4"

        cfg = load_app_config()

        self.assertEqual(cfg.models.asr_model_name, "Qwen/custom-asr")
        self.assertEqual(cfg.models.asr_model_path, "/models/custom-asr")
        self.assertTrue(cfg.models.local_files_only)
        self.assertEqual(cfg.concurrency.max_concurrent_decode, 12)
        self.assertEqual(cfg.concurrency.batch_size, 4)

    def test_raises_for_invalid_values(self):
        os.environ["APP_CFG__SERVER__PORT"] = "not-an-int"

        with self.assertRaises(ConfigError):
            load_app_config()

    def test_raises_for_invalid_grpc_batching_mode(self):
        os.environ["APP_CFG__GRPC__INFER_BATCHING_MODE"] = "invalid-mode"
        with self.assertRaises(ConfigError):
            load_app_config()

    def test_missing_external_config_falls_back_to_defaults(self):
        os.environ["APP_CONFIG_PATH"] = "/tmp/does-not-exist/runtime.json"
        cfg = load_app_config()
        self.assertEqual(cfg.server.port, 8000)


if __name__ == "__main__":
    unittest.main()
