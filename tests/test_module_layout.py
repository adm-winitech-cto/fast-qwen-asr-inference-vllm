import unittest
from pathlib import Path

from tests import _test_path  # noqa: F401


class ModuleLayoutTests(unittest.TestCase):
    def test_new_package_paths_are_importable(self):
        from fast_qwen_asr.config.app_config import load_app_config
        from fast_qwen_asr.api.grpc_service import STTServiceServicer
        from fast_qwen_asr.core.runtime_state import RuntimeState

        self.assertIsNotNone(load_app_config)
        self.assertIsNotNone(STTServiceServicer)
        self.assertIsNotNone(RuntimeState)

    def test_legacy_wrapper_files_removed(self):
        root = Path(__file__).resolve().parents[1]
        wrappers = [
            "_bootstrap.py",
            "app_config.py",
            "grpc_runtime.py",
            "grpc_service.py",
            "inference_utils.py",
            "server_runtime.py",
            "server_utils.py",
            "server.py",
            "sitecustomize.py",
        ]
        for rel in wrappers:
            self.assertFalse((root / rel).exists(), f"legacy wrapper still exists: {rel}")


if __name__ == "__main__":
    unittest.main()
