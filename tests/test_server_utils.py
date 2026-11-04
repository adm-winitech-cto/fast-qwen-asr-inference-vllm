import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests import _test_path  # noqa: F401
from fast_qwen_asr.utils.server import (
    map_language,
    parse_json_object,
    resolve_model_source,
    should_emit_partial,
    validate_stream_start,
)


class ServerUtilsTests(unittest.TestCase):
    def test_map_language_known_and_passthrough(self):
        self.assertEqual(map_language("de"), "German")
        self.assertEqual(map_language("ZH"), "Chinese")
        self.assertEqual(map_language("xx"), "xx")
        self.assertIsNone(map_language(None))

    def test_validate_stream_start(self):
        self.assertIsNone(validate_stream_start(16000, "pcm_s16le", 16000))
        self.assertIsNone(validate_stream_start(16000, None, 16000))
        self.assertEqual(
            validate_stream_start(8000, "pcm_s16le", 16000),
            "Only pcm_s16le @ 16000Hz supported",
        )

    def test_should_emit_partial(self):
        self.assertTrue(should_emit_partial(last_partial_ts=0.0, now=1.0, partial_interval_ms=100))
        self.assertFalse(should_emit_partial(last_partial_ts=1.0, now=1.05, partial_interval_ms=100))

    def test_parse_json_object(self):
        self.assertEqual(parse_json_object("{\"type\":\"start\"}"), {"type": "start"})
        self.assertIsNone(parse_json_object("[]"))
        self.assertIsNone(parse_json_object("not-json"))

    def test_resolve_model_source_prefers_local_path(self):
        with TemporaryDirectory() as td:
            model_dir = Path(td) / "model"
            model_dir.mkdir(parents=True)
            selected = resolve_model_source(
                local_path=str(model_dir),
                model_name="Qwen/Qwen3-ASR-1.7B",
                local_files_only=True,
                model_label="ASR",
            )
        self.assertEqual(selected, str(model_dir))

    def test_resolve_model_source_requires_local_path_in_offline_mode(self):
        with self.assertRaises(RuntimeError):
            resolve_model_source(
                local_path=None,
                model_name="Qwen/Qwen3-ASR-1.7B",
                local_files_only=True,
                model_label="ASR",
            )


if __name__ == "__main__":
    unittest.main()
