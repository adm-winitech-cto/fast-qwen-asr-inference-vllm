import unittest

from tests import _test_path  # noqa: F401
from fast_qwen_asr.core.runtime_state import RuntimeState


class RuntimeStateTests(unittest.TestCase):
    def test_initializes_with_expected_limits(self):
        state = RuntimeState(max_concurrent_decode=4, max_concurrent_infer=2)
        self.assertEqual(state.model_status, "starting")
        self.assertEqual(state.decode_limit, 4)
        self.assertEqual(state.infer_limit, 2)

    def test_tracks_model_presence(self):
        state = RuntimeState(max_concurrent_decode=1, max_concurrent_infer=1)
        self.assertFalse(state.has_model("asr"))
        state.models["asr"] = object()
        self.assertTrue(state.has_model("asr"))


if __name__ == "__main__":
    unittest.main()
