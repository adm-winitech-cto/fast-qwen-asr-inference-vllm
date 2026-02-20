import unittest

from tests import _test_path  # noqa: F401
from fast_qwen_asr.utils.inference import (
    is_supported_model_name,
    split_into_batches,
)


class InferenceUtilsTests(unittest.TestCase):
    def test_split_into_batches(self):
        source = [1, 2, 3, 4, 5]
        chunks = list(split_into_batches(source, 2))
        self.assertEqual(chunks, [[1, 2], [3, 4], [5]])

    def test_split_into_batches_rejects_zero(self):
        with self.assertRaises(ValueError):
            list(split_into_batches([1], 0))

    def test_is_supported_model_name(self):
        configured_name = "Qwen/Qwen3-ASR-1.7B"
        configured_path = "/data/shared/Qwen3-ASR/models/1.7b-v2-cp47728"

        self.assertTrue(is_supported_model_name("", configured_name, configured_path))
        self.assertTrue(is_supported_model_name("Qwen/Qwen3-ASR-1.7B", configured_name, configured_path))
        self.assertTrue(is_supported_model_name("1.7b-v2-cp47728", configured_name, configured_path))
        self.assertFalse(is_supported_model_name("whisper_r_ko", configured_name, configured_path))


if __name__ == "__main__":
    unittest.main()
