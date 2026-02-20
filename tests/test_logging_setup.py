import logging
import shutil
import tempfile
import unittest
from pathlib import Path

from tests import _test_path  # noqa: F401


class LoggingSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.mkdtemp(prefix="asr-logs-")

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_configure_logging_creates_logs_dir_and_file_handler(self):
        from fast_qwen_asr.logging.setup import configure_logging

        logs_dir = Path(self._tmp_dir) / "logs"
        log_file = logs_dir / "app.log"

        configure_logging(
            level="INFO",
            log_file_path=log_file,
            max_bytes=1024,
            backup_count=2,
        )

        self.assertTrue(logs_dir.exists())
        root = logging.getLogger()
        handlers = root.handlers
        self.assertTrue(any(h.__class__.__name__ == "RotatingFileHandler" for h in handlers))
        root.info("logging test message")

    def test_format_includes_level_and_logger_name(self):
        from fast_qwen_asr.logging.setup import build_log_formatter

        formatter = build_log_formatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=12,
            msg="hello",
            args=(),
            exc_info=None,
        )
        rendered = formatter.format(record)
        self.assertIn("INFO", rendered)
        self.assertIn("test.logger", rendered)
        self.assertIn("hello", rendered)


if __name__ == "__main__":
    unittest.main()
