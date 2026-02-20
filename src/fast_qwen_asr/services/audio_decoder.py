from __future__ import annotations

import io
import subprocess
from typing import Tuple

import numpy as np
import soundfile as sf


def read_audio_file(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode audio payload using soundfile first and ffmpeg fallback for compressed formats.
    """
    try:
        with io.BytesIO(file_bytes) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr
    except Exception:
        process = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate(input=file_bytes)
        if process.returncode != 0:
            raise ValueError(f"FFmpeg decoding failed: {err.decode(errors='ignore')}")
        with io.BytesIO(out) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr
