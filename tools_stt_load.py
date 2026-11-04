"""Simulate N callers talking to the STT server the way stt-client does.

Each channel "speaks" a real utterance: every --period ms it re-transcribes the
audio so far (a partial, like retranscribe_period_ms), then 0.7 s after the end
it sends the final. Then it stays quiet for --gap s (the AI caller talking) and
repeats until --seconds have passed. Prints final-recognition latency and the
number of inferences. --period 0 sends finals only.

usage: python tools_stt_load.py --channels 3 --period 1000 --seconds 60 WAV...
"""
import argparse
import json
import random
import statistics
import threading
import time
import wave

import grpc
import numpy as np

import stt_grpc_pb2
import stt_grpc_pb2_grpc


def load(path):
    with wave.open(path) as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if rate != 16000:
        idx = np.arange(0, len(pcm), rate / 16000)
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--endpoint", default="127.0.0.1:50051")
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--period", type=int, default=1000, help="partial re-transcription period in ms; 0 = finals only")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--model", default="asr")
    a = ap.parse_args()
    clips = [load(p) for p in a.wavs]
    finals, partials, lock = [], [0], threading.Lock()
    stop_at = time.monotonic() + a.seconds

    def channel(seed):
        rnd = random.Random(seed)
        stub = stt_grpc_pb2_grpc.STTServiceStub(grpc.insecure_channel(a.endpoint)) if hasattr(stt_grpc_pb2_grpc, "STTServiceStub") else None
        if stub is None:
            name = [n for n in dir(stt_grpc_pb2_grpc) if n.endswith("Stub")][0]
            stub = getattr(stt_grpc_pb2_grpc, name)(grpc.insecure_channel(a.endpoint))
        time.sleep(rnd.uniform(0, 1.5))
        while time.monotonic() < stop_at:
            audio = clips[rnd.randrange(len(clips))]
            seconds = len(audio) / 16000
            spoken = 0.0
            start = time.monotonic()
            while a.period and spoken + a.period / 1000 < seconds:
                spoken += a.period / 1000
                time.sleep(max(0.0, start + spoken - time.monotonic()))
                stub.Infer(stt_grpc_pb2.InferRequest(audio=audio[: int(spoken * 16000)].tobytes(), lang="ko", model_name=a.model), timeout=30)
                with lock:
                    partials[0] += 1
            time.sleep(max(0.0, start + seconds + 0.7 - time.monotonic()))  # 700 ms silence -> final
            t0 = time.monotonic()
            stub.Infer(stt_grpc_pb2.InferRequest(audio=audio.tobytes(), lang="ko", model_name=a.model), timeout=30)
            with lock:
                finals.append(time.monotonic() - t0)
            time.sleep(a.gap)

    threads = [threading.Thread(target=channel, args=(i,)) for i in range(a.channels)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    f = sorted(finals)
    print(json.dumps({"channels": a.channels, "period_ms": a.period, "finals": len(f), "partials": partials[0],
                      "final_p50": round(statistics.median(f), 3), "final_p90": round(f[int(0.9 * (len(f) - 1))], 3),
                      "final_max": round(f[-1], 3)}))


if __name__ == "__main__":
    main()
