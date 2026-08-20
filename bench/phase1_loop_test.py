"""Phase 1 verification — drive the real voice loop end to end, without a human.

Streams synthetic speech into the live WebSocket at realtime pace, then silence,
and checks that the server endpoints correctly, transcribes, replies, and speaks
inside the latency budget. This exercises everything except the browser: VAD
endpointing, speculative STT, LLM streaming, span chunking, Piper, and the wire
protocol.

It also tests the case that matters most for feel — a mid-answer thinking pause
long enough to look like the end of a turn (plan.md §2.4.3). If the server
endpoints there, it would cut the candidate off, which is the single rudest
failure this system can have.

Usage:
    # terminal 1
    .venv/Scripts/python.exe -m coach.server
    # terminal 2
    .venv/Scripts/python.exe bench/phase1_loop_test.py
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WS_URL = "ws://127.0.0.1:8000/ws"
FRAME = 512          # samples @ 16 kHz — one Silero frame
FRAME_MS = FRAME / 16000 * 1000

ANSWER = (
    "A mutex enforces mutual exclusion, so only one thread holds it at a time, "
    "and the thread that acquired it has to be the one that releases it. "
    "A semaphore is just a counter, so it can allow several threads through."
)


# --------------------------------------------------------------------------- #
# Building the fake microphone signal
# --------------------------------------------------------------------------- #

def speech_16k(text: str) -> np.ndarray:
    """Render `text` to 16 kHz float32 mono using Piper, then resample."""
    from piper import PiperVoice

    voice = PiperVoice.load(str(ROOT / "data" / "voices" / "en_US-lessac-high.onnx"))
    parts, rate = [], 22050
    for c in voice.synthesize(text):
        parts.append(c.audio_int16_bytes)
        rate = c.sample_rate
    pcm = np.frombuffer(b"".join(parts), dtype=np.int16).astype(np.float32) / 32768

    n_out = int(len(pcm) * 16000 / rate)
    return np.interp(
        np.linspace(0, len(pcm) - 1, n_out), np.arange(len(pcm)), pcm
    ).astype(np.float32)


def silence(ms: int) -> np.ndarray:
    # Real rooms are never digitally silent, and a perfectly zero signal is an
    # unrealistically easy case for a VAD. Add a faint noise floor.
    n = int(16000 * ms / 1000)
    return (np.random.randn(n) * 0.0008).astype(np.float32)


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


# --------------------------------------------------------------------------- #

class Client:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.events: list[dict] = []
        self.audio_spans: list[tuple[int, int]] = []   # (rate, n_bytes)
        self.first_audio_at: float | None = None
        self.t_endpoint: float | None = None

    async def reader(self) -> None:
        async for msg in self.ws:
            if isinstance(msg, bytes):
                if self.first_audio_at is None:
                    self.first_audio_at = time.perf_counter()
                rate = struct.unpack("<I", msg[:4])[0]
                self.audio_spans.append((rate, len(msg) - 4))
            else:
                ev = json.loads(msg)
                ev["_t"] = time.perf_counter()
                self.events.append(ev)
                self._log(ev)

    @staticmethod
    def _log(ev: dict) -> None:
        t = ev["type"]
        if t == "state":
            print(f"    [state] {ev['state']}")
        elif t == "transcript":
            flag = "" if ev["confident"] else "  <-- LOW CONFIDENCE"
            print(f"    [heard] {ev['text']!r} (logprob {ev['avg_logprob']}){flag}")
        elif t == "span":
            print(f"    [span ] {ev['synth_ms']}ms synth / {ev['audio_ms']}ms audio"
                  f"  {ev['text']!r}")
        elif t == "assistant":
            print(f"    [reply] {ev['text']!r}")
        elif t == "error":
            print(f"    [ERROR] {ev['message']}")

    async def send_audio(self, x: np.ndarray, realtime: bool = True) -> None:
        """Stream audio in 512-sample frames, paced like a real microphone."""
        for i in range(0, len(x) - FRAME, FRAME):
            await self.ws.send(to_pcm16(x[i:i + FRAME]))
            if realtime:
                await asyncio.sleep(FRAME_MS / 1000)

    def metrics(self) -> dict | None:
        for ev in reversed(self.events):
            if ev["type"] == "metrics":
                return ev["metrics"]
        return None

    def states(self) -> list[str]:
        return [e["state"] for e in self.events if e["type"] == "state"]


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

async def scenario_normal(ws, speech: np.ndarray) -> tuple[bool, dict]:
    print("\n" + "=" * 70)
    print("SCENARIO 1 — a normal answer, then the candidate stops")
    print("=" * 70)
    c = Client(ws)
    task = asyncio.create_task(c.reader())

    await c.send_audio(silence(400))
    print(f"    speaking {len(speech)/16000:.1f}s of audio...")
    await c.send_audio(speech)
    # Long enough to trip the 1300 ms endpoint and hear the whole reply.
    await c.send_audio(silence(2200))
    await asyncio.sleep(6)
    task.cancel()

    m = c.metrics()
    if not m:
        print("    FAIL — server never completed a turn")
        return False, {}

    print(f"\n    answer length      {m['audio_s']:.1f} s")
    print(f"    STT                {m['stt_ms']:.0f} ms"
          f"  ({'speculative' if m['stt_speculative'] else 'blocking'})")
    print(f"    STT paid post-end  {m['stt_wait_ms']:.0f} ms   <- should be ~0")
    print(f"    LLM first token    {m['llm_ttft_ms']:.0f} ms")
    print(f"    LLM first span     {m['llm_first_span_ms']:.0f} ms")
    print(f"    TTS first audio    {m['tts_first_ms']:.0f} ms")
    print(f"    ENDPOINT -> AUDIO  {m['first_audio_ms']:.0f} ms   (budget 1500)")
    return True, m


async def scenario_pause(ws, speech: np.ndarray) -> bool:
    """A thinking pause must NOT be treated as the end of the answer."""
    print("\n" + "=" * 70)
    print("SCENARIO 2 — a 900 ms thinking pause mid-answer (must not cut off)")
    print("=" * 70)
    c = Client(ws)
    task = asyncio.create_task(c.reader())

    half = len(speech) // 2
    await c.send_audio(silence(300))
    await c.send_audio(speech[:half])
    print("    ...pausing 900 ms mid-sentence...")
    await c.send_audio(silence(900))          # under the 1300 ms threshold
    await c.send_audio(speech[half:])
    await c.send_audio(silence(2200))
    await asyncio.sleep(6)
    task.cancel()

    turns = sum(1 for e in c.events if e["type"] == "metrics")
    heard = [e["text"] for e in c.events if e["type"] == "transcript"]
    ok = turns == 1
    print(f"\n    turns triggered: {turns}  (must be 1)")
    if heard:
        print(f"    transcript: {heard[0][:110]!r}")
    print(f"    {'PASS' if ok else 'FAIL'} — "
          f"{'survived the pause' if ok else 'cut the candidate off mid-answer'}")
    return ok


# --------------------------------------------------------------------------- #

async def main() -> int:
    import websockets

    print("rendering synthetic speech with Piper...")
    speech = speech_16k(ANSWER)
    print(f"  {len(speech)/16000:.1f} s @ 16 kHz")

    wav = ROOT / "bench" / "_phase1_answer.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(to_pcm16(speech))

    try:
        async with websockets.connect(WS_URL, max_size=None) as ws:
            hello = json.loads(await ws.recv())
            print(f"connected — endpoint={hello['endpoint_silence_ms']}ms "
                  f"rate={hello['sample_rate']}")
            await ws.send(json.dumps({
                "type": "config",
                "vocabulary": "mutex, semaphore, mutual exclusion, thread, "
                              "deadlock, atomic, concurrency",
            }))
            ok1, m = await scenario_normal(ws, speech)
    except OSError:
        print("\nserver not reachable at 127.0.0.1:8000 — start it first:")
        print("  .venv/Scripts/python.exe -m coach.server")
        return 1

    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.recv()
        ok2 = await scenario_pause(ws, speech)

    print("\n" + "=" * 70)
    print("PHASE 1 EXIT CRITERIA")
    print("=" * 70)
    checks = {
        "turn completed end to end": (ok1, "yes" if ok1 else "no"),
        "endpoint -> first audio < 1500 ms":
            (bool(m) and m["first_audio_ms"] < 1500,
             f"{m.get('first_audio_ms', 0):.0f} ms" if m else "n/a"),
        "STT cost hidden by speculation (< 150 ms after endpoint)":
            (bool(m) and m["stt_wait_ms"] < 150,
             f"{m.get('stt_wait_ms', 0):.0f} ms" if m else "n/a"),
        "thinking pause did not cut the answer off": (ok2, "yes" if ok2 else "no"),
    }
    for label, (good, val) in checks.items():
        print(f"  [{'PASS' if good else 'FAIL'}] {label:55s} {val}")

    (ROOT / "bench" / "phase1_results.json").write_text(json.dumps(
        {"metrics": m, "checks": {k: v[0] for k, v in checks.items()}}, indent=2))
    return 0 if all(v[0] for v in checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
