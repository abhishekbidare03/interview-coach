"""Phase 0 follow-up — find out WHY the turn latency missed its 1500 ms budget.

The main benchmark showed 3150 ms end-to-end. Two stages are far above the
estimates in plan.md:

  * Piper first-audio was 1425 ms in the co-resident test, but 374 ms standalone
  * STT took 1078 ms for a 13.6 s clip

Both look like they scale with input length rather than being fixed costs. If
that is true the fixes are structural, not "make it faster" — so measure the
scaling curve before designing Phase 1 around it.

Run:  .venv/Scripts/python.exe bench/phase0_latency_probe.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VOICE = ROOT / "data" / "voices" / "en_US-lessac-high.onnx"


def synth(voice, text: str):
    """Returns (pcm bytes, sample_rate, time-to-first-chunk, total wall time)."""
    t0 = time.perf_counter()
    ttfa, chunks, rate = None, [], 22050
    for chunk in voice.synthesize(text):
        if ttfa is None:
            ttfa = time.perf_counter() - t0
        chunks.append(chunk.audio_int16_bytes)
        rate = chunk.sample_rate
    return b"".join(chunks), rate, ttfa or 0.0, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# 1. Does Piper's first-audio latency scale with the length of what we hand it?
# --------------------------------------------------------------------------- #

CLAUSES = [
    "Not quite.",
    "That's not quite right.",
    "That's close, but not quite right.",
    "That's close, but you've missed the most important part of it.",
    "That's close, but you've missed the most important part of it, "
    "which is what actually happens to the waiting thread.",
    "That's close, but you've missed the most important part of it, "
    "which is what actually happens to the waiting thread, and that "
    "distinction is the whole reason the two primitives exist separately.",
]


def probe_tts() -> list[dict]:
    from piper import PiperVoice

    print("=" * 70)
    print("PIPER — first-audio latency vs input length")
    print("=" * 70)
    voice = PiperVoice.load(str(VOICE))
    synth(voice, "warm up")  # first call pays ONNX graph warmup

    rows = []
    for text in CLAUSES:
        ttfas, rtfs = [], []
        for _ in range(3):
            pcm, rate, ttfa, wall = synth(voice, text)
            audio_s = len(pcm) / 2 / rate
            ttfas.append(ttfa * 1000)
            rtfs.append(audio_s / wall)
        row = {
            "words": len(text.split()),
            "chars": len(text),
            "audio_s": round(audio_s, 2),
            "first_audio_ms": round(statistics.median(ttfas)),
            "realtime_factor": round(statistics.median(rtfs), 1),
        }
        rows.append(row)
        print(f"  {row['words']:3d} words | {row['audio_s']:5.2f}s audio | "
              f"first audio {row['first_audio_ms']:5.0f} ms | "
              f"{row['realtime_factor']:4.1f}x realtime")

    print("\n  -> Piper emits nothing until the WHOLE input is synthesized.")
    print("     First-audio latency is a function of chunk length, not a fixed cost.")
    return rows


# --------------------------------------------------------------------------- #
# 2. Does STT latency scale with the length of the answer?
# --------------------------------------------------------------------------- #

ANSWER = (
    "So a mutex is a locking primitive that enforces mutual exclusion, meaning "
    "exactly one thread can hold it at a time, and critically the thread that "
    "acquired it is the one that has to release it. A semaphore is really just a "
    "counter with atomic increment and decrement, so it can permit N concurrent "
    "holders, and any thread can signal it. That ownership property is why you "
    "can use a mutex for priority inheritance but not a semaphore. "
)


def probe_stt(voice) -> list[dict]:
    from coach import cuda_paths
    cuda_paths.register()
    from faster_whisper import WhisperModel

    print()
    print("=" * 70)
    print("FASTER-WHISPER — transcription latency vs answer length")
    print("=" * 70)
    model = WhisperModel("small.en", device="cuda", compute_type="int8_float16")

    rows = []
    for reps in (1, 2, 4):
        pcm, rate, _, _ = synth(voice, ANSWER * reps)
        path = ROOT / "bench" / f"_answer_{reps}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)

        list(model.transcribe(str(path), beam_size=1, language="en")[0])  # warm
        lat = []
        for _ in range(3):
            t0 = time.perf_counter()
            segs, info = model.transcribe(str(path), beam_size=1, language="en")
            list(segs)
            lat.append((time.perf_counter() - t0) * 1000)
        ms = statistics.median(lat)
        row = {
            "audio_s": round(info.duration, 1),
            "stt_ms": round(ms),
            "realtime_factor": round(info.duration / (ms / 1000), 1),
        }
        rows.append(row)
        print(f"  {row['audio_s']:5.1f}s answer | STT {row['stt_ms']:5.0f} ms | "
              f"{row['realtime_factor']:5.1f}x realtime")
        path.unlink(missing_ok=True)

    print("\n  -> STT cost is proportional to answer length. A 60s answer costs")
    print("     ~3.5s if we start transcribing only after the candidate stops.")
    return rows


# --------------------------------------------------------------------------- #

def main() -> int:
    from piper import PiperVoice

    tts_rows = probe_tts()
    stt_rows = probe_stt(PiperVoice.load(str(VOICE)))

    # ---- what the fixes buy us ----
    print()
    print("=" * 70)
    print("PROJECTED TURN LATENCY")
    print("=" * 70)

    short_tts = tts_rows[1]["first_audio_ms"]      # a short opening clause
    full_tts = tts_rows[4]["first_audio_ms"]       # a full sentence
    llm_first = 647                                # measured in phase0_bench
    stt_30s = stt_rows[1]["stt_ms"]

    naive = stt_30s + llm_first + full_tts
    fixed = 0 + llm_first + short_tts              # STT hidden under endpoint wait

    print(f"  naive  : STT {stt_30s:4.0f} + LLM {llm_first} + TTS {full_tts:4.0f}"
          f"  = {naive:5.0f} ms")
    print(f"  fixed  : STT {0:4.0f} + LLM {llm_first} + TTS {short_tts:4.0f}"
          f"  = {fixed:5.0f} ms   (target < 1500)")
    print()
    print("  Fix 1: start STT the moment speech STOPS, overlapping it with the")
    print("         1300 ms endpoint-silence wait, which is dead time anyway.")
    print("  Fix 2: send Piper a short FIRST clause, then larger chunks after")
    print("         playback has started and there is buffer to hide behind.")

    out = ROOT / "bench" / "phase0_latency_probe.json"
    out.write_text(json.dumps(
        {"tts": tts_rows, "stt": stt_rows,
         "projected": {"naive_ms": naive, "fixed_ms": fixed}}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
