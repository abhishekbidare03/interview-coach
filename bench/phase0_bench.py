"""Phase 0 — measure the hardware before building anything on top of it.

Every latency number in plan.md is an estimate. This script replaces them with
measurements from this specific machine.

Exit criteria (plan.md §4, Phase 0):
  * STT + LLM co-resident under ~3.2 GB VRAM
  * LLM time-to-first-token under 500 ms

Run:  uv run python bench/phase0_bench.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VOICE = ROOT / "data" / "voices" / "en_US-lessac-high.onnx"
OLLAMA = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:3b"

# Deliberately jargon-heavy: this is the STT failure mode that matters most
# (plan.md §2.4.1 — a misheard technical term gets graded as a wrong answer).
STT_PROBE = (
    "A mutex guarantees mutual exclusion, whereas a semaphore is a counter. "
    "In PostgreSQL, an idempotent migration can run twice without side effects. "
    "Kubernetes reschedules the pod when the readiness probe fails."
)


# --------------------------------------------------------------------------- #
# VRAM
# --------------------------------------------------------------------------- #

def vram_used_mb() -> int:
    """Total VRAM in use across the GPU, in MiB. Includes other processes."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


class VramPeak:
    """Polls total GPU memory in a background thread and records the peak.

    We poll device-wide usage rather than per-process because the budget that
    actually matters is 'does everything fit in 4 GB at once', and Ollama holds
    its weights in a separate process from this one.
    """

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VramPeak":
        self.peak = vram_used_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, vram_used_mb())
            except Exception:  # nvidia-smi can transiently fail under load
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class Results:
    gpu: dict = field(default_factory=dict)
    tts: dict = field(default_factory=dict)
    stt: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    coresident: dict = field(default_factory=dict)
    verdict: dict = field(default_factory=dict)


def section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


# --------------------------------------------------------------------------- #
# 1. TTS  (also produces the audio fixture the STT benchmark consumes)
# --------------------------------------------------------------------------- #

def synth(voice, text: str) -> tuple[bytes, int, float]:
    """Synthesize `text`. Returns (pcm16 bytes, sample_rate, time-to-first-chunk).

    piper-tts changed its streaming API between 1.2 and 1.3; support both so a
    version bump does not silently break the pipeline.
    """
    t0 = time.perf_counter()
    ttfa = None
    chunks: list[bytes] = []
    rate = 22050

    if hasattr(voice, "synthesize"):  # piper-tts >= 1.3 -> AudioChunk objects
        for chunk in voice.synthesize(text):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(chunk.audio_int16_bytes)
            rate = chunk.sample_rate
    else:  # piper-tts 1.2 -> raw PCM generator
        rate = voice.config.sample_rate
        for raw in voice.synthesize_stream_raw(text):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(raw)

    return b"".join(chunks), rate, ttfa or 0.0


def bench_tts(results: Results) -> Path:
    section("TTS — Piper en_US-lessac-high (CPU)")
    from piper import PiperVoice

    t0 = time.perf_counter()
    voice = PiperVoice.load(str(VOICE))
    load_s = time.perf_counter() - t0
    print(f"load: {load_s * 1000:.0f} ms")

    # The number that matters is time-to-first-audio on a SHORT sentence,
    # because streaming TTS (plan.md §2.3) only ever feeds it one sentence.
    short = "That's not quite right."
    ttfas, rtfs = [], []
    for i in range(3):
        pcm, rate, ttfa = synth(voice, short)
        audio_s = len(pcm) / 2 / rate
        wall = None
        t = time.perf_counter()
        synth(voice, short)
        wall = time.perf_counter() - t
        ttfas.append(ttfa * 1000)
        rtfs.append(audio_s / wall)
        print(f"  run {i + 1}: first audio {ttfa * 1000:6.0f} ms | "
              f"{audio_s:.2f}s audio | {audio_s / wall:5.1f}x realtime")

    pcm, rate, _ = synth(voice, STT_PROBE)
    fixture = ROOT / "bench" / "_probe.wav"
    import wave
    with wave.open(str(fixture), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    dur = len(pcm) / 2 / rate
    print(f"\nfixture for STT: {fixture.name} ({dur:.1f}s @ {rate} Hz)")

    results.tts = {
        "load_ms": round(load_s * 1000),
        "first_audio_ms_median": round(statistics.median(ttfas)),
        "realtime_factor_median": round(statistics.median(rtfs), 1),
        "sample_rate": rate,
        "device": "cpu",
        "vram_mb": 0,
    }
    return fixture


# --------------------------------------------------------------------------- #
# 2. STT
# --------------------------------------------------------------------------- #

def bench_stt(results: Results, fixture: Path):
    section("STT — faster-whisper small.en (GPU)")
    from coach import cuda_paths

    added = cuda_paths.register()
    print(f"registered {len(added)} NVIDIA DLL dir(s)")

    from faster_whisper import WhisperModel

    base = vram_used_mb()
    model, compute_type, device = None, None, None

    # int8_float16 is the right trade on Turing: int8 weights halve the memory,
    # float16 compute keeps the encoder fast. Fall back rather than dying so a
    # CUDA problem produces a measurement instead of a stack trace.
    for dev, ct in [("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")]:
        try:
            t0 = time.perf_counter()
            model = WhisperModel("small.en", device=dev, compute_type=ct)
            load_s = time.perf_counter() - t0
            device, compute_type = dev, ct
            break
        except Exception as e:
            print(f"  {dev}/{ct} failed: {str(e).splitlines()[0][:110]}")

    if model is None:
        results.stt = {"error": "no working backend"}
        return None

    print(f"backend: {device}/{compute_type}  load {load_s:.1f}s")

    # Warm up: the first transcribe pays for CUDA context + kernel autotuning.
    list(model.transcribe(str(fixture), beam_size=1)[0])
    loaded_vram = vram_used_mb()

    lat, texts = [], []
    for i in range(3):
        t0 = time.perf_counter()
        segs, info = model.transcribe(
            str(fixture), beam_size=1, language="en", vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segs)
        el = time.perf_counter() - t0
        lat.append(el * 1000)
        texts.append(text)
        print(f"  run {i + 1}: {el * 1000:6.0f} ms for {info.duration:.1f}s audio "
              f"({info.duration / el:5.1f}x realtime)")

    print(f"\ntranscript: {texts[-1]}")
    vram = loaded_vram - base
    print(f"VRAM delta: {vram} MiB")

    results.stt = {
        "model": "small.en",
        "device": device,
        "compute_type": compute_type,
        "load_s": round(load_s, 1),
        "latency_ms_median": round(statistics.median(lat)),
        "audio_s": round(info.duration, 1),
        "vram_mb": vram,
        "transcript": texts[-1],
    }
    return model


# --------------------------------------------------------------------------- #
# 3. LLM
# --------------------------------------------------------------------------- #

def bench_llm(results: Results):
    section(f"LLM — Ollama {LLM_MODEL} (GPU)")
    import httpx

    base = vram_used_mb()

    # KEEP_ALIVE=-1 pins the model in VRAM. Without it Ollama evicts after 5
    # minutes and every turn after a pause pays a multi-second cold load.
    prompt = ("You are an interviewer. The candidate said a mutex is the same as "
              "a semaphore. In two sentences, correct them.")

    ttfts, tps = [], []
    for i in range(3):
        t0 = time.perf_counter()
        ttft = None
        ntok = 0
        with httpx.stream(
            "POST", f"{OLLAMA}/api/chat", timeout=120,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "keep_alive": -1,
                "options": {"num_ctx": 8192, "temperature": 0.3},
            },
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if d.get("message", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
                if d.get("done"):
                    break
        total = time.perf_counter() - t0
        gen_tps = (ntok - 1) / (total - (ttft or 0)) if total > (ttft or 0) else 0
        ttfts.append((ttft or 0) * 1000)
        tps.append(gen_tps)
        print(f"  run {i + 1}: TTFT {(ttft or 0) * 1000:6.0f} ms | "
              f"{ntok:3d} tok | {gen_tps:5.1f} tok/s")

    loaded = vram_used_mb()
    vram = loaded - base
    print(f"\nVRAM delta: {vram} MiB")

    results.llm = {
        "model": LLM_MODEL,
        "num_ctx": 8192,
        "ttft_ms_median": round(statistics.median(ttfts)),
        "tokens_per_s_median": round(statistics.median(tps), 1),
        "vram_mb": vram,
    }


# --------------------------------------------------------------------------- #
# 4. Everything at once — the number that decides the architecture
# --------------------------------------------------------------------------- #

def bench_coresident(results: Results, stt_model, fixture: Path):
    section("CO-RESIDENT — STT + LLM + TTS under simultaneous load")
    import httpx
    from piper import PiperVoice

    voice = PiperVoice.load(str(VOICE))

    with VramPeak() as peak:
        t0 = time.perf_counter()
        segs, _ = stt_model.transcribe(str(fixture), beam_size=1, language="en")
        text = " ".join(s.text.strip() for s in segs)
        t_stt = time.perf_counter() - t0

        # Simulate one full interview turn end to end, including the streaming
        # sentence-split that Phase 1 depends on.
        t0 = time.perf_counter()
        buf, first_sentence, t_first_sentence = "", None, None
        with httpx.stream(
            "POST", f"{OLLAMA}/api/chat", timeout=120,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content":
                              f"Briefly evaluate this interview answer: {text}"}],
                "stream": True, "keep_alive": -1,
                "options": {"num_ctx": 8192, "temperature": 0.3},
            },
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                buf += d.get("message", {}).get("content", "")
                if first_sentence is None:
                    for stop in ".!?":
                        if stop in buf:
                            cut = buf.index(stop) + 1
                            first_sentence = buf[:cut].strip()
                            t_first_sentence = time.perf_counter() - t0
                            break
                if d.get("done"):
                    break

        t_tts0 = time.perf_counter()
        _, _, ttfa = synth(voice, first_sentence or "Let me think about that.")
        t_tts = time.perf_counter() - t_tts0

    turn_ms = (t_stt + (t_first_sentence or 0) + ttfa) * 1000
    print(f"STT                       {t_stt * 1000:7.0f} ms")
    print(f"LLM -> first sentence     {(t_first_sentence or 0) * 1000:7.0f} ms")
    print(f"TTS -> first audio        {ttfa * 1000:7.0f} ms")
    print(f"{'-' * 40}")
    print(f"answer-end -> first audio {turn_ms:7.0f} ms   (VAD not included)")
    print(f"\nfirst sentence spoken: {first_sentence!r}")
    print(f"PEAK VRAM (device-wide):  {peak.peak} MiB")

    results.coresident = {
        "stt_ms": round(t_stt * 1000),
        "llm_first_sentence_ms": round((t_first_sentence or 0) * 1000),
        "tts_first_audio_ms": round(ttfa * 1000),
        "turn_latency_ms": round(turn_ms),
        "peak_vram_mb": peak.peak,
        "first_sentence": first_sentence,
    }


# --------------------------------------------------------------------------- #

def main() -> int:
    if not VOICE.exists():
        print(f"missing voice model: {VOICE}", file=sys.stderr)
        return 1

    results = Results()
    info = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,memory.used,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    name, total, used, driver = (x.strip() for x in info.split(","))
    print(f"GPU: {name} | {total} total | {used} in use | driver {driver}")
    results.gpu = {"name": name, "total": total, "baseline_used": used,
                   "driver": driver}

    fixture = bench_tts(results)
    stt_model = bench_stt(results, fixture)
    bench_llm(results)
    if stt_model is not None:
        bench_coresident(results, stt_model, fixture)

    # ---- verdict against plan.md Phase 0 exit criteria ----
    section("VERDICT vs plan.md Phase 0 exit criteria")
    peak = results.coresident.get("peak_vram_mb", 0)
    ttft = results.llm.get("ttft_ms_median", 9999)
    checks = {
        "co-resident peak VRAM < 3200 MiB": (peak < 3200, f"{peak} MiB"),
        "LLM TTFT < 500 ms": (ttft < 500, f"{ttft} ms"),
        "STT ran on GPU": (results.stt.get("device") == "cuda",
                           results.stt.get("device", "?")),
        "turn latency < 1500 ms": (
            results.coresident.get("turn_latency_ms", 9999) < 1500,
            f"{results.coresident.get('turn_latency_ms', '?')} ms"),
    }
    for label, (ok, val) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:38s} -> {val}")
    results.verdict = {k: {"pass": v[0], "value": v[1]} for k, v in checks.items()}

    out = ROOT / "bench" / "phase0_results.json"
    out.write_text(json.dumps(results.__dict__, indent=2))
    print(f"\nwrote {out}")
    return 0 if all(v[0] for v in checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
