"""End-to-end verification — a whole interview through the real WebSocket.

Everything before this tested one layer at a time: Phase 1 the voice loop with no
interview logic, Phase 3 the grader with no audio, Phase 4 the flow with no
model. This drives the actual server the browser talks to, start to finish:
setup -> blueprint -> spoken questions -> spoken answers -> grading -> follow-ups
-> report.

Answers are synthesized with Piper and streamed in at realtime pace, so the VAD,
the endpointer, and the speculative-transcription path all do real work. The
answers are deliberately of mixed quality, so grading, follow-ups, and difficulty
adaptation all have something to react to.

Usage:
    # terminal 1
    PYTHONPATH=src .venv/Scripts/python.exe -m coach.server
    # terminal 2
    .venv/Scripts/python.exe bench/phase5_e2e_test.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import struct
import pathlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WS_URL = "ws://127.0.0.1:8000/ws"
FRAME = 512
FRAME_MS = FRAME / 16000 * 1000

# Answers are built from the asked question's own reference answer, truncated to
# a varying fraction. That is the only way to get *on-topic* answers of known,
# graded completeness without a human: a fixed list of canned answers is
# off-topic for whatever question the blueprint happens to pick, so every verdict
# comes back "incorrect" and follow-ups never fire.
#
# Fractions cycle full -> half -> none, so each run exercises a strong answer, a
# partial one (which should trigger a follow-up), and a non-answer.
COVERAGE_CYCLE = (1.0, 0.5, 0.0)

FALLBACK = [
    "The main difference comes down to how the data is laid out in memory and "
    "what that costs you. One gives you constant time access by index because "
    "everything is contiguous, but inserting in the middle means shifting "
    "everything after it. The other trades that away, so insertion is cheap once "
    "you have a reference, but you lose random access and you pay for pointer "
    "chasing and cache misses.",

    "I think it depends on the situation, and you would normally pick whichever "
    "one fits the access pattern you expect to have.",

    "Honestly I am not certain about that one. I have used it before but I could "
    "not tell you exactly how it works underneath.",
]


def load_bank() -> dict[str, dict]:
    """Question text -> its bank entry, so answers can be built from it."""
    import glob
    out = {}
    for f in glob.glob(str(ROOT / "data" / "bank" / "*.json")):
        for q in json.loads(pathlib.Path(f).read_text(encoding="utf-8")):
            out[q["text"].strip()] = q
    return out


def answer_for(question_text: str, bank: dict, fraction: float) -> str:
    """Build an on-topic answer covering roughly `fraction` of the checklist."""
    entry = None
    for text, q in bank.items():
        if text and text in question_text:
            entry = q
            break
    if entry is None or fraction == 0.0:
        return FALLBACK[0 if fraction == 0.0 else 1]

    ref = entry.get("reference_answer", "")
    sentences = [s.strip() for s in ref.split(". ") if s.strip()]
    keep = max(1, int(len(sentences) * fraction))
    said = ". ".join(sentences[:keep])
    return f"So, {said}." if said else FALLBACK[1]


_VOICE = None


def render(text: str) -> np.ndarray:
    global _VOICE
    from piper import PiperVoice
    voice = _VOICE or PiperVoice.load(
        str(ROOT / "data" / "voices" / "en_US-lessac-high.onnx"))
    _VOICE = voice
    parts, rate = [], 22050
    for c in voice.synthesize(text):
        parts.append(c.audio_int16_bytes)
        rate = c.sample_rate
    pcm = np.frombuffer(b"".join(parts), dtype=np.int16).astype(np.float32) / 32768
    n = int(len(pcm) * 16000 / rate)
    return np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)),
                     pcm).astype(np.float32)


def silence(ms: int) -> np.ndarray:
    return (np.random.randn(int(16000 * ms / 1000)) * 0.0008).astype(np.float32)


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


class Runner:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.questions: list[dict] = []
        self.evaluations: list[dict] = []
        self.transcripts: list[dict] = []
        self.summary: dict | None = None
        self.blueprint: dict | None = None
        self.errors: list[str] = []
        self.metrics: list[dict] = []
        self.first_audio_ms: list[float] = []
        self.speaking = asyncio.Event()
        self.idle = asyncio.Event()
        self._t_answer_end: float | None = None
        self._got_audio = False

    async def reader(self) -> None:
        async for msg in self.ws:
            if isinstance(msg, bytes):
                if not self._got_audio and self._t_answer_end:
                    self.first_audio_ms.append(
                        (time.perf_counter() - self._t_answer_end) * 1000)
                    self._got_audio = True
                continue

            ev = json.loads(msg)
            k = ev.get("type")
            if k == "blueprint":
                self.blueprint = ev
                print(f"\n  blueprint: {ev['title']} — {len(ev['questions'])} "
                      f"questions, topics={ev['topics']}")
                print(f"  curve: {ev['difficulty_curve']}")
                print(f"  vocab: {ev['vocabulary'][:64]}...")
            elif k == "question":
                self.questions.append(ev)
                tag = " [FOLLOW-UP]" if ev.get("follow_up") else ""
                print(f"\n  Q{ev['position']}/{ev['total']}{tag} "
                      f"[{ev.get('topic')}/{ev.get('mode')} d{ev.get('difficulty')}]")
                print(f"    {ev['text'][:110]}")
            elif k == "transcript":
                self.transcripts.append(ev)
                print(f"    heard: {ev['text'][:80]!r} "
                      f"(logprob {ev.get('avg_logprob')})")
            elif k == "evaluation":
                self.evaluations.append(ev)
                print(f"    -> {ev['verdict'].upper()} score {ev['score']} "
                      f"({len(ev['covered'])} covered, {len(ev['missing'])} missing)"
                      f"{' follow-up' if ev['should_follow_up'] else ''}"
                      f"  {ev['grade_ms']}ms")
            elif k == "assistant":
                print(f"    says: {ev['text'][:110]}")
            elif k == "state":
                if ev["state"] == "speaking":
                    self.speaking.set()
                if ev["state"] == "idle":
                    self.idle.set()
            elif k == "metrics":
                m = ev["metrics"]
                self.metrics.append(m)
                print(f"    timing: stt {m['stt_ms']:.0f}ms"
                      f"{' (spec)' if m['stt_speculative'] else ' BLOCKING'}"
                      f" | after endpoint {m['stt_wait_ms']:.0f}ms"
                      f" | llm first span {m['llm_first_span_ms']:.0f}ms"
                      f" | tts {m['tts_first_ms']:.0f}ms"
                      f" | ENDPOINT->AUDIO {m['first_audio_ms']:.0f}ms")
            elif k == "finished":
                self.summary = ev["summary"]
            elif k == "error":
                self.errors.append(ev["message"])
                print(f"    ERROR: {ev['message']}")

    async def send(self, audio: np.ndarray) -> None:
        for i in range(0, len(audio) - FRAME, FRAME):
            await self.ws.send(pcm16(audio[i:i + FRAME]))
            await asyncio.sleep(FRAME_MS / 1000)

    async def answer(self, audio: np.ndarray) -> None:
        """Speak one answer and wait for the coach to finish responding."""
        self.idle.clear()
        self.speaking.clear()
        self._got_audio = False
        await self.send(silence(300))
        await self.send(audio)
        self._t_answer_end = time.perf_counter()
        await self.send(silence(1800))          # trip the 1300 ms endpoint
        try:
            await asyncio.wait_for(self.idle.wait(), timeout=90)
        except asyncio.TimeoutError:
            print("    (timed out waiting for the coach to finish)")


async def main() -> int:
    import websockets

    bank = load_bank()
    print(f"loaded {len(bank)} bank questions to build answers from")

    try:
        ws = await websockets.connect(WS_URL, max_size=None)
    except OSError:
        print("\nserver not reachable — start it first:\n"
              "  PYTHONPATH=src .venv/Scripts/python.exe -m coach.server")
        return 1

    async with ws:
        hello = json.loads(await ws.recv())
        print(f"connected — bank={hello.get('bank')} questions")
        if not hello.get("bank"):
            print("bank is empty; run scripts/generate_bank.py")
            return 1

        r = Runner(ws)
        task = asyncio.create_task(r.reader())

        print("\n" + "=" * 72)
        print("RANDOM INTERVIEW — 5 questions, mixed topics (R3)")
        print("=" * 72)
        t0 = time.perf_counter()
        r.idle.clear()
        await ws.send(json.dumps({"type": "start", "mixed": True,
                                  "length": 5, "difficulty": 3}))
        await asyncio.wait_for(r.idle.wait(), timeout=180)
        print(f"\n  (opening spoken; setup took {time.perf_counter() - t0:.1f}s)")

        # Answer until the interview ends, cycling coverage so that strong,
        # partial, and empty answers all occur.
        for i in range(14):
            if r.summary is not None:
                break
            asked = r.questions[-1]["text"] if r.questions else ""
            frac = COVERAGE_CYCLE[i % len(COVERAGE_CYCLE)]
            text = answer_for(asked, bank, frac)
            print(f"\n  [answering at {frac:.0%} coverage: {text[:66]!r}]")
            await r.answer(render(text))

        if r.summary is None:
            await ws.send(json.dumps({"type": "end"}))
            await asyncio.sleep(2)
        task.cancel()

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 72)
    print("SESSION SUMMARY")
    print("=" * 72)
    print(json.dumps(r.summary, indent=2))

    n_planned = len(r.blueprint["questions"]) if r.blueprint else 0
    followups = sum(1 for q in r.questions if q.get("follow_up"))
    modes = {q.get("mode") for q in r.questions}
    verdicts = [e["verdict"] for e in r.evaluations]
    med_first = statistics.median(r.first_audio_ms) if r.first_audio_ms else 0
    med_grade = statistics.median([e["grade_ms"] for e in r.evaluations]) \
        if r.evaluations else 0
    med_server_first = statistics.median(
        [m["first_audio_ms"] for m in r.metrics]) if r.metrics else 0
    med_first_span = statistics.median(
        [m["llm_first_span_ms"] for m in r.metrics]) if r.metrics else 9999

    print("\n" + "=" * 72)
    print("END-TO-END EXIT CRITERIA")
    print("=" * 72)
    checks = {
        "interview planned and started": (r.blueprint is not None,
                                          f"{n_planned} questions"),
        "every planned question was asked":
            (len([q for q in r.questions if not q.get("follow_up")]) == n_planned,
             f"{len([q for q in r.questions if not q.get('follow_up')])}/{n_planned}"),
        "answers transcribed": (len(r.transcripts) >= n_planned,
                                f"{len(r.transcripts)}"),
        "answers graded": (len(r.evaluations) >= n_planned, f"{len(r.evaluations)}"),
        "verdicts vary with answer quality":
            (len(set(verdicts)) >= 2, f"{sorted(set(verdicts))}"),
        "follow-ups fired on partial answers": (followups > 0, f"{followups}"),
        "session reached a report": (r.summary is not None,
                                     "yes" if r.summary else "no"),
        "no errors": (not r.errors, f"{len(r.errors)}"),
        # Measured server-side from the endpoint firing, which is what
        # plan.md §2.3 budgets. The client-side figure would also include the
        # 1300 ms endpoint-silence window, but that is deliberate design (§2.4.3)
        # and the candidate is silent through it — counting it as lag would be
        # measuring the feature as if it were the bug.
        "median endpoint -> first audio < 1500 ms":
            (0 < med_server_first < 1500, f"{med_server_first:.0f} ms"),
        # Informational, not on the critical path any more: grading now runs
        # under the spoken acknowledgement.
        "grading fully hidden behind speech":
            (med_first_span < 400, f"first span at {med_first_span:.0f} ms"),
    }
    for label, (ok, val) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:44s} {val}")
    print(f"\n  modes exercised: {sorted(m for m in modes if m)}")

    (ROOT / "bench" / "phase5_results.json").write_text(json.dumps(
        {"summary": r.summary, "checks": {k: v[0] for k, v in checks.items()},
         "first_audio_ms": r.first_audio_ms,
         "verdicts": verdicts}, indent=2))
    return 0 if all(v[0] for v in checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
