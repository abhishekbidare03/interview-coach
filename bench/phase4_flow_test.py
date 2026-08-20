"""Phase 4 verification — the adaptive flow, with the model stubbed out.

The interview state machine has rules that are easy to get subtly wrong and hard
to see in a live session: when a follow-up fires, how difficulty moves, whether a
question can repeat, whether the session ends where it should. Those are
decisions in code, not model behaviour, so they are tested against a scripted
grader rather than a real one — deterministic, instant, and no GPU.

Model *quality* is Phase 3's job. This file only checks the machinery around it.

Run:  .venv/Scripts/python.exe bench/phase4_flow_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coach.bank import Bank                              # noqa: E402
from coach.blueprint import build                        # noqa: E402
from coach.evaluate import Evaluation                    # noqa: E402
from coach.interview import Interview, Stage             # noqa: E402
from coach.schema import Mode, Verdict                   # noqa: E402


class ScriptedEvaluator:
    """Returns a fixed sequence of verdicts so flow can be asserted exactly."""

    def __init__(self, verdicts: list[Verdict]) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0

    async def evaluate(self, q, answer: str) -> Evaluation:
        v = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        # A good verdict covers everything; anything less leaves a gap, which is
        # what should_follow_up keys off.
        missing = [] if v.is_good else list(q.expected_points[:1])
        return Evaluation(
            verdict=v,
            score=1.0 if v.is_good else (0.5 if not v.is_poor else 0.1),
            covered=q.expected_points if v.is_good else q.expected_points[1:],
            missing=missing,
            should_follow_up=bool(missing) and not v.is_poor,
            opening_line="(opener)",
        )


class NullLLM:
    async def stream(self, *_a, **_k):
        for tok in ("Detail ", "goes ", "here."):
            yield tok

    async def complete(self, *_a, **_k) -> str:
        return "ok"


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


async def run(bank: Bank, verdicts: list[Verdict], length: int = 6,
              mixed: bool = True, seed: int = 5):
    bp = await build(bank, bank.topics, length=length, mixed=mixed, seed=seed)
    iv = Interview(bp, NullLLM(), evaluator=ScriptedEvaluator(verdicts))
    await iv.opening()

    asked, follow_ups, guard = [], 0, 0
    while guard < 40:
        guard += 1
        text = await iv.next_question()
        if text is None:
            break
        was_fu = text not in (iv._phrase(t.question) for t in iv.state.turns) \
            and iv.state.turns and iv.state.pending_follow_up is None \
            and not text.startswith(("Let's start", "Next question", "Last one",
                                     "Now something"))
        asked.append(text)
        follow_ups += bool(was_fu)
        await iv.submit("an answer", seconds=20.0, was_follow_up=bool(was_fu))
        # Drain the spoken feedback the way the server will.
        async for _ in iv.feedback(iv.state.turns[-1].evaluation, "an answer"):
            pass
    return iv, asked, follow_ups


async def main() -> int:
    bank = Bank()
    if len(bank) < 8:
        print("bank too small — run: python scripts/generate_bank.py --all")
        return 1

    results: dict[str, bool] = {}

    # -- 1. a clean run, every answer strong -------------------------------- #
    section("1. ALL-STRONG RUN — no follow-ups, reaches the end")
    iv, asked, fus = await run(bank, [Verdict.CORRECT, Verdict.STRONG] * 10)
    n = iv.state.blueprint.length
    print(f"  questions asked: {len(asked)}  follow-ups: {fus}")
    print(f"  stage: {iv.state.stage}  turns recorded: {len(iv.state.turns)}")
    ok = iv.state.stage is Stage.FINISHED and len(iv.state.turns) == n and fus == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] strong answers never trigger a follow-up")
    results["strong answers do not trigger follow-ups"] = ok

    # -- 2. partial answers must be probed ---------------------------------- #
    section("2. ALL-PARTIAL RUN — every question gets exactly one follow-up")
    iv, asked, fus = await run(bank, [Verdict.PARTIAL, Verdict.ADEQUATE] * 20)
    n = iv.state.blueprint.length
    print(f"  questions in blueprint: {n}")
    print(f"  utterances asked: {len(asked)}  of which follow-ups: {fus}")
    # One follow-up per question, never a follow-up to a follow-up.
    ok = fus == n and len(asked) == 2 * n
    print(f"  [{'PASS' if ok else 'FAIL'}] exactly one follow-up per question, "
          f"never chained")
    results["one follow-up per question, never chained"] = ok

    # -- 3. weak answers move on rather than piling on ---------------------- #
    section("3. ALL-WEAK RUN — struggling candidate is not interrogated")
    iv, asked, fus = await run(bank, [Verdict.INCORRECT, Verdict.WEAK] * 20)
    print(f"  utterances asked: {len(asked)}  follow-ups: {fus}")
    ok = fus == 0 and iv.state.stage is Stage.FINISHED
    print(f"  [{'PASS' if ok else 'FAIL'}] weak answers move on instead of probing")
    results["weak answers move on"] = ok

    # -- 4. difficulty responds to performance ------------------------------ #
    # Difficulty moved out of the state machine and into interviewer.py, where
    # an LLM call decides it. What is checked here is the arithmetic rule that
    # backs that call, because it is what runs when Ollama is unreachable and
    # it must not need the model to be sane. The adaptation as a whole — the
    # decision, and the harder question actually being asked — is covered by
    # bench/phase8_adaptive_test.py, which does need Ollama.
    section("4. DIFFICULTY ADJUSTMENT (offline fallback rule)")
    from coach.interviewer import Interviewer, _arithmetic_move

    def offline(script: list[Verdict], score: float) -> int:
        iv = Interviewer(llm=None)
        for v in script:
            iv.read.record(v, score)
            iv.offset = max(-1, min(2, iv.offset +
                                    {"easier": -1, "same": 0, "harder": 1}
                                    [_arithmetic_move(iv.read)]))
        return iv.offset

    up = offline([Verdict.CORRECT] * 5, 1.0)
    down = offline([Verdict.INCORRECT] * 5, 0.1)
    wobble = offline([Verdict.CORRECT, Verdict.CORRECT, Verdict.CORRECT,
                      Verdict.INCORRECT, Verdict.CORRECT], 0.9)
    print(f"  after all-correct    : offset {up:+d}")
    print(f"  after all-wrong      : offset {down:+d}")
    print(f"  after one stumble    : offset {wobble:+d}")
    ok = up > 0 > down and wobble > down
    print(f"  [{'PASS' if ok else 'FAIL'}] difficulty moves with performance")
    results["difficulty adapts"] = ok

    # -- 5. no question is ever repeated ------------------------------------ #
    section("5. REPETITION GUARD")
    iv, *_ = await run(bank, [Verdict.PARTIAL] * 30, length=10)
    ids = [t.question.id for t in iv.state.turns]
    uniq = len(set(ids))
    print(f"  {len(ids)} turns over {uniq} distinct questions")
    # Partial answers produce two turns per question (answer + follow-up), so
    # distinct questions should be exactly half.
    ok = uniq == iv.state.blueprint.length
    print(f"  [{'PASS' if ok else 'FAIL'}] every blueprint question asked once")
    results["no repeated questions"] = ok

    # -- 6. behavioural questions never get a factual verdict --------------- #
    section("6. MODE INTEGRITY (§2.1)")
    iv, *_ = await run(bank, [Verdict.ADEQUATE] * 30, length=10)
    bad = [t for t in iv.state.turns
           if t.question.mode is Mode.BEHAVIOURAL
           and t.evaluation.verdict in (Verdict.CORRECT, Verdict.INCORRECT,
                                        Verdict.PARTIAL)]
    beh = sum(1 for t in iv.state.turns if t.question.mode is Mode.BEHAVIOURAL)
    print(f"  behavioural turns: {beh}   graded with a factual verdict: {len(bad)}")
    ok = not bad
    print(f"  [{'PASS' if ok else 'FAIL'}] behavioural answers never judged "
          f"correct/incorrect")
    results["behavioural never graded as correct/incorrect"] = ok

    # -- 7. the report ------------------------------------------------------ #
    section("7. SESSION SUMMARY (feeds Phase 6)")
    iv, *_ = await run(bank, [Verdict.CORRECT, Verdict.PARTIAL, Verdict.INCORRECT] * 10)
    s = iv.summary()
    print(json.dumps(s, indent=2))
    print(f"\n  closing line: {iv.closing_line()}")
    ok = s["questions"] > 0 and "weakest_topic" in s and 0 <= s["mean_score"] <= 1
    print(f"  [{'PASS' if ok else 'FAIL'}] summary is well formed")
    results["summary well formed"] = ok

    section("PHASE 4 EXIT CRITERIA")
    for label, good in results.items():
        print(f"  [{'PASS' if good else 'FAIL'}] {label}")

    (ROOT / "bench" / "phase4_results.json").write_text(json.dumps(results, indent=2))
    return 0 if all(results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
