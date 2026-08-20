"""Phase 8 verification — the Python bank and the adaptive interviewer.

Two things are being checked, and they are related.

**The bank.** A screening round is mostly basics. The generated Python bank was
neither — no difficulty-1 questions at all, and questions averaging forty words
with two clauses welded together. This asserts the shape of the hand-written
replacement, because "mostly basics, phrased like a person" is a property that
quietly decays as entries get added.

**The adaptation.** The interviewer is supposed to read the *run* rather than
the last answer, and to move the difficulty accordingly. That is checked three
ways: a candidate who answers everything well should climb, one who answers
nothing well should not, and a single stumble inside a good run should not
undo it.

The interviewer half needs Ollama; the bank half does not. Run:

    .venv\\Scripts\\python.exe bench\\phase8_adaptive_test.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coach.bank import Bank                                  # noqa: E402
from coach.blueprint import build, difficulty_curve          # noqa: E402
from coach.interviewer import Interviewer, _clean_lead       # noqa: E402
from coach.llm import LLM                                    # noqa: E402
from coach.schema import Verdict                             # noqa: E402


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------- #

def test_bank(results: dict) -> None:
    section("1. THE PYTHON BANK IS SHAPED LIKE A SCREENING ROUND")
    bank = Bank()
    qs = bank.all_for("python")
    spread = Counter(q.difficulty for q in qs)
    print(f"  {len(qs)} questions, {len({q.subtopic for q in qs})} subtopics")
    for d in sorted(spread):
        print(f"    difficulty {d}: {spread[d]:>3}  {'#' * spread[d]}")

    basic = spread[1] + spread[2]
    results["majority of questions are basic or easy"] = check(
        "majority basic or easy", basic > len(qs) / 2,
        f"{basic}/{len(qs)} = {basic * 100 // len(qs)}%")
    results["every difficulty level is populated"] = check(
        "all five levels populated", all(spread[d] for d in range(1, 6)),
        str(dict(sorted(spread.items()))))

    section("2. QUESTIONS SOUND LIKE A PERSON ASKING THEM")
    lengths = [len(q.text.split()) for q in qs]
    longest = max(qs, key=lambda q: len(q.text.split()))
    print(f"  words per question: median {statistics.median(lengths):.0f}, "
          f"max {max(lengths)}")
    print(f"  longest: {longest.text}")

    results["questions are short enough to speak"] = check(
        "no question over 20 words", max(lengths) <= 20, f"max {max(lengths)}")
    doubles = [q for q in qs if q.text.count("?") > 1]
    results["one question per question"] = check(
        "none contain two questions", not doubles,
        doubles[0].text if doubles else "")

    # The generated bank opened 5 of 8 questions with "Can you explain the".
    # Variety in the opening words is what stops an interview sounding like a
    # form being read out.
    stems = Counter(" ".join(q.text.split()[:3]).lower() for q in qs)
    top, n = stems.most_common(1)[0]
    print(f"  most repeated opening: {top!r} x{n}")
    results["question openings vary"] = check(
        "no opening used by more than a third", n <= len(qs) / 3,
        f"{top!r} x{n}")

    section("3. EXPECTED POINTS ARE GRADEABLE")
    pts = [p for q in qs for p in q.expected_points]
    long_pts = [p for p in pts if len(p.split()) > 12]
    print(f"  {len(pts)} points, median {statistics.median(len(p.split()) for p in pts):.0f} words")
    results["expected points are terse"] = check(
        "no point over 12 words", not long_pts, long_pts[0] if long_pts else "")
    results["every question has a reference answer"] = check(
        "all have a reference answer",
        all(len(q.reference_answer.split()) > 15 for q in qs))


def test_curve(results: dict) -> None:
    section("4. THE DIFFICULTY CURVE CLIMBS, AND THE LEVEL PICKER MATTERS")
    curves = {}
    for base, label in ((2, "Junior"), (3, "Mid-level"), (4, "Senior")):
        c = difficulty_curve(8, base)
        curves[label] = c
        print(f"  {label:<10} {c}")

    results["curve starts below the target level"] = check(
        "opens below the chosen level", curves["Mid-level"][0] < 3,
        f"starts at {curves['Mid-level'][0]}")
    results["curve ends above the target level"] = check(
        "finishes above the chosen level", curves["Mid-level"][-1] > 3,
        f"ends at {curves['Mid-level'][-1]}")
    first_half = curves["Mid-level"][:4]
    results["first half stays easy"] = check(
        "first half at or below the chosen level", max(first_half) <= 3,
        str(first_half))
    # The bug this catches: with the floor clamped at 1, an additive ramp gave
    # Junior and Mid-level byte-identical curves.
    results["difficulty setting changes the curve"] = check(
        "the three settings differ",
        len({tuple(c) for c in curves.values()}) == 3)


def test_lead_filter(results: dict) -> None:
    section("5. THE LEAD-IN FILTER REJECTS WHAT IT SHOULD")
    cases = [
        ("Good, let's push a little further", True,  "a normal lead-in"),
        ("Okay.",                             True,  "very short is fine"),
        ("So what is a decorator?",           False, "contains a question"),
        ("**Next up**",                       False, "markdown"),
        ("Interviewer: Right, moving on",     True,  "speaker label stripped"),
        ("Now consider hash collisions",      False, "names a subject it cannot know"),
        ("Let's talk about decorators",       False, "names a technical term"),
        ("Continue with a harder one",        False, "an instruction, not speech"),
        ("Same topic, one level up",          False, "mentions the topic"),
        (" ".join(["word"] * 20),             False, "too long"),
        ("",                                  False, "empty"),
        (None,                                False, "not a string"),
    ]
    ok = True
    for raw, want, why in cases:
        got = bool(_clean_lead(raw, frozenset({"decorators", "closure"})))
        flag = "ok " if got == want else "BAD"
        print(f"    {flag} {why:<32} -> "
              f"{_clean_lead(raw, frozenset({'decorators', 'closure'}))!r}")
        ok &= got == want
    results["lead-in filter accepts and rejects correctly"] = check(
        "all filter cases behave", ok)


# --------------------------------------------------------------------------- #

async def run_reads(results: dict) -> None:
    section("6. THE INTERVIEWER READS THE RUN, NOT THE LAST ANSWER")
    async with LLM() as llm:
        if not (await llm.health())["has_model"]:
            print("  SKIPPED — ollama is not serving the model")
            return

        async def play(name: str, script: list[tuple[Verdict, float]]) -> int:
            iv = Interviewer(llm)
            moves = []
            for i, (verdict, score) in enumerate(script):
                iv.record(verdict, score)
                d = await iv.decide(iv.level_for(2 + i // 3), len(script) + 1)
                moves.append(f"{d.move[0].upper()}{'*' if d.source == 'guard' else ''}")
            print(f"  {name:<22} {' '.join(moves)}   final offset {iv.offset:+d}")
            return iv.offset

        strong = await play("all strong", [(Verdict.CORRECT, 1.0)] * 5)
        weak = await play("all weak", [(Verdict.INCORRECT, 0.1)] * 5)
        wobble = await play("strong, one stumble",
                            [(Verdict.CORRECT, 1.0), (Verdict.CORRECT, 0.9),
                             (Verdict.CORRECT, 1.0), (Verdict.INCORRECT, 0.2),
                             (Verdict.CORRECT, 0.9)])
        print("  (* = a move the guard overrode)")

        results["a strong candidate gets harder questions"] = check(
            "strong run pushes the level up", strong > 0, f"offset {strong:+d}")
        results["a weak candidate is not pushed"] = check(
            "weak run does not push up", weak <= 0, f"offset {weak:+d}")
        results["one stumble does not undo a good run"] = check(
            "a single bad answer does not collapse the level",
            wobble > weak, f"offset {wobble:+d} vs {weak:+d} for the weak run")


async def run_substitution(results: dict) -> None:
    section("7. A STRONG CANDIDATE ACTUALLY GETS ASKED HARDER QUESTIONS")
    # The decision is worthless if it does not change the question. This walks
    # the real Interview, forcing every answer to be correct, and checks the
    # questions that come out are harder than the ones that were planned.
    from coach.evaluate import Evaluation, Evaluator
    from coach.interview import Interview

    class AlwaysCorrect(Evaluator):
        async def evaluate(self, q, answer):
            return Evaluation(verdict=Verdict.CORRECT, score=1.0,
                              covered=list(q.expected_points),
                              opening_line="That's right.")

    bank = Bank()
    async with LLM() as llm:
        if not (await llm.health())["has_model"]:
            print("  SKIPPED — ollama is not serving the model")
            return

        bp = await build(bank, ["python"], length=8, base_difficulty=3,
                         llm=None, seed=7)
        planned = list(bp.difficulty_curve)
        iv = Interview(bp, llm, AlwaysCorrect(llm), bank=bank)

        asked: list[int] = []
        decision = None
        await iv.opening()
        while True:
            text = await iv.next_question(decision)
            if text is None:
                break
            q = iv.state.current
            asked.append(q.difficulty)
            print(f"    d{q.difficulty} (planned d{planned[len(asked) - 1]})  {text}")
            ev = await iv.submit("a complete and correct answer")
            if iv.state.pending_follow_up:          # skip the probe; not scored
                iv.state.pending_follow_up = None
                iv._advance(ev)
            nxt = iv.next_level()
            decision = await iv.interviewer.decide(*nxt) if nxt else None

        print(f"\n  planned: {planned}")
        print(f"  asked:   {asked}")
        results["adaptation reaches the question actually asked"] = check(
            "a flawless run is asked harder questions than planned",
            sum(asked) > sum(planned),
            f"total difficulty {sum(asked)} vs {sum(planned)} planned")
        results["no question is asked twice"] = check(
            "no repeats", len({t.question.id for t in iv.state.turns})
            == len(iv.state.turns))


async def main() -> int:
    results: dict[str, bool] = {}
    test_bank(results)
    test_curve(results)
    test_lead_filter(results)
    await run_reads(results)
    await run_substitution(results)

    section("PHASE 8 EXIT CRITERIA")
    for label, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    (ROOT / "bench" / "phase8_results.json").write_text(
        json.dumps({"checks": results}, indent=2))
    return 0 if all(results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
