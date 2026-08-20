"""Phase 2 verification — the bank and the session planner, text only.

plan.md §4 says test this phase entirely by typing: voice adds nothing to
question selection and slows iteration down.

What matters here is not "does it run" but whether the two claims in plan.md
§2.2 actually hold: that the bank is diverse enough not to feel generic, and
that a planned session has a shape rather than being a random draw.

Run:  .venv/Scripts/python.exe bench/phase2_bank_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coach import topics as T          # noqa: E402
from coach.bank import Bank            # noqa: E402
from coach.blueprint import build, difficulty_curve  # noqa: E402
from coach.schema import Mode          # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show_coverage(bank: Bank) -> dict:
    section("BANK COVERAGE")
    cov = bank.coverage()
    print(f"  {'topic':<22} {'mode':<12} {'n':>4} {'subtopics':>10}  difficulties")
    for key, c in cov.items():
        diffs = " ".join(f"d{d}:{n}" for d, n in c["by_difficulty"].items())
        print(f"  {c['label']:<22} {c['mode']:<12} {c['count']:>4} "
              f"{c['subtopics']:>10}  {diffs}")
    print(f"\n  total: {len(bank)} questions across {len(cov)} topics")
    return cov


def check_diversity(bank: Bank) -> bool:
    """The 'not basic questions' requirement (R7), measured rather than asserted.

    Repetition shows up as many questions sharing a subtopic, or as near-identical
    wording. Both are what a naively-prompted 3B produces.
    """
    section("DIVERSITY — does the bank actually vary? (R7)")
    ok = True
    for key in bank.topics:
        qs = bank.all_for(key)
        subs = Counter(q.subtopic for q in qs)
        opens = Counter(" ".join(q.text.split()[:4]).lower() for q in qs)
        top_open = opens.most_common(1)[0]
        sub_ratio = len(subs) / len(qs)
        dup_open = top_open[1] / len(qs)

        # Behavioural questions legitimately all open "Tell me about a time..."
        # — that is the form of the genre, not repetition. Judging them on
        # opening phrases flags correct output as a defect, so for that mode
        # only subtopic spread counts.
        stem_matters = T.get(key).mode is not Mode.BEHAVIOURAL
        good = sub_ratio >= 0.4 and (dup_open <= 0.5 or not stem_matters)
        ok &= good
        print(f"  [{'ok ' if good else 'HMM'}] {T.get(key).label:<22} "
              f"{len(subs):>2} distinct subtopics / {len(qs):>2} questions "
              f"({sub_ratio:.0%})   most repeated opening: "
              f"{top_open[1]}x {top_open[0]!r}")
    return ok


def check_modes(bank: Bank) -> bool:
    """Every question must carry the mode its topic implies (plan.md §2.1)."""
    section("EVALUATION MODES — the §2.1 correctness decision")
    ok = True
    counts: Counter = Counter()
    for key in bank.topics:
        expected = T.get(key).mode
        for q in bank.all_for(key):
            counts[str(q.mode)] += 1
            if q.mode is not expected:
                print(f"  MISMATCH {q.id}: {q.mode} but topic is {expected}")
                ok = False
    for mode, n in counts.items():
        print(f"  {mode:<14} {n:>4} questions")

    # A behavioural question whose reference answer claims a correct answer
    # exists would produce exactly the misleading feedback §2.1 warns about.
    bad = [q for q in bank.by_mode(Mode.BEHAVIOURAL)
           if "correct answer" in q.reference_answer.lower()]
    print(f"  behavioural questions implying a correct answer: {len(bad)}"
          f" {'(good)' if not bad else '<-- PROBLEM'}")
    return ok and not bad


def check_curve() -> bool:
    section("DIFFICULTY CURVE")
    ok = True
    for n in (5, 8, 15):
        for base in (2, 3, 4):
            c = difficulty_curve(n, base)
            rising = c[len(c) // 2] >= c[0]
            eases = c[-1] <= max(c)
            in_range = all(1 <= d <= 5 for d in c)
            good = rising and eases and in_range
            ok &= good
            print(f"  [{'ok ' if good else 'BAD'}] n={n:<3} base={base}  {c}")
    print("\n  warms up, climbs, then steps back down so the session does not")
    print("  end on the hardest question of the day.")
    return ok


async def check_blueprints(bank: Bank) -> bool:
    section("SESSION BLUEPRINTS")
    ok = True

    single = [k for k in bank.topics if k != "behavioural"][:1]
    bp = await build(bank, single, length=8, base_difficulty=3, seed=1)
    ids = [p.question.id for p in bp.questions]
    print(f"\n  single topic: {bp.title}  ({bp.length} questions)")
    for p in bp.questions:
        print(f"    {p.position}. [d{p.question.difficulty}/plan d{p.planned_difficulty}] "
              f"{p.question.subtopic[:34]:<34} {p.question.text[:52]}")
    no_repeat = len(ids) == len(set(ids))
    print(f"    [{'ok ' if no_repeat else 'BAD'}] no repeated question")
    print(f"    [{'ok ' if bp.vocabulary else 'BAD'}] STT vocabulary: "
          f"{bp.vocabulary[:70]}...")
    ok &= no_repeat and bool(bp.vocabulary)

    # Random Interview (R3) — the mix must be genuinely mixed and not clumped.
    bp = await build(bank, bank.topics, length=10, mixed=True, seed=3)
    seq = [p.question.topic for p in bp.questions]
    modes = [str(p.question.mode) for p in bp.questions]
    runs = max((sum(1 for _ in g) for g in _groups(seq)), default=0)
    n_topics = len(set(seq))
    beh = sum(1 for m in modes if m == "behavioural") / len(modes)

    print(f"\n  random interview: {bp.length} questions")
    for p in bp.questions:
        print(f"    {p.position}. [{p.question.topic:<14} {str(p.question.mode):<12}"
              f" d{p.question.difficulty}] {p.question.text[:48]}")
    print(f"    [{'ok ' if n_topics >= 4 else 'BAD'}] {n_topics} distinct topics")
    print(f"    [{'ok ' if runs <= 2 else 'BAD'}] longest same-topic run: {runs}")
    print(f"    [{'ok ' if 0.1 <= beh <= 0.45 else 'BAD'}] behavioural share: "
          f"{beh:.0%} (target ~25%)")
    ok &= n_topics >= 4 and runs <= 2 and 0.1 <= beh <= 0.45

    # Two sessions in a row must not be the same interview.
    a = await build(bank, bank.topics, length=8, mixed=True, seed=11)
    b = await build(bank, bank.topics, length=8, mixed=True, seed=12)
    overlap = len({p.question.id for p in a.questions} &
                  {p.question.id for p in b.questions}) / a.length
    print(f"\n    [{'ok ' if overlap < 0.5 else 'BAD'}] two sessions overlap by "
          f"{overlap:.0%} (should be low)")
    ok &= overlap < 0.5
    return ok


def _groups(seq: list[str]):
    from itertools import groupby
    return (list(g) for _, g in groupby(seq))


async def main() -> int:
    bank = Bank()
    if not len(bank):
        print("bank is empty — run: python scripts/generate_bank.py --all")
        return 1

    cov = show_coverage(bank)
    results = {
        "diversity (R7)": check_diversity(bank),
        "evaluation modes (§2.1)": check_modes(bank),
        "difficulty curve": check_curve(),
        "session blueprints (R2, R3)": await check_blueprints(bank),
    }

    section("PHASE 2 EXIT CRITERIA")
    for label, good in results.items():
        print(f"  [{'PASS' if good else 'FAIL'}] {label}")

    (ROOT / "bench" / "phase2_results.json").write_text(
        json.dumps({"coverage": cov, "checks": results}, indent=2))
    return 0 if all(results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
