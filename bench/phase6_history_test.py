"""Phase 6 verification — persistence and the progress view.

plan.md §4: *"History view — progress over time. For prep, this is where the
actual value compounds."* A single session's score is noise; whether your last
three DBMS answers beat your first three is signal. This checks that the data
needed to say that is actually stored and computed correctly.

Uses a temporary database and synthetic sessions, so it needs no GPU and does
not touch the real history.

Run:  .venv/Scripts/python.exe bench/phase6_history_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coach.evaluate import Evaluation                     # noqa: E402
from coach.interview import InterviewState, TurnRecord    # noqa: E402
from coach.schema import (Blueprint, Mode, PlannedQuestion,  # noqa: E402
                          Question, Verdict)
from coach.store import Store                             # noqa: E402


class FakeInterview:
    """Just enough of Interview for the store to read."""

    def __init__(self, blueprint, turns) -> None:
        self.state = InterviewState(blueprint=blueprint)
        self.state.turns = turns
        self.state.started_at = time.time() - 300

    def summary(self):
        from coach.interview import Interview
        return Interview.summary(self)


def make_question(topic: str, mode: Mode, i: int) -> Question:
    return Question(
        topic=topic, difficulty=3, mode=mode,
        text=f"A {topic} question number {i}?",
        reference_answer="Some reference answer for grading purposes.",
        expected_points=[f"point {j} about {topic}" for j in range(4)],
    )


def make_session(store: Store, topic_scores: dict[str, float],
                 title: str) -> int:
    """One synthetic session where each topic is answered at a given score."""
    planned, turns = [], []
    pos = 0
    for topic, score in topic_scores.items():
        pos += 1
        q = make_question(topic, Mode.FACTUAL, pos)
        planned.append(PlannedQuestion(question=q, position=pos,
                                       planned_difficulty=3))
        n_cov = round(score * len(q.expected_points))
        verdict = (Verdict.CORRECT if score >= 0.75 else
                   Verdict.INCORRECT if score < 0.35 else Verdict.PARTIAL)
        turns.append(TurnRecord(
            position=pos, question=q, answer=f"An answer about {topic}.",
            evaluation=Evaluation(
                verdict=verdict, score=score,
                covered=q.expected_points[:n_cov],
                missing=q.expected_points[n_cov:]),
            answer_seconds=25.0))

    bp = Blueprint(title=title, topics=list(topic_scores), questions=planned,
                   difficulty_curve=[3] * len(planned))
    sid = store.start_session(bp)
    store.finish_session(sid, FakeInterview(bp, turns))
    return sid


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp()) / "sessions.db"
    store = Store(path=tmp)
    results: dict[str, bool] = {}

    section("1. SESSIONS ARE STORED AND READ BACK")
    # Four sessions: dbms improves steadily, networking degrades, os is flat.
    # Four is the minimum the trend calculation will act on — it compares two
    # halves, so three answers cannot produce one.
    s1 = make_session(store, {"dbms": 0.25, "networking": 1.00, "os": 0.50},
                      "Session one")
    time.sleep(0.02)
    make_session(store, {"dbms": 0.50, "networking": 0.75, "os": 0.50},
                 "Session two")
    time.sleep(0.02)
    make_session(store, {"dbms": 0.75, "networking": 0.50, "os": 0.50},
                 "Session three")
    time.sleep(0.02)
    make_session(store, {"dbms": 1.00, "networking": 0.25, "os": 0.50},
                 "Random Interview")

    recent = store.recent()
    print(f"  sessions stored: {len(recent)}")
    for r in recent:
        print(f"    #{r['id']} {r['title']:<16} {r['answered']} answered "
              f"mean {r['mean_score']:.2f}  topics={r['topics']}")
    ok = len(recent) == 4 and recent[0]["title"] == "Random Interview"
    print(f"  [{'PASS' if ok else 'FAIL'}] four sessions, newest first")
    results["sessions stored and ordered"] = ok

    section("2. A SESSION ROUND-TRIPS WITH ITS TURNS")
    full = store.session(s1)
    print(f"  session #{s1}: {full['title']}, {len(full['turns'])} turns")
    t0 = full["turns"][0]
    print(f"    Q: {t0['question'][:60]}")
    print(f"    verdict={t0['verdict']} score={t0['score']} "
          f"missing={len(t0['missing'])} points")
    ok = (len(full["turns"]) == 3 and t0["verdict"] in
          {"correct", "partial", "incorrect"} and isinstance(t0["missing"], list))
    print(f"  [{'PASS' if ok else 'FAIL'}] turns stored with verdict and gaps")
    results["turns round-trip with missing points"] = ok

    section("3. TOPIC PROGRESS AND TREND")
    prog = store.topic_progress()
    for p in prog:
        arrow = ("—" if p["trend"] is None else
                 f"{p['trend']:+.2f}")
        print(f"    {p['topic']:<12} {p['answered']} answered  "
              f"avg {p['avg_score']:.2f}  trend {arrow}")

    by = {p["topic"]: p for p in prog}
    # Weakest first, so the list reads as a study plan.
    ordered = [p["topic"] for p in prog]
    sorted_ok = ordered == sorted(ordered, key=lambda t: by[t]["avg_score"])
    improving = by["dbms"]["trend"] is not None and by["dbms"]["trend"] > 0
    slipping = by["networking"]["trend"] is not None and by["networking"]["trend"] < 0
    steady = abs(by["os"]["trend"] or 0) < 0.01

    print(f"  [{'PASS' if sorted_ok else 'FAIL'}] weakest topic listed first")
    print(f"  [{'PASS' if improving else 'FAIL'}] dbms detected as improving "
          f"(0.25 -> 1.00)")
    print(f"  [{'PASS' if slipping else 'FAIL'}] networking detected as slipping "
          f"(1.00 -> 0.25)")
    print(f"  [{'PASS' if steady else 'FAIL'}] os detected as steady "
          f"(0.50 throughout)")
    results["weakest topic first"] = sorted_ok
    results["improvement detected"] = improving
    results["decline detected"] = slipping
    results["flat topic reads as steady"] = steady

    section("4. HEADLINE STATS")
    st = store.stats()
    print(json.dumps(st, indent=2))
    ok = st["sessions"] == 4 and st["questions"] == 12 and st["avg_score"] > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] stats aggregate across sessions")
    results["stats aggregate"] = ok

    section("5. AN ABANDONED SESSION IS NOT REPORTED")
    bp = Blueprint(title="Abandoned", topics=["os"], questions=[],
                   difficulty_curve=[])
    store.start_session(bp)          # started, never finished
    ok = len(store.recent()) == 4
    print(f"  sessions in history after starting a 5th: {len(store.recent())}")
    print(f"  [{'PASS' if ok else 'FAIL'}] unfinished sessions excluded")
    results["unfinished sessions excluded"] = ok

    section("PHASE 6 EXIT CRITERIA")
    for label, good in results.items():
        print(f"  [{'PASS' if good else 'FAIL'}] {label}")

    (ROOT / "bench" / "phase6_results.json").write_text(
        json.dumps({"checks": results, "topics": prog, "stats": st}, indent=2))
    return 0 if all(results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
