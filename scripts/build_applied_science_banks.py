r"""Build the Applied Scientist question banks — hand-written, like Python's.

Five factual topics and one open-ended one, covering the shape of an Amazon
Applied Scientist loop: ML breadth, ML depth, statistics and experimentation,
NLP, generative AI, and an ML system design round.

Written by hand for the same reason `build_python_bank.py` exists. A 3B model
asked for "an interview question about the bias-variance tradeoff" produces a
forty-word compound question with restated expected points, and the failure is
worse here than for Python: it also gets the content subtly wrong, and a bank
that teaches you a wrong answer is worse than no bank.

Every entry is validated at build time by `bank_kit` — length, point count,
phrasing variety, and whether the reference answer can actually be read aloud.

Run:  .venv\Scripts\python.exe scripts\build_applied_science_banks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bank_kit as kit                                  # noqa: E402
from coach.schema import Mode                           # noqa: E402

from banks import (deep_learning, dsa, genai, ml, ml_design,  # noqa: E402
                   model_eval, nlp, rag, stats)

TOPICS = [
    ("ml", Mode.FACTUAL, ml.ENTRIES),
    ("deep_learning", Mode.FACTUAL, deep_learning.ENTRIES),
    ("stats", Mode.FACTUAL, stats.ENTRIES),
    ("model_eval", Mode.FACTUAL, model_eval.ENTRIES),
    ("nlp", Mode.FACTUAL, nlp.ENTRIES),
    ("genai", Mode.FACTUAL, genai.ENTRIES),
    ("rag", Mode.FACTUAL, rag.ENTRIES),
    ("dsa", Mode.FACTUAL, dsa.ENTRIES),
    ("ml_design", Mode.OPEN_ENDED, ml_design.ENTRIES),
]


def main() -> int:
    # Validate everything before writing anything. The cross-topic check can
    # only run once all the banks are built, and a half-written set of files is
    # worse than none.
    built = {t: kit.build(t, mode, entries) for t, mode, entries in TOPICS}
    kit.check_no_cross_topic_duplicates(built)

    for topic, questions in built.items():
        path = kit.write(topic, questions)
        kit.report(topic, questions)
        print(f"    -> {path}")

    kit.summarise(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
