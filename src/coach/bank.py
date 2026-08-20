"""Question bank: load, query, and select without repeating yourself.

The bank is generated offline by `scripts/generate_bank.py`. At runtime this
module only reads it. Selection is deliberately deterministic-ish and cheap —
picking the next question is on the latency path, so it must not involve the LLM
(plan.md §2.3). The LLM's runtime job is to *adapt* the chosen question, not to
choose it.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

from . import topics as T
from .config import BANK
from .schema import Mode, Question, load_questions

log = logging.getLogger(__name__)


class Bank:
    def __init__(self, path: Path = BANK) -> None:
        self.path = path
        self._by_topic: dict[str, list[Question]] = {}
        self.load()

    def load(self) -> None:
        self._by_topic.clear()
        for key in T.TOPICS:
            qs = load_questions(self.path / f"{key}.json")
            if qs:
                self._by_topic[key] = qs
        log.info("bank loaded: %d questions across %d topics",
                 sum(len(v) for v in self._by_topic.values()), len(self._by_topic))

    # -- introspection ------------------------------------------------------ #

    @property
    def topics(self) -> list[str]:
        return sorted(self._by_topic)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_topic.values())

    def coverage(self) -> dict[str, dict]:
        """What the bank actually contains — used by the Phase 2 report."""
        out: dict[str, dict] = {}
        for key, qs in self._by_topic.items():
            by_diff: dict[int, int] = defaultdict(int)
            for q in qs:
                by_diff[q.difficulty] += 1
            out[key] = {
                "label": T.get(key).label,
                "mode": str(T.get(key).mode),
                "count": len(qs),
                "subtopics": len({q.subtopic for q in qs}),
                "by_difficulty": dict(sorted(by_diff.items())),
            }
        return out

    # -- selection ---------------------------------------------------------- #

    def pick(self, topic: str, difficulty: int, exclude: set[str],
             rng: random.Random | None = None) -> Question | None:
        """Closest question to `difficulty` in `topic` that has not been asked.

        Falls back outward through neighbouring difficulties rather than failing,
        because a thin bank at one level should not end the interview.
        """
        rng = rng or random
        pool = [q for q in self._by_topic.get(topic, []) if q.id not in exclude]
        if not pool:
            return None

        # Prefer subtopics not yet seen this session; repeating a subtopic at a
        # different difficulty is the most obvious form of "generic questions".
        used_subs = {q.subtopic for q in self._by_topic.get(topic, [])
                     if q.id in exclude}
        fresh = [q for q in pool if q.subtopic not in used_subs]
        candidates = fresh or pool

        best = min(abs(q.difficulty - difficulty) for q in candidates)
        return rng.choice([q for q in candidates
                           if abs(q.difficulty - difficulty) == best])

    def all_for(self, topic: str) -> list[Question]:
        return list(self._by_topic.get(topic, []))

    def by_mode(self, mode: Mode) -> list[Question]:
        return [q for qs in self._by_topic.values() for q in qs if q.mode is mode]
