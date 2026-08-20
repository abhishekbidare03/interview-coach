"""Session planning — "it should think and start the interview" (R2).

plan.md §2.3 puts this work off the latency path on purpose: it runs after the
user hits Start and before the first question, behind a "preparing your
interview" state, so it can afford ~10 s that mid-conversation reasoning cannot.

The structure of the interview is decided here, deterministically:

* **the difficulty curve** — open well below the target level, climb, finish
  one step above it
* **which topics appear, and in what order** — for Random Interview (R3), a
  weighted mix that never asks three behavioural questions in a row
* **the STT vocabulary** for the whole session (plan.md §2.4.1)

The LLM contributes exactly one thing here: the spoken opening line. Everything
structural is arithmetic, because a 3B model asked to "plan an interview"
produces something plausible and subtly wrong, and there is no reason to let it.

The plan is a starting point, not a script. It is drawn up for a candidate
nobody has met yet, so once there is evidence `interviewer.py` shifts the level
and `interview._select` swaps in a question to match.
"""

from __future__ import annotations

import logging
import random

from . import topics as T
from .bank import Bank
from .llm import LLM
from .schema import Blueprint, PlannedQuestion

log = logging.getLogger(__name__)

DEFAULT_LENGTH = 8


# Where each band ends, as a fraction of the way through. Deliberately
# bottom-heavy: 55% of the interview sits in the lower two bands, because a
# screening round spends most of its time confirming the basics are solid and
# only pushes once it knows they are. An interview that opens at its target
# level tells you nothing about the floor, and rattles people who would have
# been fine.
_BANDS: tuple[float, ...] = (0.25, 0.55, 0.80, 1.01)


def difficulty_curve(n: int, base: int) -> list[int]:
    """Mostly basics, climbing to a short harder tail.

    `base` is the level the candidate picked, and it is treated as the *centre
    of gravity* rather than the starting point: the curve opens two steps below
    it and finishes one step above, spending its first half underneath it.

    The interpolation runs between those two ends rather than adding fixed
    steps to the floor. Adding steps looks equivalent and is not — with the
    floor clamped at 1, Junior and Mid-level both produced 1,1,2,2,3,3,4,4 and
    the difficulty selector did nothing at all.

    Anything above the tail comes from the runtime, not from here — see
    `interviewer.py`. A stretch question is something a candidate earns by
    answering well, not something the plan inflicts on them regardless.
    """
    if n <= 1:
        return [max(1, min(5, base))]

    floor = max(1, base - 2)
    top = min(5, base + 1)
    last = len(_BANDS) - 1

    curve: list[int] = []
    for i in range(n):
        frac = i / (n - 1)
        band = next(b for b, cut in enumerate(_BANDS) if frac < cut)
        d = floor + round((top - floor) * band / last)
        curve.append(max(1, min(5, d)))
    return curve


def _interleave(keys: list[str], weights: dict[str, float], n: int,
                rng: random.Random) -> list[str]:
    """Pick n topics by weight, without letting one topic clump.

    Sampling independently gives runs like behavioural-behavioural-behavioural,
    which feels broken rather than random. Filling largest-remainder first and
    then spacing the result keeps the mix honest and the order varied.
    """
    counts = {k: max(1, round(weights.get(k, 0) * n)) for k in keys}
    while sum(counts.values()) > n:
        counts[max(counts, key=lambda k: counts[k])] -= 1
    while sum(counts.values()) < n:
        counts[min(counts, key=lambda k: counts[k])] += 1

    pool = [k for k, c in counts.items() for _ in range(c) if c > 0]
    rng.shuffle(pool)

    # Push apart any adjacent duplicates.
    for i in range(1, len(pool)):
        if pool[i] == pool[i - 1]:
            for j in range(i + 1, len(pool)):
                if pool[j] != pool[i]:
                    pool[i], pool[j] = pool[j], pool[i]
                    break
    return pool[:n]


async def build(bank: Bank, topic_keys: list[str], *, length: int = DEFAULT_LENGTH,
                base_difficulty: int = 3, mixed: bool = False,
                llm: LLM | None = None, seed: int | None = None) -> Blueprint:
    """Plan a whole interview before the first question is asked."""
    rng = random.Random(seed)

    available = [k for k in topic_keys if bank.all_for(k)]
    if not available:
        raise ValueError(
            f"no questions in the bank for {topic_keys}. "
            f"Run: python scripts/generate_bank.py --topic {topic_keys[0]}")
    if len(available) < len(topic_keys):
        log.warning("skipping topics with an empty bank: %s",
                    sorted(set(topic_keys) - set(available)))

    curve = difficulty_curve(length, base_difficulty)

    if mixed:
        weights = {k: T.RANDOM_MIX.get(k, 0.1) for k in available}
        total = sum(weights.values()) or 1.0
        order = _interleave(available, {k: v / total for k, v in weights.items()},
                            length, rng)
    else:
        order = [available[i % len(available)] for i in range(length)]

    planned: list[PlannedQuestion] = []
    asked: set[str] = set()
    for i, (key, diff) in enumerate(zip(order, curve)):
        q = bank.pick(key, diff, asked, rng)
        if q is None:
            # That topic is exhausted; borrow from another rather than truncate.
            for alt in available:
                if q := bank.pick(alt, diff, asked, rng):
                    break
        if q is None:
            log.warning("bank exhausted after %d questions", len(planned))
            break
        asked.add(q.id)
        planned.append(PlannedQuestion(question=q, position=i + 1,
                                       planned_difficulty=diff))

    used = list(dict.fromkeys(p.question.topic for p in planned))
    labels = [T.get(k).label for k in used]
    title = ("Random Interview" if mixed else
             labels[0] if len(labels) == 1 else " + ".join(labels))

    bp = Blueprint(
        title=title,
        topics=used,
        questions=planned,
        difficulty_curve=[p.planned_difficulty for p in planned],
        vocabulary=T.vocabulary_for(used),
        mixed=mixed,
    )
    bp.opening_line = await _opening(llm, bp) if llm else _fallback_opening(bp)
    return bp


def _fallback_opening(bp: Blueprint) -> str:
    what = "a mix of technical and behavioural questions" if bp.mixed \
        else " and ".join(T.get(k).label for k in bp.topics)
    return (f"Hi, thanks for making the time. Today we'll go through {bp.length} "
            f"questions on {what}. Take your time, and think out loud where it "
            f"helps. Ready when you are.")


async def _opening(llm: LLM, bp: Blueprint) -> str:
    """The one genuinely generative part of planning. Cheap and low-risk."""
    what = "a mix of technical and behavioural questions" if bp.mixed \
        else " and ".join(T.get(k).label for k in bp.topics)
    try:
        text = await llm.complete([
            {"role": "system",
             "content": "You are an interviewer. Everything you write is read "
                        "aloud, so use plain sentences with no markdown."},
            {"role": "user",
             "content": f"Greet the candidate and open a practice interview "
                        f"covering {what}, {bp.length} questions. Two or three "
                        f"sentences, warm but professional. Do not ask the first "
                        f"question yet."},
        ], temperature=0.7)
        text = text.strip()
        # A 3B will sometimes ignore "do not ask the first question yet" and
        # bolt one on; the blueprint owns question order, so refuse the drift.
        if 40 <= len(text) <= 420 and text.count("?") <= 1:
            return text
        log.info("rejected generated opening (%d chars); using fallback", len(text))
    except Exception:
        log.warning("opening line generation failed; using fallback", exc_info=True)
    return _fallback_opening(bp)
