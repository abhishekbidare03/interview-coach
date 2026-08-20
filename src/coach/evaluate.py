"""Answer evaluation — three modes, one dispatch (plan.md §2.1).

Two design decisions carry this module.

**1. The model checks coverage; the code decides the verdict.**
Asking a 3B "is this answer correct?" produces an unstable, ungroundable verdict.
Asking it "did they say this specific thing? yes or no" — once per expected point,
with the reference answer in context — is close to entailment checking, which is
something a small model can actually do. The verdict is then arithmetic on the
coverage vector, which makes it consistent, tunable, and explainable.

**2. The first spoken sentence is templated, not generated.**
The verdict line ("That's not quite right.") is known the instant grading
finishes, so it goes straight to Piper while the LLM is still writing the
explanation. That reuses the Phase 1 finding — start audio early, elaborate
behind it — and removes the LLM entirely from the critical path for the first
~350 ms of speech.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field

from .llm import LLM
from .schema import Mode, Question, Verdict

log = logging.getLogger(__name__)


@dataclass
class Evaluation:
    verdict: Verdict
    score: float                    # 0..1, the coverage ratio
    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    should_follow_up: bool = False
    opening_line: str = ""          # spoken immediately, templated
    grade_ms: float = 0.0
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"verdict": str(self.verdict), "score": round(self.score, 2),
                "covered": self.covered, "missing": self.missing,
                "should_follow_up": self.should_follow_up,
                "grade_ms": round(self.grade_ms)}


# --------------------------------------------------------------------------- #
# Grading prompts — one per mode
# --------------------------------------------------------------------------- #

_GRADER_SYSTEM = (
    "You grade spoken interview answers against a checklist. "
    "You reply with a single JSON object and nothing else."
)

# The verb matters more than it looks. Asking "did the candidate SAY this?"
# works for factual claims and is nonsense for behavioural criteria — a
# structural requirement like "says what they personally did rather than what
# the team did" is something an answer SATISFIES, not something it recites.
# Phase 3's first fixture run graded 0/4 on every behavioural answer, strong
# ones included, because the checklist was being matched literally.
_TASK = {
    Mode.FACTUAL:
        "For each point, decide whether the candidate STATED that idea.",
    Mode.OPEN_ENDED:
        "For each consideration, decide whether the candidate RAISED it.",
    Mode.BEHAVIOURAL:
        "Each item is a requirement the candidate's story must SATISFY. For "
        "each one, decide whether their story satisfies it. Judge the story "
        "they told, not the words they used.",
}

# Checked before anything else. Without it the grader pattern-matches the
# checklist against whatever text it is given: the end-to-end run scored a
# vacuous, off-topic answer 4/4 STRONG on a system design question, because
# "raised it at all, even briefly" is satisfied by almost any plausible words.
_ON_TOPIC = (
    "Before checking anything: does the answer actually respond to the question "
    "that was asked? If it is about a different subject, or is vague filler with "
    "no real content, mark every item false and stop there."
)

_RUBRIC = {
    Mode.FACTUAL: (
        "Mark a point covered only if the candidate stated that idea. Wording "
        "may differ; meaning may not. Do not give credit for something they "
        "merely implied or for something only the reference answer says."
    ),
    Mode.OPEN_ENDED: (
        "First check whether the answer addresses the question that was asked "
        "at all. If it is about a different subject, or is generic filler with "
        "no design content, mark EVERY item false — no exceptions. "
        "Only if it genuinely engages with the question: mark a consideration "
        "covered when the candidate raised it, even briefly, since a design "
        "discussion is allowed to be high level. Do not require them to reach "
        "the same conclusion."
    ),
    Mode.BEHAVIOURAL: (
        "These are structural requirements, not facts. Judge whether the "
        "candidate's story actually contains each element. Be strict about "
        "specificity: 'we improved performance' does not count as a concrete "
        "outcome, and 'the team decided' does not count as personal ownership. "
        "There is no correct story — never judge the choices they describe."
    ),
}


def _grade_system(mode: Mode) -> str:
    """The whole static instruction set, identical for every call in a mode.

    Prompt layout is a latency decision, not a style one. llama.cpp caches the
    longest common *prefix* of consecutive prompts, so anything constant must
    come first to be reused. The original version put the question and answer
    first and the rubric last, which made every grading call a fresh prefix and
    cost ~2.1 s. Three mode-specific system prompts means the boilerplate is
    processed once and reused for the rest of the session.
    """
    return (
        f"{_GRADER_SYSTEM}\n\n{_ON_TOPIC}\n{_TASK[mode]}\n{_RUBRIC[mode]}\n\n"
        # One boolean per item, not a packed "YNYY" string. Packing was tried to
        # save output tokens and cut ~200 ms, but accuracy collapsed — every
        # answer came back 2/4 regardless of quality, because a single short
        # string lets the model emit a habitual pattern instead of deciding each
        # item. One array element per checklist item forces one decision per
        # item. plan.md §5 ranks evaluation quality above latency; this is that
        # trade, made explicitly.
        'Reply with JSON and nothing else, in the form '
        '{"covered": [true, false, ...]} with exactly one value per checklist '
        'item, in order.'
    )


def _grade_prompt(q: Question, answer: str) -> list[dict]:
    points = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(q.expected_points))
    ref = (f"\nModel answer (for your judgement only; the candidate never sees "
           f"it):\n{q.reference_answer}\n" if q.mode is not Mode.BEHAVIOURAL
           else "")
    user = (
        f"Question asked:\n{q.text}\n{ref}\n"
        f"Checklist ({len(q.expected_points)} items):\n{points}\n\n"
        f"Candidate's spoken answer:\n\"\"\"{answer}\"\"\""
    )
    return [{"role": "system", "content": _grade_system(q.mode)},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Verdict thresholds — coverage ratio to verdict, per mode
# --------------------------------------------------------------------------- #

_THRESHOLDS: dict[Mode, tuple[float, float]] = {
    # (good_at_or_above, poor_below)
    Mode.FACTUAL: (0.75, 0.35),
    Mode.OPEN_ENDED: (0.60, 0.30),   # design answers are never exhaustive
    Mode.BEHAVIOURAL: (0.75, 0.40),  # only 4 structural points; each matters
}


def _verdict_for(mode: Mode, score: float) -> Verdict:
    good, poor = _THRESHOLDS[mode]
    if mode is Mode.FACTUAL:
        return (Verdict.CORRECT if score >= good else
                Verdict.INCORRECT if score < poor else Verdict.PARTIAL)
    return (Verdict.STRONG if score >= good else
            Verdict.WEAK if score < poor else Verdict.ADEQUATE)


# --------------------------------------------------------------------------- #
# Spoken openers — varied so a session does not sound like a form letter
# --------------------------------------------------------------------------- #

_OPENERS: dict[Verdict, tuple[str, ...]] = {
    Verdict.CORRECT: ("That's right.", "Yes, exactly.", "Correct.",
                      "That's the answer I was looking for."),
    Verdict.PARTIAL: ("You're on the right track,", "Partly right,",
                      "That's a good start,", "Close, but not complete,"),
    Verdict.INCORRECT: ("That's not quite right.", "Not quite.",
                        "That's not the answer I'd expect.", "I'd push back on that."),
    Verdict.STRONG: ("That's a strong answer.", "Good, that works well.",
                     "That's well structured.", "Nicely put."),
    Verdict.ADEQUATE: ("That's a reasonable answer,", "That works,",
                       "Good, though there's more there,", "That's solid as far as it goes,"),
    Verdict.WEAK: ("Let's tighten that up.", "There's a gap there.",
                   "That needs more.", "I'd want more from that answer."),
    Verdict.UNCLEAR: ("Sorry, I didn't catch that.",),
}


def _opening_line(verdict: Verdict, rng: random.Random) -> str:
    return rng.choice(_OPENERS[verdict])


# --------------------------------------------------------------------------- #
# Elaboration — streamed behind the opener
# --------------------------------------------------------------------------- #

_ELABORATE_SYSTEM = (
    "You are an interviewer speaking directly to the candidate, face to face. "
    "Address them as 'you'. Never refer to them as 'they' or 'the candidate' — "
    "they are in the room with you. Everything you say is read aloud: plain "
    "sentences only, no markdown, no lists, no code, no symbols, and never a "
    "label like 'Correct:' or 'Answer:'. Be concise and specific."
)


def elaboration_prompt(q: Question, answer: str, ev: Evaluation) -> list[dict]:
    """Prompt for the explanation spoken after the templated opener.

    The opener has already delivered the verdict, so this must not repeat it —
    hence the explicit instruction not to restate whether they were right.
    """
    missing = "\n".join(f"- {m}" for m in ev.missing) or "- nothing significant"

    # Every word below is second person. Instructing the model to address the
    # candidate as "you" while the prompt itself calls them "they" loses: the
    # first end-to-end run said "They missed mentioning that sharding is..."
    # aloud, to the candidate's face. The surrounding text has to model the
    # voice we want, not merely describe it.
    if q.mode is Mode.FACTUAL:
        if ev.verdict is Verdict.CORRECT:
            task = ("In one sentence, add a detail or nuance that sharpens what "
                    "you said. Do not praise you again.")
        else:
            # R6: when wrong, they must actually hear the correct answer.
            task = (f"Say what you missed, then give the correct answer plainly. "
                    f"Base it on this:\n{q.reference_answer}\n"
                    f"Two or three sentences.")
    elif q.mode is Mode.BEHAVIOURAL:
        task = ("Name the specific structural element your story was missing and "
                "what you should have said instead. Never suggest the choices "
                "you describe were wrong - only how you told the story. "
                "Two sentences.")
    else:
        task = ("Name the considerations you did not raise and why each one "
                "matters at scale. Two or three sentences.")

    return [
        {"role": "system", "content": _ELABORATE_SYSTEM},
        {"role": "user", "content":
            f"You asked me: {q.text}\n\n"
            f"I answered: \"\"\"{answer}\"\"\"\n\n"
            f"I did not cover:\n{missing}\n\n"
            f"Now reply to me directly, saying 'you' to mean me. {task}\n\n"
            f"Do not restate whether I was right or wrong - that has already "
            f"been said. Start straight into the substance."},
    ]


# --------------------------------------------------------------------------- #

class Evaluator:
    def __init__(self, llm: LLM, seed: int | None = None) -> None:
        self.llm = llm
        self._rng = random.Random(seed)

    async def evaluate(self, q: Question, answer: str) -> Evaluation:
        """Grade one answer. Fast and non-streaming — the output is tiny."""
        t0 = time.perf_counter()

        if not answer.strip():
            return Evaluation(verdict=Verdict.UNCLEAR, score=0.0,
                              missing=list(q.expected_points),
                              opening_line=_opening_line(Verdict.UNCLEAR, self._rng))

        covered_flags = await self._grade(q, answer)
        covered = [p for p, c in zip(q.expected_points, covered_flags) if c]
        missing = [p for p, c in zip(q.expected_points, covered_flags) if not c]

        score = len(covered) / len(q.expected_points) if q.expected_points else 0.0
        verdict = _verdict_for(q.mode, score)

        return Evaluation(
            verdict=verdict,
            score=score,
            covered=covered,
            missing=missing,
            # Probe a gap rather than moving on, but not when the answer was a
            # write-off — re-asking a struggling candidate is demoralising, and
            # plan.md §4 Phase 4 wants weak answers to move on.
            should_follow_up=bool(missing) and not verdict.is_poor,
            opening_line=_opening_line(verdict, self._rng),
            grade_ms=(time.perf_counter() - t0) * 1000,
            raw={"covered_flags": covered_flags},
        )

    async def _grade(self, q: Question, answer: str) -> list[bool]:
        n = len(q.expected_points)
        try:
            raw = await self.llm.complete(_grade_prompt(q, answer),
                                          temperature=0.1, json_mode=True)
            flags = _parse_flags(json.loads(raw), n)
        except Exception as e:
            log.warning("grading failed (%s); falling back to keyword overlap",
                        type(e).__name__)
            return [_overlaps(p, answer) for p in q.expected_points]

        return flags


def _truthy(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "y", "1", "covered")
    return bool(v)


def _parse_flags(data: dict, n: int) -> list[bool]:
    """Read the coverage vector, accepting whichever shape the model produced.

    The prompt asks for a packed "YNYY" string, but a 3B drifts back to a JSON
    boolean array often enough that rejecting it would waste a grade for no
    reason. Both are cheap to accept.
    """
    raw = data.get("c", data.get("covered", data.get("coverage")))

    if isinstance(raw, str):
        flags = [ch.upper() == "Y" for ch in raw if ch.upper() in ("Y", "N")]
    elif isinstance(raw, list):
        flags = [_truthy(v) for v in raw]
    else:
        raise ValueError(f"unusable coverage field: {raw!r}")

    if not flags:
        raise ValueError("empty coverage vector")
    # Wrong length happens; pad as not-covered rather than fail the whole turn.
    return (flags + [False] * n)[:n]


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "that", "this", "it", "its", "with", "as", "by", "be",
    "they", "their", "them", "you", "your", "not", "but", "can", "will",
}


def _overlaps(point: str, answer: str, threshold: float = 0.5) -> bool:
    """Last-resort coverage check when the model fails to return usable JSON.

    Crude on purpose: it exists so one bad response degrades a single grade
    instead of ending the interview.
    """
    words = {w for w in re.findall(r"[a-z]+", point.lower()) if w not in _STOP}
    if not words:
        return False
    said = set(re.findall(r"[a-z]+", answer.lower()))
    return len(words & said) / len(words) >= threshold
