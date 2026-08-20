"""The judgement layer: how hard to push next, and how to lead into it.

R7 asks for an interviewer that "understands the situation and asks
accordingly". Until now that was pure arithmetic — a good answer nudged a
counter up, a bad one nudged it down — and the spoken lead-in was one of four
canned strings. It worked, and it sounded exactly like what it was.

What a real interviewer does between questions is two things at once:

1. **Decide where to go next.** Not from the last answer alone. Someone who has
   been solid for four questions and then fumbles one gets the benefit of the
   doubt; someone who has been shaky throughout and finally gets one right does
   not immediately get a hard question.
2. **Say something that connects.** "Next question" is a robot. "Good — let's
   push on that a bit" tells the candidate they were heard.

Both are given to the model here, but on a short leash:

* It picks from **three** moves, never a raw difficulty number, and a move that
  contradicts the last verdict is overridden. A 3B asked for a number will
  happily answer 7.
* It writes the **lead-in only**, never the question. Rewording a bank question
  invalidates the reference answer and the expected points the grader depends
  on — the same reason `interview._phrase` never called the model either.

**This is off the latency path.** `brain.on_answer` starts the call while the
spoken feedback is still being synthesized, and awaits it only once that audio
is already playing. It costs about 300 ms against roughly 4 s of speech.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field

from .llm import LLM
from .schema import Verdict

log = logging.getLogger(__name__)

MOVES = ("easier", "same", "harder")

# How far the running offset may drift from the planned curve. Two steps up is
# enough to turn a mid-level plan into genuinely senior questions for someone
# who is clearly comfortable; one step down is enough to find a floor without
# the descent becoming the whole interview.
MAX_UP = 2
MIN_DOWN = -1


@dataclass
class Read:
    """What we know about the candidate so far, in a form both a prompt and a
    fallback rule can use."""

    scores: list[float] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)

    def record(self, verdict: Verdict, score: float) -> None:
        self.verdicts.append(verdict)
        self.scores.append(score)

    @property
    def answered(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def recent_mean(self) -> float:
        tail = self.scores[-3:]
        return sum(tail) / len(tail) if tail else 0.0

    @property
    def streak(self) -> int:
        """Consecutive good (+) or poor (-) verdicts, most recent first.

        This is the thing a single verdict cannot tell you, and the reason the
        model is given history rather than just the last result.
        """
        if not self.verdicts:
            return 0
        last = self.verdicts[-1]
        if last.is_good:
            n = 0
            for v in reversed(self.verdicts):
                if not v.is_good:
                    break
                n += 1
            return n
        if last.is_poor:
            n = 0
            for v in reversed(self.verdicts):
                if not v.is_poor:
                    break
                n += 1
            return -n
        return 0

    def briefing(self, level: int, total: int, suggestion: str) -> str:
        """The candidate, in the fewest tokens that still carry the shape.

        The rule's own recommendation is included. Asking a 3B to choose freely
        between three options makes it pick the safe middle every time — five
        consecutive perfect answers produced five 'same' moves. Asking it to
        confirm or overturn a concrete proposal is a much easier task, and one
        it turns out to be good at.
        """
        recent = ", ".join(
            f"{v}({s:.0%})" for v, s in
            zip(self.verdicts[-3:], self.scores[-3:])) or "nothing yet"
        return (f"Answers so far: {self.answered} of {total}, "
                f"averaging {self.mean:.0%}.\n"
                f"Last few, oldest to newest: {recent}.\n"
                f"Current difficulty: {level} out of 5.\n\n"
                f"Suggested move: {suggestion}.")


@dataclass
class Decision:
    move: str = "same"
    lead_in: str = ""
    source: str = "fallback"      # llm | fallback | guard — surfaced in /debug

    @property
    def delta(self) -> int:
        return {"easier": -1, "same": 0, "harder": 1}[self.move]

    def as_dict(self) -> dict:
        return {"move": self.move, "lead_in": self.lead_in, "source": self.source}


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

# Static, and first, so llama.cpp reuses the prefix across every turn of the
# session (the same prompt-cache trick the grader uses).
_SYSTEM = (
    "You are an experienced technical interviewer deciding how hard to pitch "
    "your next question.\n\n"
    "You are given the candidate's run so far and a suggested move. Confirm "
    "the suggestion, or overturn it if the run tells you otherwise.\n\n"
    "  easier - they are struggling; ease off\n"
    "  same   - keep the current level\n"
    "  harder - they are comfortable; push them\n\n"
    "Judge the run, not only the last answer. Two or three good answers in a "
    "row means harder. Two or three poor ones means easier. One bad answer "
    "inside a good run is not a collapse, and one good answer after several "
    "bad ones is not a breakthrough.\n\n"
    "Also write the short line you SAY OUT LOUD to the candidate, who is "
    "sitting in front of you, just before you ask. One sentence, under ten "
    "words, plain spoken words, no markdown, no question of your own. It is a "
    "transition only — do not name the subject of the next question, because "
    "you have not been told what it is.\n\n"
    "Good lines: \"Good, let's push a little further.\" \"Right, stepping it "
    "up.\" \"Okay, next one.\" \"Let's take a simpler one.\"\n"
    "Bad lines: \"Continue this pattern with a harder question.\" (that is an "
    "instruction to yourself, not speech) \"Now consider hash collisions.\" "
    "(names a subject)\n\n"
    'Reply with JSON only: {"move": "...", "lead": "..."}'
)

# Ten words is about the length of a line someone actually says while turning a
# page. At fourteen the model wrote two sentences of encouragement before every
# question — "Great job so far. Let's see how you handle the next problem." —
# which is worse than the canned line it was replacing.
_MAX_LEAD_WORDS = 10

# Phrases that introduce a subject. A lead-in is a transition, so it has no
# business naming what is coming — and it *cannot* know: the move it returns is
# what decides which question gets substituted in, so any subject it names was
# guessed. Left unchecked the model announced "Now consider hash collisions"
# before a question about the import system, and once "a more complex slot
# machine scenario" before one about concurrency.
_INTRODUCERS = re.compile(
    r"\b(consider|considering|about|regarding|involving|covering|concerning|"
    r"focus(?:ing)? on|turn(?:ing)? to|move (?:on )?to|topic|"
    r"scenario|question on)\b", re.I)

# A lead-in that reads as an instruction rather than as speech. The model is
# being told what to do and sometimes hands that instruction straight back —
# "Continue this pattern with a more challenging question." is a stage
# direction, and read aloud to a candidate it is baffling.
_SELF_DIRECTED = re.compile(
    r"^(continue|proceed|ask|pose|present|provide|give|select|choose|"
    r"increase|decrease|maintain|adjust|switch)\b", re.I)


def _clean_lead(text: object, banned: frozenset[str] = frozenset()) -> str:
    """Accept the model's lead-in only if it is really a spoken lead-in.

    A 3B asked for "the line before the question" will sometimes hand back the
    question as well, or a stage direction, or three sentences. Everything
    rejected here falls through to a canned line, which is merely bland — while
    a bad lead-in read aloud is actively confusing, because the candidate hears
    something that does not match the question that follows it.

    `banned` is the session's STT vocabulary — the technical terms this
    interview is about. A transition containing one of them is describing
    content, whatever grammar it used to get there.
    """
    if not isinstance(text, str):
        return ""
    lead = " ".join(text.split()).strip().strip('"')
    lead = re.sub(r"^(interviewer|coach|you)\s*:\s*", "", lead, flags=re.I)

    if not lead or len(lead.split()) > _MAX_LEAD_WORDS:
        return ""
    if "?" in lead:                       # it asked something; the bank asks
        return ""
    # More than one sentence is a speech, not a lead-in.
    if re.search(r"[.!]\s+\S", lead):
        return ""
    if any(c in lead for c in "*_`#\n<>{}"):
        return ""
    if not lead[0].isalpha():
        return ""
    if _INTRODUCERS.search(lead) or _SELF_DIRECTED.match(lead):
        return ""
    if banned and {w.strip(".,!;:'\"").lower() for w in lead.split()} & banned:
        return ""
    if lead[-1] not in ".,!:;-—":
        lead += "."
    return lead


# Used when the model is unavailable, slow, or produced something unusable.
# Keyed by move so the fallback still reflects the decision that was made.
_CANNED: dict[str, tuple[str, ...]] = {
    "harder": ("Good. Let's push a little further.", "Right, stepping it up.",
               "Okay, something harder then.", "Good — let's go deeper."),
    "same":   ("Okay, next one.", "Right, moving on.", "Let's keep going.",
               "Okay, next question."),
    "easier": ("Let's come back to fundamentals.", "Okay, let's take a step back.",
               "Right, something more straightforward.",
               "Let's reset with an easier one."),
}


def _arithmetic_move(read: Read) -> str:
    """The rule the model is being asked to improve on, kept as the floor.

    Two in a row either way is the threshold: reacting to a single answer makes
    the difficulty oscillate, which reads as the interviewer having no memory.
    """
    streak = read.streak
    if streak >= 2 and read.recent_mean >= 0.7:
        return "harder"
    if streak <= -2:
        return "easier"
    return "same"


class Interviewer:
    """Decides the next move. One model call, off the latency path."""

    def __init__(self, llm: LLM, seed: int | None = None,
                 vocabulary: str = "") -> None:
        self.llm = llm
        # The session's technical terms, used to reject a lead-in that has
        # wandered into naming content. Reuses the STT bias list from
        # topics.py rather than maintaining a second vocabulary.
        self._banned = frozenset(
            w.strip().lower() for term in vocabulary.split(",")
            for w in term.split() if len(w.strip()) > 2)
        self.read = Read()
        self._rng = random.Random(seed)
        self.offset = 0               # running adjustment to the planned curve
        self._used_leads: set[str] = set()

    def record(self, verdict: Verdict, score: float) -> None:
        self.read.record(verdict, score)

    def _canned(self, move: str) -> str:
        """A canned lead-in for this move, preferring one not used yet."""
        options = _CANNED[move]
        fresh = [c for c in options if c.lower() not in self._used_leads]
        return self._rng.choice(fresh or list(options))

    def level_for(self, planned: int) -> int:
        return max(1, min(5, planned + self.offset))

    async def decide(self, level: int, total: int) -> Decision:
        """Pick the next move and the line that introduces it."""
        suggestion = _arithmetic_move(self.read)
        decision = Decision(move=suggestion, source="fallback")

        try:
            raw = await self.llm.complete(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user",
                  "content": self.read.briefing(level, total, suggestion)}],
                temperature=0.6,      # the lead-in is the creative half
                json_mode=True,
            )
            data = json.loads(raw)
            move = str(data.get("move", "")).strip().lower()
            if move in MOVES:
                decision = Decision(
                    move=move,
                    lead_in=_clean_lead(data.get("lead"), self._banned),
                    source="llm")
            else:
                log.info("interviewer returned an unusable move %r", move)
        except Exception:
            log.warning("interviewer decision failed; using the rule",
                        exc_info=True)

        decision = self._guard(decision)

        # At this temperature a 3B lands on the same phrasing repeatedly — three
        # of eight questions opened "Great, let's see how you handle the next
        # one." Falling back to the canned rotation is better than hearing one
        # sentence all session.
        if decision.lead_in and decision.lead_in.lower() in self._used_leads:
            log.info("dropping a repeated lead-in %r", decision.lead_in)
            decision.lead_in = ""
        if not decision.lead_in:
            decision.lead_in = self._canned(decision.move)
        self._used_leads.add(decision.lead_in.lower())

        self.offset = max(MIN_DOWN, min(MAX_UP, self.offset + decision.delta))
        log.info("interviewer: %s (%s) offset=%+d lead=%r",
                 decision.move, decision.source, self.offset, decision.lead_in)
        return decision

    def _guard(self, d: Decision) -> Decision:
        """Refuse moves that contradict what just happened.

        The model is good at the ambiguous middle and unreliable at the obvious
        ends — it will occasionally say "harder" straight after a wrong answer,
        which to the candidate feels like being punished for struggling.
        """
        last = self.read.verdicts[-1] if self.read.verdicts else None
        if last is None:
            return d
        if d.move == "harder" and last.is_poor:
            log.info("overriding 'harder' after a poor answer")
            return Decision(move="same", source="guard")
        if d.move == "easier" and last.is_good and self.read.streak >= 2:
            log.info("overriding 'easier' during a good run")
            return Decision(move="same", source="guard")
        return d
