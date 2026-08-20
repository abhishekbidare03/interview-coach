"""The interview state machine — what makes this an interview and not a chatbot.

Responsibilities:

* ask the blueprint's questions in order, adapted to what has been said (R7)
* grade each answer in the right mode, and say the correct answer when the
  candidate is wrong (R5, R6)
* follow up on a specific gap rather than moving on (Phase 4)
* adjust difficulty from performance
* never repeat a question

**A note on context growth.** plan.md §3 flags a long transcript overflowing the
8K window as a real risk and specifies a rolling summariser. It turns out not to
be needed, and the reason is structural: grading is per-question and
self-contained — the prompt is one question, its expected points, its reference
answer, and one answer. The transcript is never in the model's context at all.
Only follow-ups look backwards, and they look back exactly one turn. Context is
bounded by construction rather than by summarisation, so the summariser is not
implemented. The full transcript is still kept here for the Phase 6 report,
where it is read by the user rather than the model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum, auto
from typing import AsyncIterator

from .bank import Bank
from .evaluate import Evaluation, Evaluator, _verdict_for, elaboration_prompt
from .interviewer import Decision, Interviewer
from .llm import LLM
from .schema import Blueprint, Mode, Question, Verdict

log = logging.getLogger(__name__)


class Stage(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    ASKING = "asking"
    LISTENING = "listening"
    EVALUATING = "evaluating"
    SPEAKING = "speaking"
    FINISHED = "finished"


@dataclass
class TurnRecord:
    """One question-and-answer, kept for the post-interview report (Phase 6)."""
    position: int
    question: Question
    answer: str
    evaluation: Evaluation
    was_follow_up: bool = False
    answer_seconds: float = 0.0
    asked_text: str = ""


@dataclass
class InterviewState:
    blueprint: Blueprint
    stage: Stage = Stage.IDLE
    index: int = 0                  # index into blueprint.questions
    turns: list[TurnRecord] = field(default_factory=list)
    asked_ids: set[str] = field(default_factory=set)
    pending_follow_up: str | None = None
    started_at: float = 0.0
    last_asked_text: str = ""       # exactly what was spoken, for the report

    # Difficulty adaptation lives on the Interviewer now, not here — it is a
    # judgement about the candidate rather than a fact about the session.

    @property
    def current(self) -> Question | None:
        if self.index >= len(self.blueprint.questions):
            return None
        return self.blueprint.questions[self.index].question

    @property
    def finished(self) -> bool:
        return self.index >= len(self.blueprint.questions)

    @property
    def progress(self) -> tuple[int, int]:
        return min(self.index + 1, self.blueprint.length), self.blueprint.length


# --------------------------------------------------------------------------- #

ASK_SYSTEM = (
    "You are conducting a spoken practice interview. Everything you say is read "
    "aloud: plain sentences, no markdown, no lists, no code, no symbols. Never "
    "answer your own question."
)


class Interview:
    """Drives one session. Emits spoken text; the caller handles the audio."""

    def __init__(self, blueprint: Blueprint, llm: LLM,
                 evaluator: Evaluator | None = None,
                 bank: Bank | None = None) -> None:
        self.llm = llm
        self.evaluator = evaluator or Evaluator(llm)
        self.interviewer = Interviewer(llm, vocabulary=blueprint.vocabulary)
        # Without a bank the blueprint is followed exactly as planned. The
        # interview still runs; it just cannot swap in a harder or easier
        # question than the one the plan chose.
        self.bank = bank
        self.state = InterviewState(blueprint=blueprint)

    # -- asking ------------------------------------------------------------- #

    async def opening(self) -> str:
        self.state.stage = Stage.ASKING
        self.state.started_at = time.time()
        return self.state.blueprint.opening_line

    async def next_question(self, decision: Decision | None = None) -> str | None:
        """The text to speak for the next question, or None if finished.

        A pending follow-up takes priority over advancing — that is the
        mechanism behind R7, "understand the situation and ask accordingly".

        `decision` is the interviewer's read on the candidate (see
        `interviewer.py`). It does two things: it may substitute a harder or
        easier question than the plan chose, and it supplies the spoken line
        that leads into it.
        """
        if self.state.pending_follow_up:
            text = self.state.pending_follow_up
            self.state.pending_follow_up = None
            self.state.stage = Stage.LISTENING
            self.state.last_asked_text = text
            return text

        if self.state.finished:
            self.state.stage = Stage.FINISHED
            return None

        q = self._select(decision)
        self.state.asked_ids.add(q.id)
        self.state.stage = Stage.LISTENING
        self.state.last_asked_text = self._phrase(q, decision)
        return self.state.last_asked_text

    def _select(self, decision: Decision | None) -> Question:
        """The question to actually ask, which may not be the one planned.

        The blueprint fixes a difficulty for every slot before the interview
        starts. That curve is a plan for a candidate nobody has met yet, so once
        there is evidence, the level it asks for is a starting point rather than
        an instruction: `Interviewer.offset` shifts it, and a question at the
        shifted level is swapped in.

        The substitution is written back into the blueprint slot so that
        `state.current`, the grader, and the end-of-session report all agree on
        what was asked.
        """
        slot = self.state.blueprint.questions[self.state.index]
        q = slot.question
        if self.bank is None:
            return q

        target = self.interviewer.level_for(slot.planned_difficulty)
        # Re-pick when the level has moved, and also when the planned question
        # has already been asked — an earlier substitution may have consumed it.
        if target == q.difficulty and q.id not in self.state.asked_ids:
            return q

        alt = self.bank.pick(q.topic, target, self.state.asked_ids)
        if alt is None or alt.id == q.id:
            return q

        log.info("difficulty %d -> %d: swapped in %r",
                 q.difficulty, alt.difficulty, alt.subtopic or alt.id)
        slot.question = alt
        return alt

    def _phrase(self, q: Question, decision: Decision | None = None) -> str:
        """Wrap a bank question in a spoken lead-in.

        The question text itself is never touched by the model. Rewording is
        where a 3B silently changes what is being asked, which would invalidate
        the reference answer and the expected points the grader depends on — so
        only the lead-in is generated, and only after `_clean_lead` has checked
        it does not contain a question of its own.
        """
        n, total = self.state.progress
        if n == 1:
            lead = "Let's start with this."
        elif q.mode is Mode.BEHAVIOURAL:
            lead = "Now something about your experience."
        elif decision is not None and decision.lead_in:
            lead = decision.lead_in
        elif n == total:
            lead = "Last one."
        else:
            lead = "Next question."
        return f"{lead} {q.text}"

    def next_level(self) -> tuple[int, int] | None:
        """(current difficulty, total questions) for the upcoming slot.

        Read *before* asking the interviewer to decide, because deciding shifts
        the offset — the model has to be told where things stand now, not where
        they are about to be.

        The upcoming *subject* is deliberately not included. The decision is
        what selects the question, so at this point the subject is not yet
        settled, and a lead-in written against the planned one would announce
        the wrong thing.
        """
        if self.state.finished:
            return None
        slot = self.state.blueprint.questions[self.state.index]
        return (self.interviewer.level_for(slot.planned_difficulty),
                self.state.blueprint.length)

    # -- answering ---------------------------------------------------------- #

    async def submit(self, answer: str, seconds: float = 0.0,
                     was_follow_up: bool = False) -> Evaluation:
        """Grade an answer and decide what happens next."""
        q = self.state.current
        if q is None:
            raise RuntimeError("interview already finished")

        self.state.stage = Stage.EVALUATING
        ev = await self.evaluator.evaluate(q, answer)

        if was_follow_up and self.state.turns:
            ev = self._merge_follow_up(answer, ev, seconds)
        else:
            self.state.turns.append(TurnRecord(
                position=self.state.index + 1, question=q, answer=answer,
                evaluation=ev, was_follow_up=was_follow_up,
                answer_seconds=seconds,
                asked_text=self.state.last_asked_text or self._phrase(q),
            ))

        # One follow-up per question, and only when there is a specific gap to
        # probe. Following up on a follow-up turns an interview into an
        # interrogation, and following up after a weak answer just compounds it.
        if ev.should_follow_up and not was_follow_up:
            self.state.pending_follow_up = self._follow_up_text(q, ev)
        else:
            # One data point per question, recorded once the question is fully
            # done — after its follow-up, if it had one. Recording the first
            # answer immediately would let a single question move the
            # difficulty twice, and would score it on half of what was said.
            if ev.verdict is not Verdict.UNCLEAR:
                self.interviewer.record(ev.verdict, ev.score)
            self._advance(ev)

        self.state.stage = Stage.SPEAKING
        return ev

    def _merge_follow_up(self, answer: str, ev: Evaluation,
                         seconds: float) -> Evaluation:
        """Fold a follow-up answer into the turn it was probing.

        A follow-up is part of one question, not a question of its own, but it
        was being graded as one — against the original question's expected
        points, which the probe never asked about. The result was a 0.0 turn in
        almost every report, dragging the average down for the crime of
        answering the extra question that was asked.

        Coverage is the union of both answers, because between them the
        candidate did say those things, and that is what the score means.
        """
        prev = self.state.turns[-1]
        q = prev.question

        covered = list(dict.fromkeys(prev.evaluation.covered + ev.covered))
        missing = [p for p in q.expected_points if p not in covered]
        score = len(covered) / len(q.expected_points) if q.expected_points else 0.0

        merged = replace(
            prev.evaluation, score=score, covered=covered, missing=missing,
            verdict=_verdict_for(q.mode, score), should_follow_up=False,
        )
        prev.evaluation = merged
        prev.answer = f"{prev.answer} {answer}".strip()
        prev.answer_seconds += seconds
        prev.was_follow_up = True

        # The spoken reply still has to react to what was just said, so the
        # opener is re-drawn for the merged verdict rather than reused.
        return replace(merged, opening_line=ev.opening_line)

    # Probes that work for any gap, because they reveal nothing. Rotated so a
    # session with three follow-ups does not use one line three times.
    _PROBES = (
        "Can you take that a bit further?",
        "Is there anything else you would add there?",
        "What else matters there?",
        "Say a little more about that.",
    )

    def _follow_up_text(self, q: Question, ev: Evaluation) -> str:
        """Probe the gap, using a bank seed if the question has one.

        Bank seeds were written offline with the whole question in view, so they
        are better than anything generated mid-conversation — and free.
        """
        if q.follow_up_seeds:
            return q.follow_up_seeds[0]
        # Without a seed, a generic probe — never the missing point spliced
        # into a question. Expected points are terse statements written for the
        # grader, and dropping one into "what about ...?" produced "what about
        # each process has an independent interpreter and GIL?", which is both
        # ungrammatical and a hint at the answer being asked for.
        self._probe_i = getattr(self, "_probe_i", -1) + 1
        return self._PROBES[self._probe_i % len(self._PROBES)]

    def _advance(self, ev: Evaluation) -> None:
        """Move to the next slot.

        Difficulty is no longer nudged here. It used to be a one-line rule on
        the last verdict alone, which made the level oscillate — right answer,
        harder; wrong answer, easier; right answer, harder — and reads to the
        candidate as an interviewer with no memory. `Interviewer.decide` now
        owns it and looks at the run.
        """
        self.state.index += 1

    # -- speaking the feedback ---------------------------------------------- #

    async def feedback(self, ev: Evaluation, answer: str) -> AsyncIterator[str]:
        """Yield the spoken response: templated verdict first, then detail.

        The opener is known the instant grading finishes, so it goes to the
        synthesiser while the LLM is still writing the explanation — the same
        trick as Phase 1, applied one level up.
        """
        q = self.state.turns[-1].question
        yield ev.opening_line

        if ev.verdict is Verdict.UNCLEAR:
            return

        async for token in self.llm.stream(elaboration_prompt(q, answer, ev)):
            yield token

    # -- closing ------------------------------------------------------------ #

    def summary(self) -> dict:
        """Everything the Phase 6 report needs, computed without the model."""
        turns = self.state.turns
        if not turns:
            return {"questions": 0}

        by_topic: dict[str, list[float]] = {}
        for t in turns:
            by_topic.setdefault(t.question.topic, []).append(t.evaluation.score)

        scores = [t.evaluation.score for t in turns]
        return {
            "title": self.state.blueprint.title,
            "questions": len(turns),
            "mean_score": round(sum(scores) / len(scores), 2),
            "verdicts": {v: sum(1 for t in turns if str(t.evaluation.verdict) == v)
                         for v in {str(t.evaluation.verdict) for t in turns}},
            "by_topic": {k: round(sum(v) / len(v), 2) for k, v in by_topic.items()},
            "weakest_topic": min(by_topic, key=lambda k: sum(by_topic[k]) / len(by_topic[k])),
            "duration_s": round(time.time() - self.state.started_at),
            "follow_ups": sum(1 for t in turns if t.was_follow_up),
        }

    def closing_line(self) -> str:
        s = self.summary()
        if not s.get("questions"):
            return "We didn't get through any questions. Let's try again."
        from . import topics as T
        weak = T.get(s["weakest_topic"]).label
        return (f"That's the end of the interview. You answered "
                f"{s['questions']} questions. Your weakest area was {weak}, "
                f"so that's where I'd focus next. Full breakdown is on screen.")
