"""What to say back — the policy layer above the voice transport.

`pipeline.Session` owns audio: microphone in, endpointing, transcription, speech
out. It should not also own *what the reply is*. A brain is the thing that turns
a transcript into spoken text, so the Phase 1 chat loop and the real interview
can share one audio path without either knowing about the other.

Every brain yields plain text spans. The pipeline chunks and synthesizes them;
nothing here knows Piper exists.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Protocol

from .bank import Bank
from .evaluate import Evaluator
from .interview import Interview
from .interviewer import Decision
from .llm import LLM
from .schema import Blueprint, Verdict

log = logging.getLogger(__name__)


class Brain(Protocol):
    """Turns transcripts into things to say."""

    async def start(self) -> AsyncIterator[str]:
        """Opening utterance, before the candidate has said anything."""
        ...

    async def on_answer(self, text: str) -> AsyncIterator[str]:
        """Everything to say in response to one answer."""
        ...

    @property
    def vocabulary(self) -> str:
        """Terms to bias STT toward right now (plan.md §2.4.1)."""
        ...

    @property
    def finished(self) -> bool:
        ...


# --------------------------------------------------------------------------- #

CHAT_SYSTEM = (
    "You are a friendly interviewer having a spoken conversation. "
    "Reply in at most three short sentences. Never use markdown, bullet points, "
    "code blocks, or emoji — everything you say is read aloud."
)


class ChatBrain:
    """Phase 1's plain conversation. Kept as the pipeline's smoke test."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.history = [{"role": "system", "content": CHAT_SYSTEM}]

    async def start(self) -> AsyncIterator[str]:
        yield "Hi. Say something and I'll reply."

    async def on_answer(self, text: str) -> AsyncIterator[str]:
        self.history.append({"role": "user", "content": text})
        parts: list[str] = []
        async for token in self.llm.stream(self.history):
            parts.append(token)
            yield token
        self.history.append({"role": "assistant", "content": "".join(parts)})

    @property
    def vocabulary(self) -> str:
        return ""

    @property
    def finished(self) -> bool:
        return False


# --------------------------------------------------------------------------- #

class InterviewBrain:
    """The real thing: ask, grade, respond, follow up, move on.

    Ordering here is what the candidate experiences as responsiveness. Each
    answer produces, in order:

        1. the templated verdict line   — ready the moment grading finishes
        2. the streamed explanation     — generated while (1) is being spoken
        3. the next question            — from the blueprint, no generation

    Only step 2 involves waiting on the model for speech, and by then audio is
    already playing. Same principle as Phase 1's short first span, one level up.
    """

    def __init__(self, blueprint: Blueprint, llm: LLM,
                 evaluator: Evaluator | None = None,
                 on_event=None, bank: Bank | None = None) -> None:
        self.interview = Interview(blueprint, llm, evaluator, bank=bank)
        self.blueprint = blueprint
        self._emit = on_event            # async callback for UI events
        self._awaiting_follow_up = False

    # -- lifecycle ---------------------------------------------------------- #

    async def start(self) -> AsyncIterator[str]:
        yield await self.interview.opening()
        async for span in self._ask():
            yield span

    async def _ask(self, decision: Decision | None = None) -> AsyncIterator[str]:
        text = await self.interview.next_question(decision)
        if text is None:
            await self._event({"type": "finished",
                               "summary": self.interview.summary()})
            yield self.interview.closing_line()
            return

        st = self.interview.state
        q = st.current
        n, total = st.progress
        await self._event({
            "type": "question", "text": text, "position": n, "total": total,
            "topic": q.topic if q else None,
            "mode": str(q.mode) if q else None,
            "difficulty": q.difficulty if q else None,
            "follow_up": self._awaiting_follow_up,
            "decision": decision.as_dict() if decision else None,
        })
        yield text

    async def on_answer(self, text: str) -> AsyncIterator[str]:
        was_fu = self._awaiting_follow_up

        # Grading costs ~2 s and nothing can be said until it finishes, because
        # the verdict determines every word that follows. So start it now and
        # speak an acknowledgement over the top of it. A real interviewer does
        # exactly this — you do not get silence while they think, you get "okay,
        # got it". End-to-end this moved first audio from ~2.5 s to ~0.4 s.
        grading = asyncio.create_task(
            self.interview.submit(text, was_follow_up=was_fu))
        yield self._acknowledgement()

        ev = await grading

        await self._event({"type": "evaluation", **ev.as_dict(),
                           "was_follow_up": was_fu})

        # A `pending_follow_up` means the next utterance probes this answer
        # rather than advancing the blueprint.
        self._awaiting_follow_up = self.interview.state.pending_follow_up is not None

        # Deciding how hard to go next is a second model call, and it is started
        # here rather than after the feedback for the same reason grading is
        # started before the acknowledgement: the feedback below is roughly four
        # seconds of speech, so a ~300 ms decision running underneath it is
        # free. Awaited only once that audio is already playing.
        deciding = None
        if not self._awaiting_follow_up and ev.verdict is not Verdict.UNCLEAR:
            deciding = asyncio.create_task(self._decide())

        try:
            async for span in self.interview.feedback(ev, text):
                yield span

            if ev.verdict is Verdict.UNCLEAR:
                # Do not advance or grade further — re-ask the same question.
                return

            decision = await deciding if deciding is not None else None
            deciding = None
        finally:
            # A candidate who hangs up mid-feedback should not leave a task
            # raising into the event loop with nobody listening.
            if deciding is not None:
                deciding.cancel()

        async for span in self._ask(decision):
            yield span

    # Long enough to cover most of the grading wait when spoken (~1.5 s of
    # audio), short enough not to sound like stalling. Varied so a session does
    # not repeat one tic.
    _ACKS = (
        "Okay, thank you.", "Right, got it.", "Mm, okay, thanks.",
        "Understood, thanks.", "Okay, noted.", "Right, thank you.",
    )

    def _acknowledgement(self) -> str:
        self._ack_i = getattr(self, "_ack_i", -1) + 1
        # Rotate rather than sample: random choice repeats back-to-back often
        # enough to be noticeable over a short interview.
        return self._ACKS[self._ack_i % len(self._ACKS)]

    async def _decide(self) -> Decision | None:
        """Ask the interviewer where to go next. Never fatal to the turn."""
        nxt = self.interview.next_level()
        if nxt is None:
            return None
        try:
            return await self.interview.interviewer.decide(*nxt)
        except Exception:
            log.warning("next-move decision failed; asking as planned",
                        exc_info=True)
            return None

    # -- plumbing ----------------------------------------------------------- #

    async def _event(self, payload: dict) -> None:
        if self._emit is not None:
            await self._emit(payload)

    @property
    def vocabulary(self) -> str:
        return self.blueprint.vocabulary

    @property
    def finished(self) -> bool:
        return self.interview.state.finished \
            and not self.interview.state.pending_follow_up
