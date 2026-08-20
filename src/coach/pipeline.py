"""The voice loop: audio in -> VAD -> STT -> LLM -> TTS -> audio out.

Phase 1 builds this with no interview logic at all. It is a plain conversation.
The point is to prove the latency budget on real hardware with a real microphone
before anything is built on top of it — plan.md §4 calls this the phase that
de-risks the project.

Two Phase 0 findings are implemented here, and they are the whole reason this
file is more complicated than a straight await chain:

  * **Speculative transcription** (`Event.PAUSE`). Transcription starts when
    speech stops, not when the endpoint fires, so its cost overlaps the 1300 ms
    silence window instead of following it.
  * **A short first speech span** (`SpeechChunker`). Piper's first-audio latency
    is proportional to the text it is handed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from .config import VAD as VAD_CFG
from .llm import LLM
from .stt import STT, Transcript
from .tts import TTS, SpeechChunker
from .vad import Endpointer, Event, frames

log = logging.getLogger(__name__)

@dataclass
class TurnMetrics:
    """Per-turn latency breakdown, shown live in the Phase 1 debug panel."""
    audio_s: float = 0.0
    stt_ms: float = 0.0
    stt_speculative: bool = False
    stt_wait_ms: float = 0.0        # what STT actually cost AFTER the endpoint
    llm_ttft_ms: float = 0.0
    llm_first_span_ms: float = 0.0
    tts_first_ms: float = 0.0
    first_audio_ms: float = 0.0     # endpoint -> first audio byte sent
    total_ms: float = 0.0
    spans: int = 0
    confident: bool = True

    def as_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


Emit = Callable[[dict], Awaitable[None]]        # send a JSON event to the client
EmitAudio = Callable[[bytes, int], Awaitable[None]]  # send PCM + sample rate


@dataclass
class Session:
    """One WebSocket conversation."""

    stt: STT
    tts: TTS
    llm: LLM
    emit: Emit
    emit_audio: EmitAudio
    pool: ThreadPoolExecutor

    brain: object = None               # what to say back; see brain.py

    _ep: Endpointer = field(default_factory=Endpointer)
    _carry: bytearray = field(default_factory=bytearray)
    _spec: asyncio.Task | None = None
    _spec_samples: int = 0
    _busy: bool = False
    _endpoint_at: float = 0.0

    def __post_init__(self) -> None:
        self._ep.load()

    @property
    def vocabulary(self) -> str | None:
        """STT bias terms for the current moment (plan.md §2.4.1)."""
        return getattr(self.brain, "vocabulary", None) or None

    async def begin(self) -> None:
        """Speak the brain's opening utterance."""
        if self.brain is None:
            return
        self._busy = True
        m = TurnMetrics()
        self._endpoint_at = time.perf_counter()
        try:
            await self.emit({"type": "state", "state": "speaking"})
            await self._speak_stream(self.brain.start(), m)
        finally:
            await self.emit({"type": "state", "state": "idle"})
            self._busy = False
            self._ep.reset()

    # ------------------------------------------------------------------ #
    # Audio ingress
    # ------------------------------------------------------------------ #

    async def feed_audio(self, pcm16: bytes) -> None:
        """Called for every binary frame arriving from the browser."""
        if self._busy:
            # We are speaking. Phase 1 does not support barge-in (plan.md §7
            # lists it as optional), so drop mic input rather than transcribe
            # our own voice coming back through the speakers.
            return

        loop = asyncio.get_running_loop()
        for frame in frames(pcm16, self._carry):
            events = await loop.run_in_executor(self.pool, self._ep.feed, frame)
            for ev in events:
                await self._on_event(ev)

    async def _on_event(self, ev: Event) -> None:
        if ev is Event.SPEECH_START:
            await self.emit({"type": "state", "state": "listening"})

        elif ev is Event.PAUSE:
            # The key latency trick: transcribe NOW, during the silence window.
            self._start_speculative()

        elif ev is Event.RESUME:
            # Still thinking. Throw the speculation away; it is stale.
            self._cancel_speculative()

        elif ev in (Event.ENDPOINT, Event.TOO_LONG):
            self._endpoint_at = time.perf_counter()
            audio = self._ep.audio
            self._ep.reset()
            self._carry.clear()
            asyncio.create_task(self._handle_answer(audio))

    # ------------------------------------------------------------------ #
    # Speculative transcription
    # ------------------------------------------------------------------ #

    def _start_speculative(self) -> None:
        if self._spec is not None:
            return
        audio = self._ep.audio
        if audio.size < VAD_CFG.sample_rate // 2:   # under 0.5 s, not worth it
            return
        self._spec_samples = audio.size
        loop = asyncio.get_running_loop()
        self._spec = asyncio.ensure_future(
            loop.run_in_executor(self.pool, self.stt.transcribe,
                                 audio, self.vocabulary)
        )

    def _cancel_speculative(self) -> None:
        if self._spec is not None:
            self._spec.cancel()
            self._spec = None
            self._spec_samples = 0

    async def _resolve_transcript(self, audio: np.ndarray,
                                  m: TurnMetrics) -> Transcript:
        """Use the speculative result if it covers the audio; else transcribe."""
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        # Everything added after the speculation started was silence, so a
        # speculative transcript is still valid — that is what makes this safe.
        if self._spec is not None and self._spec_samples > 0:
            try:
                tr = await self._spec
                m.stt_speculative = True
                m.stt_ms = tr.latency_ms
                m.stt_wait_ms = (time.perf_counter() - t0) * 1000
                return tr
            except asyncio.CancelledError:
                pass
            finally:
                self._spec = None
                self._spec_samples = 0

        tr = await loop.run_in_executor(self.pool, self.stt.transcribe,
                                        audio, self.vocabulary)
        m.stt_speculative = False
        m.stt_ms = tr.latency_ms
        m.stt_wait_ms = (time.perf_counter() - t0) * 1000
        return tr

    # ------------------------------------------------------------------ #
    # The turn
    # ------------------------------------------------------------------ #

    async def _handle_answer(self, audio: np.ndarray) -> None:
        if self._busy:
            return
        self._busy = True
        m = TurnMetrics(audio_s=audio.size / VAD_CFG.sample_rate)
        try:
            await self.emit({"type": "state", "state": "thinking"})
            tr = await self._resolve_transcript(audio, m)

            m.confident = tr.is_confident
            await self.emit({
                "type": "transcript", "text": tr.text,
                "confident": tr.is_confident,
                "avg_logprob": round(tr.avg_logprob, 2),
            })

            if not tr.is_confident:
                # plan.md §2.4.2 — never grade a transcript we do not trust.
                await self._speak_only("Sorry, I didn't catch that. "
                                       "Could you say it again?", m)
                return

            await self._respond(tr.text, m)

        except Exception:
            log.exception("turn failed")
            await self.emit({"type": "error", "message": "Something broke mid-turn."})
        finally:
            m.total_ms = (time.perf_counter() - self._endpoint_at) * 1000
            await self.emit({"type": "metrics", "metrics": m.as_dict()})
            await self.emit({"type": "state", "state": "idle"})
            self._busy = False
            self._ep.reset()

    async def _respond(self, text: str, m: TurnMetrics) -> None:
        """Hand the transcript to the brain and speak whatever comes back."""
        await self._speak_stream(self.brain.on_answer(text), m)

    async def _speak_stream(self, tokens, m: TurnMetrics) -> None:
        """Chunk a token stream into spans and synthesize each as it completes.

        The brain yields a mix of whole utterances (a templated verdict line, a
        bank question) and raw LLM tokens. Feeding everything through one
        chunker keeps the span-sizing rule from Phase 1 in a single place.
        """
        chunker = SpeechChunker()
        t0 = time.perf_counter()
        speaking = False
        first = True
        # Record what was SPOKEN, span by span. Accumulating raw tokens and
        # joining them is wrong either way: joining with spaces splits words
        # ("sh arding", "CD Ns") because LLM tokens carry their own leading
        # space, and joining without one runs whole utterances together, since
        # a brain also yields complete sentences as single items. Spans are
        # already stripped, complete phrases, so they join cleanly.
        spoken: list[str] = []

        async def flush_span(span: str) -> None:
            nonlocal speaking
            if not speaking:
                m.llm_first_span_ms = (time.perf_counter() - t0) * 1000
                await self.emit({"type": "state", "state": "speaking"})
                speaking = True
            spoken.append(span)
            await self._speak(span, m)

        async for token in tokens:
            if first:
                m.llm_ttft_ms = (time.perf_counter() - t0) * 1000
                first = False
            for span in chunker.feed(token):
                await flush_span(span)
            # A brain yields two different kinds of item: raw LLM tokens, and
            # whole utterances (a templated verdict line, a bank question). The
            # chunker concatenates whatever it is fed, so a complete utterance
            # must be flushed or the next one is glued onto its tail — which
            # produced "Okay, noted.Not quite." in the transcript. An item
            # containing a space is a phrase, not a single LLM token; that is
            # the reliable signal. (A length threshold was tried first and
            # failed on "Okay, noted." at exactly the boundary.)
            if token.rstrip().endswith((".", "!", "?")) and " " in token.strip():
                for span in chunker.flush():
                    await flush_span(span)

        for span in chunker.flush():
            await flush_span(span)

        said = " ".join(spoken).strip()
        if said:
            await self.emit({"type": "assistant", "text": said})

    async def _speak(self, span: str, m: TurnMetrics) -> None:
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        chunks = await loop.run_in_executor(
            self.pool, lambda: list(self.tts.synthesize(span)))
        for c in chunks:
            if m.spans == 0:
                m.tts_first_ms = (time.perf_counter() - t0) * 1000
                m.first_audio_ms = (time.perf_counter() - self._endpoint_at) * 1000
            m.spans += 1
            await self.emit({"type": "span", "text": c.text,
                             "synth_ms": round(c.synth_ms),
                             "audio_ms": round(c.audio_ms)})
            await self.emit_audio(c.pcm, c.sample_rate)

    async def _speak_only(self, text: str, m: TurnMetrics) -> None:
        await self.emit({"type": "state", "state": "speaking"})
        await self.emit({"type": "assistant", "text": text})
        await self._speak(text, m)
