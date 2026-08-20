"""Piper text-to-speech, chunked for low first-audio latency.

Phase 0 measured the constraint this module exists to work around: Piper emits
nothing until it has synthesized the entire string it was given, at ~3.6x
realtime on this CPU. First-audio latency is therefore proportional to input
length — 4 words costs 358 ms, 20 words costs 1660 ms.

So the rule is: make the FIRST chunk short, then get greedy. Once playback has
started, a growing audio buffer hides the synthesis time of everything after it.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterator

from .config import TTS as CFG

log = logging.getLogger(__name__)

# Where we are willing to break speech. Ordered weakest-boundary-last, because
# the first chunk takes whatever it can get.
_CLAUSE_BREAK = re.compile(r"[,;:—–]")
_SENTENCE_BREAK = re.compile(r"[.!?]")
_BREAKS = re.compile(r"[.!?,;:—–]")


@dataclass
class Chunk:
    """One synthesized audio span, ready to ship to the browser."""
    pcm: bytes            # signed 16-bit little-endian mono
    sample_rate: int
    text: str
    synth_ms: float
    audio_ms: float
    index: int


class TTS:
    """Thread-safe Piper wrapper. Synthesis is CPU-bound and blocking."""

    def __init__(self, cfg=CFG) -> None:
        self.cfg = cfg
        self._voice = None
        self._lock = threading.Lock()
        self.sample_rate = 22050

    # -- lifecycle ---------------------------------------------------------- #

    def load(self) -> None:
        """Load the ONNX voice. Takes ~3.5 s, so call it at server start."""
        from piper import PiperVoice

        t0 = time.perf_counter()
        with self._lock:
            self._voice = PiperVoice.load(str(self.cfg.model_path))
        log.info("piper loaded in %.0f ms (%s)",
                 (time.perf_counter() - t0) * 1000, self.cfg.voice)

    def warmup(self) -> None:
        """First synthesis pays ONNX graph setup. Burn it before a user waits."""
        t0 = time.perf_counter()
        for _ in self.synthesize("Ready."):
            pass
        log.info("piper warmup %.0f ms", (time.perf_counter() - t0) * 1000)

    # -- synthesis ---------------------------------------------------------- #

    def synthesize(self, text: str) -> Iterator[Chunk]:
        """Synthesize one span of text. Yields once; Piper does not stream."""
        if self._voice is None:
            raise RuntimeError("TTS.load() was never called")

        from piper.config import SynthesisConfig

        text = text.strip()
        if not text:
            return

        syn = SynthesisConfig(length_scale=self.cfg.length_scale)
        t0 = time.perf_counter()
        parts: list[bytes] = []
        rate = self.sample_rate

        # Piper's ONNX session is not thread-safe; serialize access.
        with self._lock:
            for chunk in self._voice.synthesize(text, syn_config=syn):
                parts.append(chunk.audio_int16_bytes)
                rate = chunk.sample_rate

        pcm = b"".join(parts)
        self.sample_rate = rate
        yield Chunk(
            pcm=pcm,
            sample_rate=rate,
            text=text,
            synth_ms=(time.perf_counter() - t0) * 1000,
            audio_ms=len(pcm) / 2 / rate * 1000,
            index=0,
        )



# --------------------------------------------------------------------------- #
# Chunking a streaming token feed into speakable spans
# --------------------------------------------------------------------------- #

class SpeechChunker:
    """Turns a stream of LLM tokens into spans sized for gapless, fast speech.

    Two competing pressures, both measured in Phase 0/1:

    * **The first span must be short.** Piper synthesizes the whole string before
      emitting anything, at ~3.6x realtime, so first-audio latency is
      proportional to the text handed over. A 7-word opener cost 1071 ms and
      blew the budget; a 4-word one costs ~400 ms.
    * **Later spans must not be short.** Span N's audio has to play for longer
      than span N+1 takes to synthesize, or playback underruns and you hear a
      gap. At 3.6x realtime the safe growth per span is up to 3.6x; we use 2.5x
      to leave margin for CPU contention with the LLM.

    So: start small, then ramp. The buffer built by early spans pays for the
    longer, better-sounding spans that follow.
    """

    # Opener size. Shorter is faster but a one-word opener is a trap: it starves
    # every following span, since each may only grow ~3x, so you get a chain of
    # two- and three-word fragments. 3-6 words is the sweet spot — under the
    # latency budget, and still a phrase rather than a fragment.
    FIRST_MIN = 3
    FIRST_MAX = 6               # ceiling for the no-punctuation fallback cut
    GROWTH = 3.0                # per-span word growth; 3.6x is the underrun limit
    MIN_BUDGET = 8
    UNLIMITED_AFTER = 3         # by then the buffer is deep; sentences only

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0
        self._last_words = 0

    @property
    def emitted(self) -> int:
        return self._emitted

    def feed(self, token: str) -> Iterator[str]:
        self._buf += token
        while (cut := self._find_cut()) is not None:
            span, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if not span:
                break
            self._emitted += 1
            self._last_words = len(span.split())
            yield span

    def flush(self) -> Iterator[str]:
        span, self._buf = self._buf.strip(), ""
        if span:
            self._emitted += 1
            self._last_words = len(span.split())
            yield span

    # -- internals ---------------------------------------------------------- #

    @property
    def _budget(self) -> int | None:
        """Max words for the span being assembled. None = unlimited.

        Relative to the span before it, not a fixed table. The safety condition
        is that span N's audio must outlast span N+1's synthesis; at ~3.6x
        realtime that permits up to 3.6x growth per span, so 2.5x leaves margin
        for CPU contention with the LLM. Being relative is what makes a very
        short opener safe — a one-word "Exactly," is followed by a four-word
        span, not a twelve-word one that would underrun it.
        """
        if self._emitted >= self.UNLIMITED_AFTER:
            return None
        return max(self.MIN_BUDGET, int(self._last_words * self.GROWTH))

    def _find_cut(self) -> int | None:
        return self._first_cut() if self._emitted == 0 else self._later_cut()

    def _first_cut(self) -> int | None:
        """Take the earliest natural boundary available, then bail out at the cap.

        A punctuation break as short as two words ("Exactly," / "Not quite.") is
        ideal: it is a real prosodic unit, so it sounds deliberate rather than
        clipped, and it gets audio playing in ~250 ms.
        """
        if len(self._buf.split()) < self.FIRST_MIN:
            return None

        best = None
        for m in _BREAKS.finditer(self._buf):
            n = len(self._buf[: m.end()].split())
            if n < self.FIRST_MIN:
                continue
            if n > self.FIRST_MAX:
                break
            best = m.end()
            break                      # earliest acceptable boundary wins
        if best is not None:
            return best

        # No punctuation arrived in range. Cut on a word boundary rather than
        # spend the whole latency budget waiting for a comma.
        return self._word_cut(self.FIRST_MAX)

    def _later_cut(self) -> int | None:
        """Prefer a sentence end; accept a clause break if the budget is spent."""
        budget = self._budget
        words = len(self._buf.split())

        for m in _SENTENCE_BREAK.finditer(self._buf):
            end = m.end()
            # Do not split "3.5" or "e.g." — require whitespace or end of buffer.
            if end < len(self._buf) and not self._buf[end].isspace():
                continue
            if budget is None or len(self._buf[:end].split()) <= budget:
                return end
            break

        if budget is None or words <= budget:
            return None

        # Over budget with no sentence end in sight. Fall back to a clause
        # break, then to a bare word boundary.
        for m in _CLAUSE_BREAK.finditer(self._buf):
            if len(self._buf[: m.end()].split()) >= self.FIRST_MIN:
                return m.end()
        return self._word_cut(budget)

    def _word_cut(self, n_words: int) -> int | None:
        """Index just past the n-th word, or None if it is still streaming in."""
        cut = -1
        for _ in range(n_words):
            nxt = self._buf.find(" ", cut + 1)
            if nxt == -1:
                return None            # last word may still be incomplete
            cut = nxt
        return cut
