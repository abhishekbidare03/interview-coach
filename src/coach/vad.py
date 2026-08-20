"""Silero VAD endpointing, tuned for interview answers rather than chat.

Two things make this different from an off-the-shelf VAD wrapper:

1. **A long silence threshold.** plan.md §2.4.3 — chat assistants endpoint at
   ~500 ms, which cuts people off mid-thought. Interview answers are full of
   thinking pauses. We wait ~1300 ms.

2. **A `PAUSE` event fired the instant speech stops**, well before the endpoint
   is confirmed. That is what lets the pipeline start transcribing during the
   silence window instead of after it — Phase 0 showed this hides the entire STT
   cost for any answer under ~25 seconds. If the candidate resumes talking, a
   `RESUME` event tells the pipeline to throw the speculative transcript away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .config import VAD as CFG

log = logging.getLogger(__name__)

FRAME_SAMPLES = 512  # what Silero expects at 16 kHz; not negotiable


class Event(Enum):
    SPEECH_START = auto()
    PAUSE = auto()      # speech stopped; start speculative transcription
    RESUME = auto()     # they were just thinking; discard the speculation
    ENDPOINT = auto()   # silence held long enough; the answer is final
    TOO_LONG = auto()   # safety valve


@dataclass
class Endpointer:
    """Streaming endpoint detector. Feed it 512-sample float32 frames."""

    cfg: object = CFG
    _model: object = field(default=None, repr=False)
    _speaking: bool = False
    _paused: bool = False
    _speech_ms: float = 0.0
    _silence_ms: float = 0.0
    _total_ms: float = 0.0
    _buffer: list[np.ndarray] = field(default_factory=list)

    def load(self) -> None:
        from silero_vad import load_silero_vad

        self._model = load_silero_vad(onnx=False)
        log.info("silero vad loaded (cpu)")

    def reset(self) -> None:
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()
        self._speaking = False
        self._paused = False
        self._speech_ms = self._silence_ms = self._total_ms = 0.0
        self._buffer.clear()

    @property
    def audio(self) -> np.ndarray:
        """Everything captured this utterance, as float32 mono @ 16 kHz."""
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._buffer)

    def feed(self, frame: np.ndarray) -> list[Event]:
        """Process one 512-sample frame. Returns the events it triggered."""
        import torch

        if self._model is None:
            raise RuntimeError("Endpointer.load() was never called")
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}")

        self._buffer.append(frame)
        frame_ms = FRAME_SAMPLES / self.cfg.sample_rate * 1000
        self._total_ms += frame_ms

        with torch.no_grad():
            prob = float(self._model(torch.from_numpy(frame), self.cfg.sample_rate))

        events: list[Event] = []
        is_speech = prob >= self.cfg.speech_threshold

        if is_speech:
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
            if not self._speaking and self._speech_ms >= self.cfg.min_speech_ms:
                self._speaking = True
                events.append(Event.SPEECH_START)
            elif self._paused:
                # A thinking pause, not the end of the answer.
                self._paused = False
                events.append(Event.RESUME)
        elif self._speaking:
            self._silence_ms += frame_ms
            # Fire PAUSE as early as we can be reasonably sure it is not just the
            # gap between two words. One frame of silence would thrash.
            if not self._paused and self._silence_ms >= 250:
                self._paused = True
                events.append(Event.PAUSE)
            if self._silence_ms >= self.cfg.endpoint_silence_ms:
                events.append(Event.ENDPOINT)

        if self._total_ms >= self.cfg.max_answer_ms:
            events.append(Event.TOO_LONG)

        return events

    # -- diagnostics -------------------------------------------------------- #

    @property
    def stats(self) -> dict:
        return {
            "speech_ms": round(self._speech_ms),
            "silence_ms": round(self._silence_ms),
            "total_ms": round(self._total_ms),
            "speaking": self._speaking,
        }


def frames(pcm16: bytes, carry: bytearray) -> list[np.ndarray]:
    """Split a raw PCM16 byte stream into whole 512-sample float32 frames.

    `carry` holds the partial frame between calls — WebSocket messages do not
    arrive on frame boundaries, and feeding Silero a short frame raises.
    """
    carry.extend(pcm16)
    n = len(carry) // (FRAME_SAMPLES * 2)
    if n == 0:
        return []
    take = n * FRAME_SAMPLES * 2
    block = np.frombuffer(bytes(carry[:take]), dtype=np.int16)
    del carry[:take]
    audio = block.astype(np.float32) / 32768.0
    return [audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n)]
