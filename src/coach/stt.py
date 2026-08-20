"""Speech-to-text via faster-whisper, with the two accuracy guards from plan.md §2.4.

Latency note (Phase 0): transcription costs ~1/20th of the audio's duration, so
a 45-second answer is a 2.3-second wait. The pipeline hides this by calling
`transcribe()` the moment speech stops rather than when the endpoint fires —
see `vad.py`. This module just needs to be fast and honest about confidence.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import cuda_paths
from .config import STT as CFG

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    avg_logprob: float
    no_speech_prob: float
    audio_s: float
    latency_ms: float

    @property
    def is_confident(self) -> bool:
        """Whether this is worth grading (plan.md §2.4.2).

        Grading a garbled transcript is the worst failure in the system: the
        candidate gets marked wrong for something they said correctly. When this
        is False the interviewer asks them to repeat instead.
        """
        return (
            bool(self.text.strip())
            and self.avg_logprob >= CFG.min_avg_logprob
            and self.no_speech_prob < 0.6
        )


class STT:
    def __init__(self, cfg=CFG) -> None:
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self.device = cfg.device
        self.compute_type = cfg.compute_type

    def load(self) -> None:
        """Load the model. ~5 s warm from disk cache; call it at server start."""
        cuda_paths.register()  # required on Windows, see cuda_paths docstring
        from faster_whisper import WhisperModel

        attempts = [
            (self.cfg.device, self.cfg.compute_type),
            (self.cfg.device, "int8"),
            ("cpu", "int8"),
        ]
        last: Exception | None = None
        for device, compute in attempts:
            try:
                t0 = time.perf_counter()
                self._model = WhisperModel(
                    self.cfg.model, device=device, compute_type=compute,
                )
                self.device, self.compute_type = device, compute
                log.info("whisper %s loaded on %s/%s in %.1f s",
                         self.cfg.model, device, compute,
                         time.perf_counter() - t0)
                if device == "cpu":
                    log.warning("STT fell back to CPU — expect ~10x slower "
                                "transcription and a sluggish interview")
                return
            except Exception as e:  # noqa: BLE001 - we want the fallback chain
                last = e
                log.warning("whisper %s/%s failed: %s", device, compute,
                            str(e).splitlines()[0][:120])
        raise RuntimeError(f"no working whisper backend: {last}")

    def warmup(self) -> None:
        silence = np.zeros(16000, dtype=np.float32)
        self.transcribe(silence)

    def transcribe(self, audio: np.ndarray, vocabulary: str | None = None) -> Transcript:
        """Transcribe float32 mono PCM at 16 kHz.

        `vocabulary` is fed to Whisper as an `initial_prompt` — plan.md §2.4.1.
        Whisper conditions on it, which measurably improves recall of jargon it
        would otherwise normalise into common English ("mutex" -> "mute ex").
        Passing the current topic's terms is the cheapest accuracy win available.
        """
        if self._model is None:
            raise RuntimeError("STT.load() was never called")

        t0 = time.perf_counter()
        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                beam_size=self.cfg.beam_size,
                language=self.cfg.language,
                vad_filter=False,          # our own Silero VAD already gated this
                initial_prompt=vocabulary,
                condition_on_previous_text=False,  # avoids repetition loops
            )
            segs = list(segments)

        text = " ".join(s.text.strip() for s in segs).strip()
        # Segment-count-weighted mean is overkill; whisper segments are similar
        # length and we only need a threshold comparison.
        avg_lp = (sum(s.avg_logprob for s in segs) / len(segs)) if segs else -10.0
        no_speech = (sum(s.no_speech_prob for s in segs) / len(segs)) if segs else 1.0

        return Transcript(
            text=text,
            avg_logprob=avg_lp,
            no_speech_prob=no_speech,
            audio_s=info.duration,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
