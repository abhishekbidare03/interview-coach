"""Single place for every model choice, threshold, and tuning constant.

Phase 0 exists to validate these numbers. If it fails its VRAM or latency exit
criteria, the fallback is to edit this file — not to hunt through modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
VOICES = DATA / "voices"
BANK = DATA / "bank"


# --------------------------------------------------------------------------- #
# Speech-to-text
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class STTConfig:
    model: str = "small.en"
    device: str = "cuda"
    # int8 weights (half the VRAM) with float16 compute (fast encoder on Turing).
    compute_type: str = "int8_float16"
    beam_size: int = 1          # greedy; beam search is not worth the latency here
    language: str = "en"

    # plan.md §2.4.2 — below this, ask the candidate to repeat rather than grade a
    # garbled transcript. Calibrated in Phase 1 against real speech.
    min_avg_logprob: float = -1.0

    # Phase 0 fallback if VRAM is tight: model="base.en"


# --------------------------------------------------------------------------- #
# Language model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LLMConfig:
    host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model: str = "qwen2.5:3b"
    num_ctx: int = 8192
    temperature: float = 0.3    # grading wants consistency, not creativity

    # -1 pins weights in VRAM for the life of the server. Sent per-request so we
    # never have to touch the user's global OLLAMA_KEEP_ALIVE.
    keep_alive: int = -1

    # Phase 0 fallback if VRAM is tight: model="qwen2.5:1.5b"


# --------------------------------------------------------------------------- #
# Text-to-speech
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TTSConfig:
    voice: str = "en_US-lessac-high"
    length_scale: float = 1.0   # >1 slower. Interviewers speak unhurriedly.

    @property
    def model_path(self) -> Path:
        return VOICES / f"{self.voice}.onnx"


# --------------------------------------------------------------------------- #
# Voice activity detection / endpointing
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 32          # Silero wants 512 samples at 16 kHz
    speech_threshold: float = 0.5

    # plan.md §2.4.3 — THE most important constant in the project. Chat defaults
    # (~500 ms) cut people off mid-thought; interview answers contain long
    # thinking pauses. Tuned against real speech in Phase 1.
    endpoint_silence_ms: int = 1300

    min_speech_ms: int = 300    # ignore coughs, "um", mic pops
    max_answer_ms: int = 180_000


STT = STTConfig()
LLM = LLMConfig()
TTS = TTSConfig()
VAD = VADConfig()
