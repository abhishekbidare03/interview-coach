"""Streaming Ollama client.

Only two things matter here: tokens must arrive incrementally (the whole
low-latency design in plan.md §2.3 depends on it), and the model must stay
pinned in VRAM between turns. Phase 0 measured a cold call at 4216 ms against
225 ms warm — a 19x penalty that would land squarely in the middle of an
interview after any pause.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .config import LLM as CFG

log = logging.getLogger(__name__)


@dataclass
class Turn:
    """Timing for one generation, for the Phase 1 latency panel."""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0
    text: str = ""
    spans: list[str] = field(default_factory=list)

    @property
    def tokens_per_s(self) -> float:
        gen_ms = self.total_ms - self.ttft_ms
        return (self.tokens - 1) / (gen_ms / 1000) if gen_ms > 0 else 0.0


class LLM:
    def __init__(self, cfg=CFG) -> None:
        self.cfg = cfg
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LLM":
        self._client = httpx.AsyncClient(base_url=self.cfg.host, timeout=180.0)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._client:
            await self._client.aclose()

    async def health(self) -> dict:
        """Confirm Ollama is up and holding the model we expect."""
        assert self._client
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        return {"up": True, "models": names,
                "has_model": any(n.startswith(self.cfg.model.split(":")[0])
                                 for n in names)}

    async def warmup(self) -> float:
        """Load weights into VRAM and pin them, before anyone is waiting."""
        t0 = time.perf_counter()
        async for _ in self.stream([{"role": "user", "content": "Say ready."}]):
            pass
        ms = (time.perf_counter() - t0) * 1000
        log.info("llm warmup %.0f ms", ms)
        return ms

    async def stream(self, messages: list[dict],
                     temperature: float | None = None,
                     json_mode: bool = False) -> AsyncIterator[str]:
        """Yield content tokens as they arrive."""
        assert self._client, "use LLM as an async context manager"

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "temperature": (self.cfg.temperature if temperature is None
                                else temperature),
            },
        }
        if json_mode:
            # Used from Phase 3 on: a 3B model will drift into prose without it.
            payload["format"] = "json"

        async with self._client.stream("POST", "/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("unparseable ollama line: %.80s", line)
                    continue
                if err := data.get("error"):
                    raise RuntimeError(f"ollama: {err}")
                if token := data.get("message", {}).get("content"):
                    yield token
                if data.get("done"):
                    return

    async def complete(self, messages: list[dict], **kw) -> str:
        return "".join([tok async for tok in self.stream(messages, **kw)])
