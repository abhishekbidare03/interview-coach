"""FastAPI + WebSocket host for the voice loop.

Phase 1 serves a deliberately plain debug page: transcript plus a live
per-stage latency breakdown. The ElevenLabs-style UI is Phase 5. Getting the
numbers visible first is the point — you cannot tune what you cannot see.

Protocol, on a single WebSocket:

  client -> server   binary  PCM16 mono @ 16 kHz microphone frames
                     text    {"type": "config", "vocabulary": "..."}
                             {"type": "reset"}

  server -> client   text    {"type": "state"|"transcript"|"span"|
                              "assistant"|"metrics"|"error"|"ready", ...}
                     binary  4-byte LE sample rate, then PCM16 mono audio

Prefixing each audio frame with its sample rate keeps the browser honest if the
voice model changes; Piper voices ship at 16 k, 22.05 k, and 24 k.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import topics as T
from .bank import Bank
from .blueprint import build as build_blueprint
from .brain import ChatBrain, InterviewBrain
from .config import LLM as LLM_CFG, STT as STT_CFG, TTS as TTS_CFG, VAD as VAD_CFG
from .llm import LLM
from .pipeline import Session
from .store import Store
from .stt import STT
from .tts import TTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("coach.server")

WEB = Path(__file__).resolve().parent.parent.parent / "web"

# Whisper and Piper are blocking and CPU/GPU bound. Three workers is enough for
# the only real concurrency we have: speculative STT overlapping a TTS span.
POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="coach")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and warm every model before accepting a connection.

    Phase 0 measured a 4216 ms cold LLM call against 225 ms warm. Paying that
    at startup instead of mid-interview is the difference between a tool that
    feels responsive and one that stutters after every pause.
    """
    app.state.stt = STT()
    app.state.tts = TTS()
    app.state.llm = LLM()
    app.state.bank = Bank()
    app.state.store = Store()

    loop = asyncio.get_running_loop()
    log.info("loading models...")
    await asyncio.gather(
        loop.run_in_executor(POOL, app.state.stt.load),
        loop.run_in_executor(POOL, app.state.tts.load),
    )
    await app.state.llm.__aenter__()

    health = await app.state.llm.health()
    if not health["has_model"]:
        log.error("ollama is missing %s — run: ollama pull %s",
                  LLM_CFG.model, LLM_CFG.model)

    log.info("warming up...")
    await asyncio.gather(
        loop.run_in_executor(POOL, app.state.stt.warmup),
        loop.run_in_executor(POOL, app.state.tts.warmup),
        app.state.llm.warmup(),
    )
    log.info("ready - stt=%s/%s llm=%s tts=%s endpoint=%dms bank=%d questions",
             app.state.stt.device, app.state.stt.compute_type,
             LLM_CFG.model, TTS_CFG.voice, VAD_CFG.endpoint_silence_ms,
             len(app.state.bank))

    yield

    await app.state.llm.__aexit__(None, None, None)
    POOL.shutdown(wait=False)


app = FastAPI(title="Interview Coach", lifespan=lifespan)


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/debug")
async def debug() -> FileResponse:
    """Phase 1's latency panel. Kept because tuning needs numbers, not vibes."""
    return FileResponse(WEB / "debug.html")


@app.get("/api/topics")
async def api_topics() -> dict:
    """Everything the setup screen needs to render the picker (R2, R3)."""
    bank = app.state.bank
    return {
        "topics": [{**t, "available": len(bank.all_for(t["key"]))}
                   for t in T.selectable()],
        "coverage": bank.coverage(),
        "lengths": [5, 8, 12],
        "difficulties": [{"value": 2, "label": "Junior"},
                         {"value": 3, "label": "Mid-level"},
                         {"value": 4, "label": "Senior"}],
    }


@app.get("/api/history")
async def api_history(limit: int = 20) -> dict:
    """Progress over time — plan.md §4 Phase 6."""
    return {"stats": app.state.store.stats(),
            "sessions": app.state.store.recent(limit),
            "topics": app.state.store.topic_progress()}


@app.get("/api/history/{session_id}")
async def api_session(session_id: int) -> dict:
    return app.state.store.session(session_id) or {"error": "not found"}


@app.get("/api/health")
async def health() -> dict:
    return {
        "stt": {"model": STT_CFG.model, "device": app.state.stt.device,
                "compute_type": app.state.stt.compute_type},
        "llm": await app.state.llm.health(),
        "tts": {"voice": TTS_CFG.voice, "sample_rate": app.state.tts.sample_rate},
        "vad": {"endpoint_silence_ms": VAD_CFG.endpoint_silence_ms,
                "sample_rate": VAD_CFG.sample_rate},
    }


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    lock = asyncio.Lock()

    async def emit(event: dict) -> None:
        async with lock:
            await sock.send_json(event)

    async def emit_audio(pcm: bytes, rate: int) -> None:
        async with lock:
            await sock.send_bytes(struct.pack("<I", rate) + pcm)

    session = Session(
        stt=app.state.stt, tts=app.state.tts, llm=app.state.llm,
        emit=emit, emit_audio=emit_audio, pool=POOL,
    )

    await emit({"type": "ready",
                "endpoint_silence_ms": VAD_CFG.endpoint_silence_ms,
                "sample_rate": VAD_CFG.sample_rate,
                "bank": len(app.state.bank)})
    log.info("client connected")

    state: dict = {"session_id": None, "saved": False}

    async def save(brain) -> None:
        """Persist once, whether the interview ended naturally or was stopped."""
        if state["saved"] or state["session_id"] is None:
            return
        if not getattr(brain, "interview", None) or not brain.interview.state.turns:
            return
        state["saved"] = True
        try:
            app.state.store.finish_session(state["session_id"], brain.interview)
        except Exception:
            log.exception("failed to save session")

    async def start_interview(cmd: dict) -> None:
        """Plan the session, then open it. Planning is off the latency path
        (plan.md §2.3) — the client shows 'preparing' while this runs."""
        mixed = bool(cmd.get("mixed"))
        keys = list(T.TOPICS) if mixed else (cmd.get("topics") or [])
        keys = [k for k in keys if k in T.TOPICS]
        if not keys:
            await emit({"type": "error", "message": "Pick at least one topic."})
            return

        await emit({"type": "state", "state": "preparing"})
        try:
            bp = await build_blueprint(
                app.state.bank, keys,
                length=int(cmd.get("length", 8)),
                base_difficulty=int(cmd.get("difficulty", 3)),
                mixed=mixed, llm=app.state.llm,
            )
        except ValueError as e:
            await emit({"type": "error", "message": str(e)})
            return

        state["session_id"] = app.state.store.start_session(bp)

        async def on_event(payload: dict) -> None:
            # Persist the moment the interview ends, not when the socket closes,
            # so the history view is up to date while the report is on screen.
            if payload.get("type") == "finished":
                await save(session.brain)
            await emit(payload)

        # The bank goes in so the interview can swap in a harder or easier
        # question than the plan chose once it has a read on the candidate.
        session.brain = InterviewBrain(bp, app.state.llm, on_event=on_event,
                                       bank=app.state.bank)
        await emit({"type": "blueprint", **bp.to_dict()})
        log.info("interview: %s, %d questions, topics=%s",
                 bp.title, bp.length, bp.topics)
        await session.begin()

    try:
        while True:
            msg = await sock.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                await session.feed_audio(data)
            elif (text := msg.get("text")) is not None:
                import json
                cmd = json.loads(text)
                kind = cmd.get("type")
                if kind == "start":
                    await start_interview(cmd)
                elif kind == "chat":
                    # Phase 1's plain conversation, kept as a smoke test.
                    session.brain = ChatBrain(app.state.llm)
                    await session.begin()
                elif kind == "end":
                    if isinstance(session.brain, InterviewBrain):
                        await save(session.brain)
                        await emit({"type": "finished",
                                    "summary": session.brain.interview.summary()})
                    session.brain = None
                    await emit({"type": "state", "state": "idle"})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket error")
    finally:
        # Covers the interview running to its natural end, and the browser tab
        # being closed midway — both should leave a record.
        if isinstance(session.brain, InterviewBrain):
            await save(session.brain)
        log.info("client disconnected")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
