"""Session persistence — SQLite, local file, no server.

plan.md §4 Phase 6: *"History view — progress over time. For prep, this is where
the actual value compounds."* A single interview tells you little; ten of them
tell you which topic you keep failing, and whether last month's weak spot is
still weak.

Kept deliberately small: two tables, no ORM, no migrations framework. The schema
is created on open and the file lives next to the question bank.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DATA

log = logging.getLogger(__name__)

DB_PATH = DATA / "sessions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    title         TEXT NOT NULL,
    topics        TEXT NOT NULL,        -- json list
    mixed         INTEGER NOT NULL DEFAULT 0,
    planned       INTEGER NOT NULL,
    answered      INTEGER NOT NULL DEFAULT 0,
    mean_score    REAL,
    duration_s    INTEGER,
    summary       TEXT                  -- json blob of the full summary
);

CREATE TABLE IF NOT EXISTS turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    question_id   TEXT NOT NULL,
    topic         TEXT NOT NULL,
    mode          TEXT NOT NULL,
    difficulty    INTEGER NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    verdict       TEXT NOT NULL,
    score         REAL NOT NULL,
    missing       TEXT,                 -- json list
    was_follow_up INTEGER NOT NULL DEFAULT 0,
    answer_s      REAL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_topic   ON turns(topic);
"""


@dataclass
class Store:
    path: Path = DB_PATH

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c

    # -- writing ------------------------------------------------------------ #

    def start_session(self, blueprint) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO sessions (started_at, title, topics, mixed, planned)"
                " VALUES (?,?,?,?,?)",
                (time.time(), blueprint.title, json.dumps(blueprint.topics),
                 int(blueprint.mixed), blueprint.length),
            )
            return int(cur.lastrowid)

    def finish_session(self, session_id: int, interview) -> None:
        """Write the turns and the summary in one transaction, at the end.

        Writing per-turn during the interview would put a disk write on the
        latency path for no benefit — a session that is abandoned halfway is not
        worth reporting on anyway.
        """
        summary = interview.summary()
        with self._conn() as c:
            c.executemany(
                "INSERT INTO turns (session_id, position, question_id, topic,"
                " mode, difficulty, question, answer, verdict, score, missing,"
                " was_follow_up, answer_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(session_id, t.position, t.question.id, t.question.topic,
                  str(t.question.mode), t.question.difficulty, t.question.text,
                  t.answer, str(t.evaluation.verdict), t.evaluation.score,
                  json.dumps(t.evaluation.missing), int(t.was_follow_up),
                  t.answer_seconds)
                 for t in interview.state.turns],
            )
            c.execute(
                "UPDATE sessions SET finished_at=?, answered=?, mean_score=?,"
                " duration_s=?, summary=? WHERE id=?",
                (time.time(), summary.get("questions", 0),
                 summary.get("mean_score"), summary.get("duration_s"),
                 json.dumps(summary), session_id),
            )
        log.info("saved session %d (%d turns)", session_id,
                 len(interview.state.turns))

    # -- reading ------------------------------------------------------------ #

    def recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM sessions WHERE finished_at IS NOT NULL"
                " ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "topics": json.loads(r["topics"]),
                 "summary": json.loads(r["summary"] or "{}")} for r in rows]

    def session(self, session_id: int) -> dict | None:
        with self._conn() as c:
            s = c.execute("SELECT * FROM sessions WHERE id=?",
                          (session_id,)).fetchone()
            if s is None:
                return None
            turns = c.execute(
                "SELECT * FROM turns WHERE session_id=? ORDER BY position, id",
                (session_id,)).fetchall()
        return {
            **dict(s), "topics": json.loads(s["topics"]),
            "summary": json.loads(s["summary"] or "{}"),
            "turns": [{**dict(t), "missing": json.loads(t["missing"] or "[]")}
                      for t in turns],
        }

    def topic_progress(self) -> list[dict]:
        """Per-topic averages, plus whether the last few are better than before.

        This is the number worth looking at: a single session's score is noise,
        but "your last three DBMS answers beat your first three" is signal.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT topic, mode, COUNT(*) n, AVG(score) avg_score"
                " FROM turns GROUP BY topic ORDER BY avg_score ASC").fetchall()
            out = []
            for r in rows:
                scores = [x[0] for x in c.execute(
                    "SELECT score FROM turns t JOIN sessions s"
                    " ON s.id=t.session_id WHERE t.topic=?"
                    " ORDER BY s.started_at, t.position", (r["topic"],))]
                # Compare the later half of your answers with the earlier
                # half. Requires at least four answers in the topic: with two
                # or three, a single lucky question swings the arrow, and a
                # confident "improving" from noise is worse than no arrow.
                half = len(scores) // 2
                trend = None
                if half >= 2:
                    trend = (sum(scores[half:]) / len(scores[half:])
                             - sum(scores[:half]) / half)
                out.append({"topic": r["topic"], "mode": r["mode"],
                            "answered": r["n"],
                            "avg_score": round(r["avg_score"], 2),
                            "trend": round(trend, 2) if trend is not None else None})
        return out

    def stats(self) -> dict:
        with self._conn() as c:
            s = c.execute(
                "SELECT COUNT(*) sessions, SUM(answered) answered,"
                " AVG(mean_score) avg_score, SUM(duration_s) total_s"
                " FROM sessions WHERE finished_at IS NOT NULL").fetchone()
        return {"sessions": s["sessions"] or 0,
                "questions": s["answered"] or 0,
                "avg_score": round(s["avg_score"], 2) if s["avg_score"] else None,
                "total_minutes": round((s["total_s"] or 0) / 60)}
