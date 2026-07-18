"""SQLite operational store (plan/02-architecture.md §5, plan/03-data-spec.md §10).

WAL mode. Phase 1 uses `utterances`, `transcriptions`, `metric_runs`; the remaining
tables are created now so later phases share one schema. All schema here; no ORM.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
    utterance_id TEXT PRIMARY KEY, path TEXT, lang TEXT, store_id TEXT,
    captured_at TEXT, duration_s REAL, speech_ratio REAL, meta_json TEXT
);
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, utterance_id TEXT, model_version TEXT,
    raw_text TEXT, final_text TEXT, avg_logprob REAL, guard_flags_json TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, transcription_id INTEGER, rule_id TEXT,
    before TEXT, after TEXT, confidence REAL
);
CREATE TABLE IF NOT EXISTS judge_verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, transcription_id INTEGER, verdict TEXT,
    corrected_reference TEXT, confusion_json TEXT, order_core_match INTEGER,
    confidence REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT, transcription_id INTEGER, reason TEXT,
    status TEXT, reviewer_note TEXT, resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS metric_runs (
    run_id TEXT PRIMARY KEY, model_version TEXT, dataset_id TEXT,
    metrics_json TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS pos_tickets (
    order_id TEXT PRIMARY KEY, store_id TEXT, items_json TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS cycle_state (
    cycle_id TEXT, step TEXT, data_json TEXT, created_at TEXT,
    PRIMARY KEY (cycle_id, step)
);
"""


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def insert_utterance(
    conn: sqlite3.Connection,
    *,
    utterance_id: str,
    path: str,
    lang: str,
    store_id: str | None,
    captured_at: str | None,
    duration_s: float,
    speech_ratio: float,
    meta: dict | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO utterances VALUES (?,?,?,?,?,?,?,?)",
        (
            utterance_id,
            path,
            lang,
            store_id,
            captured_at,
            duration_s,
            speech_ratio,
            json.dumps(meta or {}, ensure_ascii=False),
        ),
    )
    conn.commit()


def insert_transcription(
    conn: sqlite3.Connection,
    *,
    utterance_id: str,
    model_version: str | None,
    raw_text: str,
    final_text: str,
    avg_logprob: float,
    guard_flags: list[str],
) -> int:
    cur = conn.execute(
        "INSERT INTO transcriptions "
        "(utterance_id, model_version, raw_text, final_text, avg_logprob, guard_flags_json, "
        "created_at) VALUES (?,?,?,?,?,?,?)",
        (
            utterance_id,
            model_version,
            raw_text,
            final_text,
            avg_logprob,
            json.dumps(guard_flags),
            _now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_cycle_step(conn, cycle_id: str, step: str, data: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cycle_state VALUES (?,?,?,?)",
        (cycle_id, step, json.dumps(data, ensure_ascii=False), _now()),
    )
    conn.commit()


def load_cycle_step(conn, cycle_id: str, step: str) -> dict | None:
    row = conn.execute(
        "SELECT data_json FROM cycle_state WHERE cycle_id=? AND step=?", (cycle_id, step)
    ).fetchone()
    return json.loads(row[0]) if row else None


def insert_pos_ticket(conn, order_id: str, store_id: str, items: list[str]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO pos_tickets VALUES (?,?,?,?)",
        (order_id, store_id, json.dumps(items, ensure_ascii=False), _now()),
    )
    conn.commit()


def insert_metric_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    model_version: str | None,
    dataset_id: str,
    metrics: dict,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO metric_runs VALUES (?,?,?,?,?)",
        (run_id, model_version, dataset_id, json.dumps(metrics, ensure_ascii=False), _now()),
    )
    conn.commit()
