"""Raw-audio retention (plan/phases/phase-7 §7.3).

Raw audio is purged after `retention_days`; transcripts/metrics are kept (soft-delete —
never orphan DB rows). Dry-run is the DEFAULT; a real purge requires dry_run=False.
"""

from __future__ import annotations

import time
from pathlib import Path


def purge_old_audio(
    raw_root: str | Path, retention_days: int = 90, dry_run: bool = True, now: float | None = None
) -> dict:
    """Delete audio files older than retention under raw_root. Returns a report. Newer files
    and any non-audio files are untouched. dry_run=True (default) deletes nothing."""
    raw_root = Path(raw_root)
    now = now if now is not None else time.time()
    cutoff = now - retention_days * 86400
    scanned, purged, kept = 0, [], 0
    if raw_root.exists():
        for f in sorted(raw_root.rglob("*.wav")):
            scanned += 1
            if f.stat().st_mtime < cutoff:
                purged.append(f.relative_to(raw_root).as_posix())
                if not dry_run:
                    f.unlink()
            else:
                kept += 1
    return {
        "scanned": scanned,
        "purged": purged,
        "kept": kept,
        "dry_run": dry_run,
        "retention_days": retention_days,
    }


def mark_purged(conn, utterance_ids: list[str]) -> None:
    """Soft-delete: flag utterances whose audio was purged, keeping the row + metrics."""
    import sqlite3  # noqa: PLC0415

    try:
        conn.execute("ALTER TABLE utterances ADD COLUMN audio_purged INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    for uid in utterance_ids:
        conn.execute("UPDATE utterances SET audio_purged=1 WHERE utterance_id=?", (uid,))
    conn.commit()
