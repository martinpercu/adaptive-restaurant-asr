"""Review-queue CLI (plan/phases/phase-6 §6.3). Pure terminal — usable over SSH.

Iterates pending review_queue items: shows raw/final text + judge suggestion + audio path;
accept (with corrected text) / reject / skip. Accepted references join the labeled pool.
"""

from __future__ import annotations

import argparse

from ars.config import DEFAULT_CONFIG, Settings
from ars.db import connect


def pending(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, transcription_id, reason, status FROM review_queue WHERE status='pending'"
    ).fetchall()
    return [{"id": r[0], "transcription_id": r[1], "reason": r[2]} for r in rows]


def resolve(conn, item_id: int, status: str, note: str = "") -> None:
    from ars.db import _now  # noqa: PLC0415

    conn.execute(
        "UPDATE review_queue SET status=?, reviewer_note=?, resolved_at=? WHERE id=?",
        (status, note, _now(), item_id),
    )
    conn.commit()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - interactive
    parser = argparse.ArgumentParser(description="ARS review queue")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    conn = connect(settings.paths.db)
    items = pending(conn)
    print(f"{len(items)} pending review items")
    for it in items:
        tid = it["transcription_id"]
        row = conn.execute(
            "SELECT raw_text, final_text FROM transcriptions WHERE id=?", (tid,)
        ).fetchone()
        raw = row and row[0]
        final = row and row[1]
        print(f"\n[{it['id']}] reason={it['reason']}  raw={raw!r}  final={final!r}")
        choice = input("  (a)ccept / (r)eject / (s)kip> ").strip().lower()
        if choice == "a":
            note = input("  corrected reference> ").strip()
            resolve(conn, it["id"], "accepted", note)
        elif choice == "r":
            resolve(conn, it["id"], "rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
