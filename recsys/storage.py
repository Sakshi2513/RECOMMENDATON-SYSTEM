"""Correction log: what the surveyor approved or edited. SQLite, so several users can save at once."""
import csv
import io
import sqlite3
import time
from pathlib import Path

from .cleaning import mask_pii

COLUMNS = ["id", "ts", "surveyor", "negative", "positive", "suggested_actions", "shown_text",
           "final_text", "status", "model_version"]


class CorrectionLog:
    def __init__(self, folder: Path):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(folder / "corrections.db", check_same_thread=False, timeout=15)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("""CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, surveyor TEXT, negative TEXT, positive TEXT,
            suggested_actions TEXT, shown_text TEXT, final_text TEXT, status TEXT, model_version TEXT)""")
        self.con.commit()

    def add(self, *, surveyor, negative, positive, actions, shown_text, final_text, status, model_version):
        """Save one approval or edit. Phone numbers, e-mails and links are masked before storing."""
        self.con.execute(
            "INSERT INTO corrections (ts, surveyor, negative, positive, suggested_actions, shown_text, final_text, status, model_version)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), mask_pii(surveyor)[:60], mask_pii(negative), mask_pii(positive),
             ",".join(actions), mask_pii(shown_text), mask_pii(final_text), status, model_version))
        self.con.commit()

    def count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]

    def to_csv(self) -> bytes:
        rows = self.con.execute("SELECT * FROM corrections ORDER BY id").fetchall()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
        return buf.getvalue().encode("utf-8-sig")
