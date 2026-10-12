"""SQLite schema and small row dataclasses."""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS problems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    topic TEXT NOT NULL,
    difficulty TEXT NOT NULL CHECK (difficulty IN ('Easy','Medium','Hard')),
    phase INTEGER NOT NULL CHECK (phase BETWEEN 1 AND 3),
    google_freq INTEGER NOT NULL CHECK (google_freq BETWEEN 1 AND 5),
    pattern TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_id INTEGER NOT NULL REFERENCES problems(id),
    day TEXT NOT NULL,
    minutes INTEGER NOT NULL,
    solved INTEGER NOT NULL,
    mode TEXT NOT NULL DEFAULT 'deep'
);
CREATE TABLE IF NOT EXISTS recall_drills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    topic TEXT NOT NULL,
    prompt TEXT NOT NULL,
    answer TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recall_state (
    drill_id INTEGER PRIMARY KEY REFERENCES recall_drills(id),
    box INTEGER NOT NULL DEFAULT 0,
    due TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_id INTEGER NOT NULL REFERENCES problems(id),
    day TEXT NOT NULL,
    limit_min INTEGER NOT NULL,
    used_min INTEGER NOT NULL,
    solved INTEGER NOT NULL,
    communication INTEGER NOT NULL CHECK (communication BETWEEN 1 AND 5)
);
CREATE TABLE IF NOT EXISTS star_stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    competency TEXT NOT NULL,
    title TEXT NOT NULL,
    situation TEXT NOT NULL,
    task TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    created TEXT NOT NULL
);
"""


@dataclass
class Problem:
    id: int
    title: str
    topic: str
    difficulty: str
    phase: int
    google_freq: int
    pattern: str


@dataclass
class Drill:
    id: int
    title: str
    topic: str
    prompt: str
    answer: str


def db_path() -> Path:
    env = os.environ.get("GSP_DB")
    if env:
        return Path(env)
    return Path.home() / ".google-swe-prep" / "prep.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('start_date', ?)",
        (date.today().isoformat(),),
    )
    conn.commit()
    return conn
