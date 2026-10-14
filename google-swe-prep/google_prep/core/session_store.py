"""Auto-save for in-progress sessions (~/.google_prep/sessions/active.json).
A crash, closed window or Ctrl+C leaves the file behind so the next launch can resume.
Finishing a session, or quitting it on purpose with [q], deletes it."""
import json
import os

from google_prep import db

ACTIVE = "active.json"


def _path():
    folder = db.home_dir() / "sessions"
    folder.mkdir(exist_ok=True)
    return folder / ACTIVE


def save(state: dict) -> None:
    path = _path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash mid-write never leaves a half file


def load() -> dict | None:
    path = _path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        clear()  # corrupt file: drop it rather than block every launch
        return None


def clear() -> None:
    try:
        _path().unlink()
    except FileNotFoundError:
        pass
