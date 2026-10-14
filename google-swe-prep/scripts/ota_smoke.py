"""Real old-to-new upgrade check for CI (Windows). Uses a dedicated test profile, never a real one.

    python scripts/ota_smoke.py seed   --home <profile>
    python scripts/ota_smoke.py stage  --home <profile> --installer <Setup.exe> --manifest <google-prep-update.json>
    <installed or packaged google-prep-update.exe> --home <profile> --parent-pid 0
    python scripts/ota_smoke.py verify --home <profile> --install-dir <app folder> --build <number>

seed writes real study data through the app's own code; stage puts the new installer exactly where
the background updater would; verify checks every seeded record, the installed build identity, the
installer result and the pre-update database backup."""
import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPECTED = "ota-expected.json"


def seed(home: Path) -> dict:
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    os.environ["GOOGLE_PREP_HOME"] = str(home)
    from google_prep.core import scheduler, session_store, training
    from google_prep.db import get_session
    from google_prep.models import StarStory

    now = datetime.now().replace(microsecond=0)
    expected = {
        "attempt_problem": "G-1",
        "star_title": "OTA smoke: cut p99 latency",
        "intention": "OTA smoke: ask about input size before coding.",
        "design_notes": "OTA smoke notes: cite evidence, respect permissions, abstain when unsure.",
        "design_learned": ["ml-pipeline"],
        "session_problem": "G-217",
        "scratch_text": "# OTA smoke scratchpad\nseen = set()\n",
    }
    with get_session() as s:
        scheduler.set_setting(s, "start_date", now.date().isoformat())
        scheduler.log_attempt(s, expected["attempt_problem"], 25, True, True, 4, now, min_days=3, help="none")
        s.add(StarStory(dimension="Ownership", title=expected["star_title"], situation="s", task="t",
                        action="I profiled and cached.", result="p99 900 ms -> 210 ms"))
        s.commit()
        habit = training.habit_for_day(s, now.date())["id"]
        training.save_checkin(s, now, now.date().isoformat(), habit, intention=expected["intention"])
        scheduler.set_setting(s, "design:ml-rag:notes", expected["design_notes"])
        scheduler.set_setting(s, "design:ml:learned", json.dumps(expected["design_learned"]))
    session_store.save({"kind": "mock", "problem_id": expected["session_problem"], "minutes": 60, "practice": True,
                        "elapsed": 600.0, "ticks": [True, False, False, False], "shown": -1, "submitted": False,
                        "started": now.isoformat(), "submitted_at": None, "hints": 0})
    (home / "sessions" / "G-217_ota_smoke.txt").write_text(expected["scratch_text"], encoding="utf-8")
    (home / EXPECTED).write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return expected


def stage(home: Path, installer: Path, manifest_file: Path) -> Path:
    """Exactly what UpdateManager leaves behind after a verified download."""
    from google_prep.updates import cache
    from google_prep.updates.manifest import APP_ID, INSTALLER, REPOSITORY, Candidate

    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    tag = manifest["tag"]
    candidate = Candidate(manifest["version"], manifest["build_number"], manifest["commit"], tag,
                          f"https://github.com/{REPOSITORY}/releases/download/{tag}/{INSTALLER}",
                          manifest["installer"]["size"], manifest["installer"]["sha256"])
    folder = cache.root(Path(home))
    target = cache.installer_path(folder, candidate)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(installer, target)
    cache.verify(target, candidate)
    cache.write(folder / "pending.json", {"schema": 1, "app_id": APP_ID, "candidate": candidate.as_dict()})
    cache.write(folder / "result.json", {"status": "pending", "tag": tag})
    return target


def _rows(db: Path, sql: str, args=()) -> list:
    with sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True) as con:
        return con.execute(sql, args).fetchall()


def verify(home: Path, install_dir: Path, build_number: int) -> list[str]:
    home, problems = Path(home), []
    expected = json.loads((home / EXPECTED).read_text(encoding="utf-8"))
    db = home / "prep.db"

    def need(ok, message):
        if not ok:
            problems.append(message)

    need(_rows(db, "SELECT COUNT(*) FROM attempt WHERE problem_id=? AND solved=1 AND help='none'",
               (expected["attempt_problem"],))[0][0] == 1, "seeded no-help attempt is missing")
    need(_rows(db, "SELECT COUNT(*) FROM starstory WHERE title=?", (expected["star_title"],))[0][0] == 1,
         "seeded STAR story is missing")
    need(_rows(db, "SELECT COUNT(*) FROM trainingcheckin WHERE intention=?", (expected["intention"],))[0][0] == 1,
         "seeded habit check-in is missing")
    settings = dict(_rows(db, "SELECT key, value FROM setting"))
    need(settings.get("design:ml-rag:notes") == expected["design_notes"], "design notes changed or are missing")
    need(json.loads(settings.get("design:ml:learned", "[]")) == expected["design_learned"],
         "design checklist changed or is missing")
    active = home / "sessions" / "active.json"
    session = json.loads(active.read_text(encoding="utf-8")) if active.exists() else {}
    need(session.get("problem_id") == expected["session_problem"], "saved session recovery data is missing")
    scratch = home / "sessions" / "G-217_ota_smoke.txt"
    need(scratch.exists() and scratch.read_text(encoding="utf-8") == expected["scratch_text"], "saved code file changed")

    info_file = Path(install_dir) / "_internal" / "data" / "build_info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    need(info.get("build_number") == int(build_number), f"installed build is {info.get('build_number')}, not {build_number}")

    result_file = home / "updates" / "result.json"
    result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
    need(result.get("status") == "installed", f"update result is {result.get('status')!r}, not 'installed'")
    need(result.get("tag", "").endswith(f"-b{build_number}"), "update result is for a different build")

    backups = sorted((home / "updates" / "backups").glob("prep-*.db")) if (home / "updates" / "backups").exists() else []
    need(bool(backups), "no pre-update database backup was made")
    if backups:
        need(_rows(backups[-1], "PRAGMA integrity_check")[0][0] == "ok", "the database backup is damaged")
        need(_rows(backups[-1], "SELECT COUNT(*) FROM attempt WHERE problem_id=?",
                   (expected["attempt_problem"],))[0][0] == 1, "the backup does not hold the seeded progress")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OTA upgrade smoke test")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seed")
    p.add_argument("--home", type=Path, required=True)
    p = sub.add_parser("stage")
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--installer", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--install-dir", type=Path, required=True)
    p.add_argument("--build", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "seed":
        print("seeded:", ", ".join(seed(args.home)))
    elif args.command == "stage":
        print("staged:", stage(args.home, args.installer, args.manifest))
    else:
        problems = verify(args.home, args.install_dir, args.build)
        for line in problems:
            print("FAIL:", line)
        if problems:
            return 1
        print("upgrade ok: progress, STAR story, check-in, design notes, session recovery, build identity, backup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
