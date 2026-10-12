"""The Phase 0 lessons and the concept vault teach by example, so every example must be right:
each Python block runs with its asserts passing, each C++ block compiles (g++ -std=c++17) and runs,
and each predict-the-output exercise prints exactly the option marked correct."""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from google_prep import db
from google_prep.core import scheduler
from google_prep.modules import fasttrack

VAULT = db.resource_dir() / "patterns_vault"
SPEC_NOTES = [
    "00_python_basics_for_cpp_devs.md", "00_python_datastructures_for_cpp_devs.md", "01_arrays_strings.md",
    "02_stacks_monotonic.md", "03_queues_deques.md", "04_heaps_priority_queues.md", "05_hash_tables_sets.md",
    "06_linked_lists.md", "07_trees_bst.md", "08_trie_prefix_trees.md", "09_graphs_union_find.md",
    "10_dynamic_programming.md",
]
BLOCK = re.compile(r"```(python|cpp)\n(.*?)```", re.S)
# GCC on Linux is the reference compiler; other platforms' toolchains differ (clang has no bits/ headers).
HAVE_GCC = shutil.which("g++") is not None and platform.system() == "Linux"


def run_python(code: str, tmp: Path) -> subprocess.CompletedProcess:
    src = tmp / f"s{abs(hash(code))}.py"
    src.write_text(code, encoding="utf-8")
    return subprocess.run([sys.executable, "-I", str(src)], capture_output=True, text=True, timeout=60)


def run_cpp(code: str, tmp: Path) -> subprocess.CompletedProcess:
    stem = f"c{abs(hash(code))}"
    src, exe = tmp / f"{stem}.cpp", tmp / stem
    src.write_text(code, encoding="utf-8")
    r = subprocess.run(["g++", "-std=c++17", "-O0", "-o", str(exe), str(src)], capture_output=True, text=True, timeout=120)
    return r if r.returncode else subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)


def failures(jobs, tmp: Path) -> list[str]:
    """Run (label, lang, code) jobs in parallel; return the ones that failed."""
    def one(job):
        label, lang, code = job
        r = (run_python if lang == "python" else run_cpp)(code, tmp)
        return None if r.returncode == 0 else f"{label} [{lang}]\n{code}\n--- stderr ---\n{r.stderr[-1200:]}"

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        return [f for f in pool.map(one, jobs) if f]


# ---- the 12 vault notes ---------------------------------------------------------------------------
def test_vault_has_the_spec_notes():
    assert sorted(p.name for p in VAULT.glob("*.md")) == sorted(SPEC_NOTES)
    for name in SPEC_NOTES:
        text = (VAULT / name).read_text(encoding="utf-8")
        assert text.startswith("# "), name
        langs = {lang for lang, _ in BLOCK.findall(text)}
        assert langs == {"python", "cpp"}, f"{name}: needs C++ next to Python, got {langs}"
        assert "|" in text, f"{name}: needs a complexity table"


@pytest.mark.parametrize("name", SPEC_NOTES)
def test_vault_python_blocks_run(name, tmp_path):
    blocks = [(name, lang, code) for lang, code in BLOCK.findall((VAULT / name).read_text(encoding="utf-8"))
              if lang == "python"]
    assert blocks
    bad = failures(blocks, tmp_path)
    assert not bad, "\n\n".join(bad)


@pytest.mark.skipif(not HAVE_GCC, reason="C++ examples are checked with GCC on Linux")
@pytest.mark.parametrize("name", SPEC_NOTES)
def test_vault_cpp_blocks_compile_and_run(name, tmp_path):
    blocks = [(name, lang, code) for lang, code in BLOCK.findall((VAULT / name).read_text(encoding="utf-8"))
              if lang == "cpp"]
    assert blocks
    bad = failures(blocks, tmp_path)
    assert not bad, "\n\n".join(bad)


def test_patterns_point_at_existing_notes():
    for p in db.load_json("patterns.json"):
        assert p["vault_file"] is None or (VAULT / p["vault_file"]).exists(), p["id"]


# ---- Phase 0 lessons ----------------------------------------------------------------------------
@pytest.mark.parametrize("day", fasttrack.DAYS)
def test_lesson_schema(day):
    lesson = fasttrack.load_day(day)
    assert lesson["day"] == day and lesson["minutes"] == 60
    assert (VAULT / lesson["vault"]).exists()
    ids = [s["id"] for s in lesson["sections"]] + [e["id"] for e in lesson["exercises"]]
    assert len(ids) == len(set(ids))
    assert all(i.startswith(f"d{day}-") for i in ids)
    assert len(lesson["exercises"]) >= 5  # spec: at least 5 interactive syntax exercises
    for s in lesson["sections"]:
        assert s["title"] and s["points"] and s["cpp"].strip() and s["python"].strip() and s["gotcha"]
    for e in lesson["exercises"]:
        assert len(e["options"]) == 3 and len(set(e["options"])) == 3 and e["answer"] in (0, 1, 2)
        assert e["prompt"] and e["explanation"] and e["topic"]


@pytest.mark.parametrize("day", fasttrack.DAYS)
def test_lesson_examples_run(day, tmp_path):
    lesson = fasttrack.load_day(day)
    jobs = [(s["id"], "python", s["python"]) for s in lesson["sections"]]
    if HAVE_GCC:
        jobs += [(s["id"], "cpp", s["cpp"]) for s in lesson["sections"]]
    bad = failures(jobs, tmp_path)
    assert not bad, "\n\n".join(bad)


@pytest.mark.parametrize("day", fasttrack.DAYS)
def test_exercise_answers_match_real_output(day, tmp_path):
    for e in fasttrack.load_day(day)["exercises"]:
        r = run_python(e["code"], tmp_path)
        assert r.returncode == 0, (e["id"], r.stderr)
        out = r.stdout.strip()
        assert out == e["options"][e["answer"]], (e["id"], out, e["options"])
        assert sum(o == out for o in e["options"]) == 1, e["id"]


def test_day1_covers_the_spec_syllabus():
    text = json.dumps(fasttrack.load_day(1))
    for must in ("//", "-7 % 2", "int(a / b)", "float('inf')", "0xFFFFFFFF", "~", "join", "[[0] * ", "deepcopy"):
        assert must in text, must


def test_day2_covers_the_spec_syllabus():
    text = json.dumps(fasttrack.load_day(2))
    for must in ("pop(0)", "deque", "popleft", "heapq", "heappush", "defaultdict", "Counter", "ListNode",
                 "bisect_left", "bisect_right", "lru_cache"):
        assert must in text, must


# ---- schedule: Day 1 and 2 serve the modules, Day 3 starts the bank -------------------------------
@pytest.mark.fasttrack
def test_today_serves_day1_then_day2_then_the_bank(session):
    from datetime import datetime, timedelta

    from google_prep.modules import today

    start = scheduler.start_date(session, datetime(2026, 10, 12).date())
    day1 = datetime.combine(start, datetime.min.time()).replace(hour=8)
    d = today.decide(session, day1)
    assert d["kind"] == "fasttrack" and d["day"] == 1
    fasttrack.complete(session, 1, day1, 55)
    assert today.decide(session, day1)["kind"] == "done"
    day2 = day1 + timedelta(days=1)
    assert today.decide(session, day2)["day"] == 2
    fasttrack.complete(session, 2, day2, 60)
    day3 = day1 + timedelta(days=2)
    d3 = today.decide(session, day3)
    assert d3["kind"] == "warmup" and scheduler.phase_for_day(scheduler.day_number(session, day3.date())) == 1


@pytest.mark.fasttrack
def test_skipped_day1_is_served_on_day2(session):
    from datetime import datetime, timedelta

    from google_prep.modules import today

    start = scheduler.start_date(session, datetime(2026, 10, 12).date())
    day2 = datetime.combine(start + timedelta(days=1), datetime.min.time()).replace(hour=8)
    assert today.decide(session, day2)["day"] == 1  # the earliest unfinished module comes first
