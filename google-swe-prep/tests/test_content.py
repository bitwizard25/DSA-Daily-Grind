"""The Phase 0 lessons and the concept vault teach by example, so every example must be right and
easy to read for a returning learner (see tests/snippets.py):
- every Python block runs, and every `print(...)  # output` comment is exactly what it prints;
- any C++ block compiles (g++ -std=c++17) and runs;
- each quick check prints exactly the option marked correct;
- lesson text and notes stay short, teach each new word once, and avoid expert trivia."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest

import snippets
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
HAVE_GCC = snippets.have_gcc()


def run_all(checks) -> list[str]:
    """Run (label, check, code) jobs in parallel; return the failures."""
    def one(job):
        label, check, code = job
        err = check(code)
        return None if err is None else f"{label}\n{code}\n--- problem ---\n{err}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        return [f for f in pool.map(one, checks) if f]


# ---- the 12 vault notes ---------------------------------------------------------------------------
def test_vault_has_the_spec_notes():
    assert sorted(p.name for p in VAULT.glob("*.md")) == sorted(SPEC_NOTES)


@pytest.mark.parametrize("name", SPEC_NOTES)
def test_vault_note_is_plain_and_complete(name):
    text = (VAULT / name).read_text(encoding="utf-8")
    problems = snippets.lint_note(text, name)
    assert not problems, "\n".join(problems)
    assert "## Coming from C++" in text and "|" in text  # a short C++ -> Python table, and a speed table


@pytest.mark.parametrize("name", SPEC_NOTES)
def test_vault_code_runs_and_comments_tell_the_truth(name):
    text = (VAULT / name).read_text(encoding="utf-8")
    jobs = [(f"{name} python", snippets.check_python, code) for lang, code in snippets.BLOCK.findall(text) if lang == "python"]
    if HAVE_GCC:
        jobs += [(f"{name} cpp", snippets.check_cpp, code) for lang, code in snippets.BLOCK.findall(text) if lang == "cpp"]
    assert any(lang == "python" for lang, _ in snippets.BLOCK.findall(text)), f"{name}: needs a Python template"
    bad = run_all(jobs)
    assert not bad, "\n\n".join(bad)


def test_patterns_point_at_existing_notes():
    for p in db.load_json("patterns.json"):
        assert p["vault_file"] is None or (VAULT / p["vault_file"]).exists(), p["id"]


# ---- Phase 0 lessons ----------------------------------------------------------------------------
@pytest.mark.parametrize("day", fasttrack.DAYS)
def test_lesson_is_plain_and_complete(day):
    lesson = fasttrack.load_day(day)
    assert lesson["day"] == day and lesson["minutes"] == 60
    assert (VAULT / lesson["vault"]).exists()
    ids = [s["id"] for s in lesson["sections"]] + [e["id"] for e in lesson["exercises"]]
    assert len(ids) == len(set(ids)) and all(i.startswith(f"d{day}-") for i in ids)
    assert len(lesson["exercises"]) >= 5  # spec: at least 5 interactive syntax exercises
    problems = snippets.lint_lesson(lesson)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("day", fasttrack.DAYS)
def test_lesson_examples_run_and_comments_tell_the_truth(day):
    lesson = fasttrack.load_day(day)
    bad = run_all([(s["id"], snippets.check_python, s["python"]) for s in lesson["sections"]])
    assert not bad, "\n\n".join(bad)


@pytest.mark.parametrize("day", fasttrack.DAYS)
def test_exercise_answers_match_real_output(day):
    for e in fasttrack.load_day(day)["exercises"]:
        out = fasttrack.run_snippet(e["code"])
        assert out == e["options"][e["answer"]], (e["id"], out, e["options"])
        assert sum(o == out for o in e["options"]) == 1, e["id"]


@pytest.mark.parametrize("day", fasttrack.DAYS)
def test_simple_mode_has_every_text(day):
    """The language switch must never show an empty field."""
    lesson = fasttrack.load_day(day)
    for s in lesson["sections"]:
        for key in ("title", "summary", "analogy", "gotcha"):
            assert fasttrack.text(s, key, "simple") and fasttrack.text(s, key, "standard"), (s["id"], key)
    for g in lesson["glossary"]:
        assert g["meaning_simple"] and g["meaning"], g["key"]


def test_hindi_hints_are_short_and_cover_the_glossary():
    hints = fasttrack.load_hints()
    assert len(hints) >= 40
    for key, hint in hints.items():
        assert key == key.lower() and 0 < len(hint) <= 45, (key, hint)
        assert not re.search("[\u0900-\u097F]", hint), f"{key}: use Roman-script Hinglish"


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


# ---- simple mode beyond the lessons ------------------------------------------------------------------
def test_simple_texts_cover_the_bank_and_the_drills():
    texts = db.load_json("simple_text.json")
    problems = {p["id"]: p for p in db.load_json("problems.json")}
    drills = {d["id"]: d for d in db.load_json("recall_drills.json")}
    assert set(texts["problems"]) == set(problems) and set(texts["drills"]) == set(drills)
    for pid, t in texts["problems"].items():
        assert t["statement"] and len(t["statement"]) <= 220, pid
        assert t["follow_up"] and len(t["follow_up"]) <= 180, pid
        assert t["key_insight"] and len(t["key_insight"]) <= 200, pid
    for did, t in texts["drills"].items():
        assert t["prompt"] and t["pattern"] and t["edge_case"], did
        if drills[did]["type"] == "bug_hunt" and "\n\n" in drills[did]["prompt"]:
            code = drills[did]["prompt"].split("\n\n", 1)[1]
            assert t["prompt"].endswith(code), f"{did}: the bug-hunt code must stay exactly the same"


def test_code_hints_exist_for_everyday_words():
    hints = fasttrack.load_hints("code_hints")
    assert {"join", "key", "set"} <= set(hints)
    assert not {"join", "key", "set"} & set(fasttrack.load_hints())  # never matched in plain prose
