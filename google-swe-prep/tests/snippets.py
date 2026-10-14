"""Check lesson and vault code the way a learner reads it.

Python: the snippet runs in a fresh interpreter. A `print(...)` line that ends with a comment
`# <text>` must print exactly <text> every time it runs: that comment is the answer the learner
reads, so it must be true. `assert` lines must pass too.

C++: a snippet without `main` is shown short and checked inside a wrapper program
(`#include <bits/stdc++.h>`, `using namespace std;`, `int main() { ... }`). It must compile with
g++ -std=c++17 and run with exit code 0.

Also a plain-language lint: lesson text (two registers, see the writing guide) and vault prose stay
short, every new word is taught once, and each core lesson step has exactly one quick check.

Command line (used while writing content):
    python tests/snippets.py data/phase0_day1.json patterns_vault/01_arrays_strings.md ...
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS = r'''
import builtins, io, io as _io, json, runpy, sys, tokenize

src = sys.argv[1]
text = open(src, encoding="utf-8").read()
comments, print_lines = {}, set()
toks = list(tokenize.generate_tokens(_io.StringIO(text).readline))
for i, tok in enumerate(toks):
    if tok.type == tokenize.COMMENT:
        comments[tok.start[0]] = tok.string[1:].strip()
    if tok.type == tokenize.NAME and tok.string == "print" and i + 1 < len(toks) and toks[i + 1].string == "(":
        print_lines.add(tok.start[0])
expected = {line: comments[line] for line in print_lines if line in comments}
seen = {}
real_print = builtins.print

def checked_print(*args, **kwargs):
    buf = io.StringIO()
    kwargs = dict(kwargs)
    kwargs.pop("file", None)
    real_print(*args, file=buf, **kwargs)
    out = buf.getvalue()
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename != src:
        frame = frame.f_back
    if frame is not None:
        seen.setdefault(frame.f_lineno, []).append(out.rstrip("\n"))
    sys.stdout.write(out)

builtins.print = checked_print
runpy.run_path(src, run_name="__main__")
problems = []
for line, want in sorted(expected.items()):
    got = seen.get(line)
    if not got:
        problems.append(f"line {line}: print never ran (comment says {want!r})")
    for g in got or []:
        if g != want:
            problems.append(f"line {line}: prints {g!r} but the comment says {want!r}")
if problems:
    sys.stderr.write("\n".join(problems) + "\n")
    sys.exit(3)
'''

CPP_WRAP = "#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n%s\n    return 0;\n}\n"
HAS_MAIN = re.compile(r"\bint\s+main\s*\(")


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="snip-"))


def check_python(code: str) -> str | None:
    """None when the snippet runs and every commented print matches; else the problem."""
    folder = _tmp()
    try:
        src, harness = folder / "snippet.py", folder / "harness.py"
        src.write_text(code, encoding="utf-8")
        harness.write_text(HARNESS, encoding="utf-8")
        r = subprocess.run([sys.executable, "-I", str(harness), str(src.resolve())], capture_output=True,
                           text=True, timeout=60, cwd=folder)
        return None if r.returncode == 0 else (r.stderr.strip()[-1500:] or f"exit {r.returncode}")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def cpp_program(code: str) -> str:
    if HAS_MAIN.search(code):
        return code
    return CPP_WRAP % "\n".join("    " + line for line in code.splitlines())


def check_cpp(code: str) -> str | None:
    if not HAS_MAIN.search(code) and "#include" in code:
        return "a C++ snippet with #include lines must be a full program with main()"
    folder = _tmp()
    try:
        src, exe = folder / "snippet.cpp", folder / "snippet"
        src.write_text(cpp_program(code), encoding="utf-8")
        r = subprocess.run(["g++", "-std=c++17", "-O0", "-o", str(exe), str(src)], capture_output=True,
                           text=True, timeout=120)
        if r.returncode == 0:
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
        return None if r.returncode == 0 else (r.stderr.strip()[-1500:] or f"exit {r.returncode}")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def have_gcc() -> bool:
    return shutil.which("g++") is not None and sys.platform.startswith("linux")


# ---- plain-language lint -------------------------------------------------------------------------
TRIVIA = ["__builtin", "bit_length", "bit_count", "CPython", "4300", "Karatsuba", "sys.maxsize", "word-size",
          "arbitrary precision", "amortised", "idempotent"]  # expert trivia that confuses a returning learner


def sentence_problems(text: str, limit: int = 25) -> list[str]:
    """Sentences longer than `limit` words (code spans count as one word)."""
    plain = re.sub(r"`[^`]*`", "X", text)
    out = []
    for sentence in re.split(r"(?<=[.!?:])\s+", plain):
        words = sentence.split()
        if len(words) > limit:
            out.append(f"{len(words)}-word sentence: {sentence[:90]}...")
    return out


# field -> (max characters, max words per sentence); *_simple fields are shorter and plainer
TEXT_LIMITS = {"title": (36, 25), "title_simple": (36, 16), "summary": (120, 25), "summary_simple": (100, 16),
               "analogy": (260, 25), "analogy_simple": (220, 16), "gotcha": (180, 25), "gotcha_simple": (150, 16)}


def lint_lesson(lesson: dict) -> list[str]:
    problems = []
    for key in ("title", "title_simple", "subtitle", "subtitle_simple", "glossary", "sections", "exercises"):
        if not lesson.get(key):
            problems.append(f"lesson: missing {key}")
    glossary = {g.get("key"): g for g in lesson.get("glossary", [])}
    for g in lesson.get("glossary", []):
        for key in ("key", "term", "meaning", "meaning_simple", "example"):
            if not g.get(key):
                problems.append(f"glossary {g.get('key')}: missing {key}")
    taught: dict[str, str] = {}
    core = [s for s in lesson.get("sections", []) if not s.get("optional")]
    for s in lesson.get("sections", []):
        sid = s.get("id", "?")
        for key in ("title", "title_simple", "python", "points", "points_simple", "summary", "summary_simple",
                    "analogy", "analogy_simple", "bridge", "gotcha", "gotcha_simple"):
            if not s.get(key):
                problems.append(f"{sid}: missing {key}")
        for key, (chars, words) in TEXT_LIMITS.items():
            text = s.get(key, "")
            if len(text) > chars:
                problems.append(f"{sid}: {key} longer than {chars} characters")
            problems += [f"{sid} {key}: {p}" for p in sentence_problems(text, words)]
        for key, chars, words in (("points", 130, 25), ("points_simple", 110, 16)):
            items = s.get(key, [])
            if len(items) > 3:
                problems.append(f"{sid}: more than 3 {key}")
            for item in items:
                if len(item) > chars:
                    problems.append(f"{sid}: a {key} entry is longer than {chars} characters: {item[:50]}...")
                problems += [f"{sid} {key}: {p}" for p in sentence_problems(item, words)]
        if len(s.get("python", "").splitlines()) > 12:
            problems.append(f"{sid}: python example longer than 12 lines")
        bridge = s.get("bridge", [])
        if not 1 <= len(bridge) <= 3:
            problems.append(f"{sid}: needs 1 to 3 bridge rows")
        for row in bridge:
            if not row.get("cpp") or not row.get("python"):
                problems.append(f"{sid}: bridge rows need cpp and python")
            for key in ("cpp", "python"):
                if len(row.get(key, "")) > 60:
                    problems.append(f"{sid}: bridge {key} longer than 60 characters")
            if len(row.get("note", "")) > 60:
                problems.append(f"{sid}: bridge note longer than 60 characters")
        words = s.get("new_words", [])
        if len(words) > 3:
            problems.append(f"{sid}: more than 3 new_words")
        for w in words:
            if w not in glossary:
                problems.append(f"{sid}: new word {w!r} is not in the glossary")
            elif w in taught:
                problems.append(f"{sid}: new word {w!r} was already taught in {taught[w]}")
            taught[w] = sid
    problems += [f"glossary {k}: never taught in a step's new_words" for k in glossary if k not in taught]
    optional = [s for s in lesson.get("sections", []) if s.get("optional")]
    if len(optional) > 1 or (optional and lesson["sections"][-1] is not optional[0]):
        problems.append("at most one optional step, and it must be last")
    by_section = {}
    for e in lesson.get("exercises", []):
        by_section.setdefault(e.get("section"), []).append(e["id"])
        for key in ("section", "topic", "prompt", "code", "options", "explanation", "explanation_simple"):
            if key not in e:
                problems.append(f"{e.get('id')}: missing {key}")
        if len(e.get("explanation", "")) > 200:
            problems.append(f"{e['id']}: explanation longer than 200 characters")
        if len(e.get("explanation_simple", "")) > 170:
            problems.append(f"{e['id']}: explanation_simple longer than 170 characters")
        if len(e.get("code", "").splitlines()) > 6:
            problems.append(f"{e['id']}: exercise code longer than 6 lines")
    for s in core:
        if len(by_section.get(s["id"], [])) != 1:
            problems.append(f"{s['id']}: needs exactly one exercise (has {len(by_section.get(s['id'], []))})")
    for s in optional:
        if by_section.get(s["id"]):
            problems.append(f"{s['id']}: optional steps have no exercise")
    blob = json.dumps(lesson)
    problems += [f"expert trivia in the lesson: {w}" for w in TRIVIA if w in blob]
    return problems


BLOCK = re.compile(r"```(python|cpp)\n(.*?)```", re.S)
VAULT_HEADINGS = ["## When to use it", "## The idea in plain words", "## Python template", "## Coming from C++",
                  "## How fast is it?", "## Common mistakes", "## Practice in the app"]


def lint_note(text: str, name: str) -> list[str]:
    problems = []
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        problems.append(f"{name}: first line must be '# Title'")
    if not any(line.startswith("> In one line:") for line in lines[:6]):
        problems.append(f"{name}: needs '> In one line: ...' right under the title")
    for heading in VAULT_HEADINGS:
        if not any(line.startswith(heading) for line in lines):
            problems.append(f"{name}: missing section '{heading}'")
    if not 80 <= len(lines) <= 260:
        problems.append(f"{name}: {len(lines)} lines (aim for 120 to 220)")
    in_code = False
    for no, line in enumerate(lines, 1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.startswith("|") or not line.strip():
            continue
        if len(line) > 220:
            problems.append(f"{name}:{no}: prose line longer than 220 characters")
        problems += [f"{name}:{no}: {p}" for p in sentence_problems(line)]
    problems += [f"{name}: expert trivia: {w}" for w in TRIVIA if w in text]
    return problems


def check_lesson_file(path: Path, cpp: bool = True) -> list[str]:
    lesson = json.loads(path.read_text(encoding="utf-8"))
    problems = lint_lesson(lesson)
    for s in lesson["sections"]:
        if (err := check_python(s.get("python", ""))):
            problems.append(f"{s['id']} python: {err}")
    for e in lesson.get("exercises", []):
        r = subprocess.run([sys.executable, "-I", "-c", e["code"]], capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        if r.returncode:
            problems.append(f"{e['id']}: code fails: {r.stderr[-300:]}")
        elif "\n" in out:
            problems.append(f"{e['id']}: must print exactly one line, got {out!r}")
        elif out != e["options"][e["answer"]]:
            problems.append(f"{e['id']}: prints {out!r} but the marked answer is {e['options'][e['answer']]!r}")
        if len(set(e["options"])) != 3:
            problems.append(f"{e['id']}: needs 3 different options")
    return problems


def check_note_file(path: Path, cpp: bool = True) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems = lint_note(text, path.name)
    for lang, code in BLOCK.findall(text):
        err = check_python(code) if lang == "python" else (check_cpp(code) if cpp else None)
        if err:
            problems.append(f"{path.name} {lang} block starting {code.strip().splitlines()[0][:50]!r}: {err}")
    return problems


def main(argv: list[str]) -> int:
    cpp = have_gcc()
    failed = False
    for arg in argv:
        path = Path(arg)
        problems = check_lesson_file(path, cpp) if path.suffix == ".json" else check_note_file(path, cpp)
        print(f"{path}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
        for p in problems:
            print("  -", p)
        failed |= bool(problems)
    if not cpp:
        print("(C++ not checked: needs g++ on Linux)")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv[1:]))
