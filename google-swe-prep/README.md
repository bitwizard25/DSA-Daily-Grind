# google-prep: Google SWE II Interview OS

Offline interview simulator and accountability coach for an 8-week Google SWE II plan.
Spec: [`google_swe_prep_spec.md`](./google_swe_prep_spec.md).

Two front ends, one database (`~/.google_prep/prep.db`):

| | Launch | For |
|---|---|---|
| **Desktop app** | double-click `google-prep.exe` (or `google-prep ui`) | The main way to use it: a real window with Today, the Python fast-track, drills, the 45-min whiteboard, roadmap, STAR studio and the concept vault |
| **Terminal** | `google-prep-cli.exe` (or `google-prep` from source) | Every command, scripting, `remind`, `sync`; `google-prep-cli tui` is the full-screen terminal version |

## Get the app

**Windows (recommended): install it once, like any desktop app.**

1. GitHub → **Actions** → latest green **build-exe** run → **Artifacts** → **GoogleSWEPrep-Setup-Windows**.
2. Unzip it and run `GoogleSWEPrep-Setup.exe`. It installs for your user only (no admin prompt) into
   `%LOCALAPPDATA%\Programs\Google SWE II OS` and adds a **Start menu** entry and a **desktop shortcut**.
3. Open **Google SWE II OS** from the Start menu or the desktop. To pin it, right-click the Start menu entry →
   **Pin to taskbar** (pinning the open window would pin the browser instead).

Updating: run a newer `GoogleSWEPrep-Setup.exe`; it replaces the old version in place. Uninstalling:
**Settings → Apps → Google SWE II OS → Uninstall**. Neither touches your progress, which is saved in
`%USERPROFILE%\.google_prep\` (shown in the app under **Settings**).

Windows SmartScreen may say "Windows protected your PC" because the installer is not code-signed
(that needs a paid certificate): click **More info → Run anyway**.

**Other ways:**

- **Portable (no install):** artifact **google-prep-Windows** holds `google-prep.exe` (the window) and
  `google-prep-cli.exe` (the terminal app). Unzip first, then double-click `google-prep.exe`.
- **macOS / Linux:** artifacts **google-prep-macOS** / **google-prep-Linux** from the same run.
- **Build it yourself** (Python 3.11+): `cd google-swe-prep`, `pip install -r requirements.txt`, then
  `python build.py` (portable programs) or `python build.py --installer` (the app folder, plus
  `dist/GoogleSWEPrep-Setup.exe` on Windows with [Inno Setup 6](https://jrsoftware.org/isdl.php)).
- **From source:** `pip install -e .` then `google-prep ui` (window) or `google-prep` (terminal).

### How the window works

The app starts a small server on `127.0.0.1` only and opens it in a chromeless Microsoft Edge or Chrome
window (Edge ships with Windows 10/11) with its own profile, so it behaves like a normal desktop app:
closing the window quits it. Every request carries a secret key made at launch, the page refuses other
hosts (no DNS-rebinding) and loads nothing from the internet. Without Edge/Chrome it opens in your
default browser and quits a few minutes after the tab closes. `GOOGLE_PREP_BROWSER` picks another
Chromium browser. The same database is used by both front ends, so you can switch any time.

Keys in the window: `Enter` does the highlighted main action on every screen; `t f d w r s v` jump to
Today, Fast-Track, Drills, Mock, Roadmap, STAR, Vault; `?` lists everything.

## Days 1–2: C++ → Python fast-track (Phase 0)

The plan starts with two 60-minute modules for C++ programmers switching to Python. Today serves them
on Days 1 and 2; the Google problem bank starts on Day 3.

| Day | Covers |
|---|---|
| 1 | Arbitrary-precision `int`, `//` and `%` on negatives vs C++ truncation, `int(a / b)`, `inf`, bitwise NOT and 32-bit masks, string immutability and `''.join`, slicing, references and the `[[0] * c] * r` aliasing bug, shallow vs deep copy |
| 2 | `list` as vector/stack (never `pop(0)`), `deque`, `heapq` (min-heap, negate for max, tuples), `dict` / `defaultdict` / `Counter`, `set`, classes for ListNode/TreeNode, tuple swaps, `bisect` = lower/upper_bound, sorting with keys, `@lru_cache` |

Each section puts the C++ you know next to the Python to write, plus one trap. Each day ends with
10 predict-the-output exercises; the answer shows what Python really prints (the snippet is run).
Every code block in the lessons and the vault is executed (Python) or compiled and run (C++,
`g++ -std=c++17`) by the test suite, so the side-by-side examples are known to be correct.

## Daily use

| When | Do | What happens |
|---|---|---|
| Any time | open the app | Today shows one **Next** card: fast-track (Days 1–2), AM recall warmup, PM timed solve (mock in weeks 7–8). An unfinished drill or mock appears first with **Continue**. |
| Exhausted | **Micro-Mode** on Today (`m`), or `google-prep-cli today --worked 11` | 10 minutes (1 pattern drill + 1 bug-hunt); counts toward the streak. |
| Mock | **Mock & Solve** (`w`) | Recommended weakest-topic problem, or pick any of the 75; then the 45-minute whiteboard below. |
| Weekly | **Roadmap** (`r`) | Readiness gauge with the formula, 14-day trend, 8-week plan, topic mastery, mistake profile. |
| STAR | **STAR Stories** (`s`) | Write and edit stories for Ambiguity, Ownership, Disagreement, Humility; 2-minute practice prompts. |
| Reference | **Concept Vault** (`v`) | 12 notes: C++ vs Python, complexity tables, templates and traps per pattern. |

Every screen shows the same frame: the section list on the left, a breadcrumb on top
(`Google SWE II OS › Phase 1 › Week 2 › Day 10 › Mock › Graphs › Course Schedule II`, current step in
bold), one title, one main button. Everything is logged automatically when a drill, solve or mock ends.

## The 45-minute whiteboard

Clock and phase bar on top, problem and live checklist on the left, a plain-text code pad on the
right (no syntax highlighting, no autocomplete, nothing runs), plus a **Dry-run trace** tab where you
step through your own code by hand.

| Time | Phase |
|---|---|
| 00:00–05:00 | Scoping: clarify constraints and edge cases |
| 05:00–35:00 | Code in plain text; trace a non-trivial input by hand |
| 35:00–40:00 | A Google follow-up banner drops in (streaming, memory caps, sharding) |
| 40:00–45:00 | 4-pillar scorecard: Analytical Ability, Coding & Craft, Verification & Rigor, System Scaling (25% each) |

Window keys: `Alt+1…4` tick the checklist · `Tab` indents 4 spaces, `Enter` keeps the indent ·
`Esc` leaves the pad · `Alt+P` pause · `Ctrl+S` submit. Scorecard: `[1] Missed it` /
`[2] Solved with Effort` / `[3] Instant Reflex` (reschedules the problem today / in 3 days / next
week), then one `y`/`n` per pillar check (your ticks are pre-noted), `Backspace` steps back.
Every keystroke is saved to `~/.google_prep/sessions/`; after a crash the app opens with
"House Robber (18m remaining) · Continue". Terminal keys for `google-prep-cli mock` are listed by `?`.

## Recall grading

`[1] Missed it` → again today · `[2] Solved with Effort` → in 3 days · `[3] Instant Reflex` → next week
(SM-2 stretches the gap further as you keep getting it). "Instant" only counts if you revealed the
answer within 20 seconds. Drills are unlabeled and rotate across topics (interleaved, not blocked).

## Problem bank (75, Google only)

Weighted per spec section 4. Every problem was reported at Google in the last 6 months (34 in the
last 30 days), each with a `key_insight`, `edge_cases` and a `google_follow_up`. Ids are the LeetCode
number (`G-200` = Number of Islands).

| Group | Share | Categories |
|---|---|---|
| Graphs & DAG execution | 35% (26) | Graphs |
| Concurrency, queues & intervals | 25% (19) | Heaps 7, Intervals 6, Monotonic Queues 6 |
| Trees, tries & hierarchical retrieval | 20% (15) | Trees (incl. binary lifting, autocomplete trie) |
| DP & sequence search | 20% (15) | DP |

Picking: weakest category first, then failed problems due for a re-solve, then the problems asked at
Google most recently (data: [liquidslr/leetcode-company-wise-problems](https://github.com/liquidslr/leetcode-company-wise-problems),
16 Aug 2026; `google-prep sync` refreshes it, `hot` lists what you are missing, `add <slug>` imports one).

Not covered by the six categories: arrays/hashing, two pointers, sliding window, binary search,
linked lists and design (e.g. LRU Cache, Minimum Window Substring). Google asks these often.

## Readiness index

`Readiness = 0.40 Pattern Mastery + 0.30 Mock Speed + 0.15 Dry-Run Accuracy + 0.15 STAR Stories`
(STAR: 2 stories per dimension.) The roadmap prints the formula with your actual numbers.

Text colours are Google's dark-theme tints (all ≥ 7:1 contrast on the slate background, WCAG AAA);
brand colours are used only for borders.

## Other commands

`fasttrack [--day 1|2]` · `solve [id]` · `problems` · `patterns [n]` · `setup --name --target --interview-date` ·
`remind` (OS scheduler commands for 07:30 / 19:30) · `resume` · `ui` (window) · `tui` (full-screen terminal).

## Files

- `~/.google_prep/prep.db`: everything (override the folder with `GOOGLE_PREP_HOME`).
- `~/.google_prep/sessions/`: the auto-saved session (`active.json`) and every scratchpad and trace.
- Upgrades keep your progress: renamed fields are migrated in place, and progress on a retired problem
  moves to the same LeetCode problem's new id.

## Tests

`pip install -r requirements.txt` then `python -m pytest`. The suite covers the scheduler, SM-2,
readiness, the CLI, the terminal UI, the window's API and server guards (token, host check, path
traversal), and runs every lesson and vault code block (C++ blocks need `g++`; they are skipped
without it).
