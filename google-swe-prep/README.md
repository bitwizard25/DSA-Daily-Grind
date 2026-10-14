# google-prep: Google SWE II Interview OS

Offline interview simulator and accountability coach for an 8-week Google SWE II plan.
Spec: [`google_swe_prep_spec.md`](./google_swe_prep_spec.md).

Two front ends, one database (`~/.google_prep/prep.db`):

| | Launch | For |
|---|---|---|
| **Desktop app** | double-click `google-prep.exe` (or `google-prep ui`) | Today, Python lessons, drills, whiteboard, progress, interview habits, design learning checklists, STAR stories and the concept vault |
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

Upgrading from an older download: `google-prep.exe` is now the window. Terminal commands live in
`google-prep-cli.exe`; old reminders that run `google-prep.exe today` simply open the window on Today.

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

## Days 1–2: Python from zero, the interview way (Phase 0)

Two 60-minute modules for someone who used C++ and learned a little Python long ago. Today serves them on
Days 1 and 2; the Google problem bank starts on Day 3.

| Day | Steps |
|---|---|
| 1 | How Python code looks · variables and None · if/elif/else, and/or/not · loops and `range` · functions · numbers never overflow (`**`, `float('inf')`) · `/`, `//`, `%` with negative numbers · strings, slicing, `''.join` · lists and the shared-row trap · copies, `enumerate`, `zip` · optional: bit tricks |
| 2 | list as array and stack · `deque` · `heapq` · `dict` · `defaultdict` and `Counter` · `set` · nodes (`ListNode`, `TreeNode`) · sorting with `key` · `bisect` · recursion and `@lru_cache` |

Every step has the same small shape, one step per screen:

1. **New words** for this step (at most 3), shown with their meaning. A **Words you know** list grows as you go.
2. **Try it**: a tiny example; each `print` shows what it prints.
3. **What happened**: 1 to 3 short lines. **Remember**: one line to keep.
4. **Show another example** (`e`): an optional second example or comparison.
5. **Compare with C++ (optional)**: expand 1 to 3 one-line comparisons when useful.
6. **Watch out**: the one mistake people make here. **Quick check**: one question (`1`/`2`/`3`).

**I did not get this** (`u`) puts a step on a review list shown at the end of the day.

**Language**: *Standard* (technical English, every term explained the first time) or *Simple + Hindi hints*
(plain English, and the first time a technical word appears on a page it gets a short Hinglish hint, e.g.
"overflow (limit se bada ho jaana)"). Switch at the top of a lesson or a note, or in Settings.

Every Python example in the lessons and the vault is run by the test suite, and every `# output` comment
is checked against what Python really prints, so the simple examples are also correct.

## Design learning in Interview prep

Start with a checklist of what to learn, not a blank code editor:

- **LLD:** classes, interfaces, state, errors and test cases (10 topics).
- **HLD:** requirements, APIs, data, scaling, reliability and trade-offs (12 topics).
- **ML / GenAI:** retrieval, agents, evaluation and failure handling (8 topics).

Each topic has a plain explanation. Check it when you can explain it and give an example,
then use **Save learning progress**. Progress stays on this computer and is self-reported.
Six optional exercises offer guided prompts, a speaking timer and saved design notes.
Ask your recruiter which design or ML rounds apply to your role.

The reading guide uses **System Design for the LLM Era** (Sampriti Mitra) for production patterns
and **Generative AI System Design Interview** (Ali Aminian and Hao Sheng) for interview examples
and evaluation discussions. It gives chapter/page references for the supplied PDFs; the books
are not bundled with the app. Read one matching section, explain it in your own words, then practise.

Today shows required tasks completed, distinct coding practice, weekly no-help solves and
small coding, reasoning and habit milestones. These are practice goals, not hiring predictions.
If a whiteboard save fails, the leave dialog offers a text download and an explicit **Leave anyway**
option. A recovery draft belongs to that window's address; download it before reopening the app.

## Daily use

| When | Do | What happens |
|---|---|---|
| Any time | open the app | Today shows one **Next** card: fast-track (Days 1–2), AM recall warmup, PM timed solve (mock in weeks 7–8). An unfinished drill or mock appears first with **Continue**. |
| Exhausted | **Micro-Mode** on Today (`m`), or `google-prep-cli today --worked 11` | 10 minutes (1 pattern drill + 1 bug-hunt); counts toward the streak. |
| Mock | **Mock & Solve** (`w`) | Recommended weakest-topic problem, or pick any of the 75; then the 45-minute whiteboard below. |
| Weekly | **Roadmap** (`r`) | Readiness gauge with the formula, 14-day trend, 8-week plan, topic mastery, mistake profile. |
| STAR | **STAR Stories** (`s`) | Write and edit stories for Ambiguity, Ownership, Disagreement, Humility; 2-minute practice prompts. |
| Role | **Role Prep** (`j`) | 11-item checklist beyond coding: recruiter questions, resume deep-dive stories, defending your numbers, timeline answers, responsible AI, human mocks. Notes stay on your computer. |
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

**Struggle first.** `I'm stuck` (`Alt+H`) shows what to try (say it back, a tiny example, the slow
answer first, which tool fits). Hints unlock only after 25 minutes of a timed solve (15 in a mock),
and there are two: the pattern's name, then the key idea. Never code.

**Honest scorecard.** The first question is "Did you use any help?": none, a hint, or *I saw the
answer* (solution, video or AI code). A hint caps the result at Solved with Effort. A seen answer is
logged as **not solved**: the problem comes back first thing tomorrow, to code again from a blank pad.
If you opened a hint in the app, "No help" is not offered.

Window keys: `Alt+1…4` tick the checklist · `Tab` indents 4 spaces, `Enter` keeps the indent ·
`Esc` leaves the pad · `Alt+P` pause · `Alt+H` stuck · `Ctrl+S` save · `Ctrl+Enter` submit. Scorecard: `[1] Missed it` /
`[2] Solved with Effort` / `[3] Instant Reflex` (reschedules the problem today / in 3 days / next
week), then one `y`/`n` per pillar check (your ticks are pre-noted), `Backspace` steps back.
Every keystroke is saved to `~/.google_prep/sessions/`; after a crash the app opens with
"House Robber (18m remaining) · Continue". Terminal keys for `google-prep-cli mock` are listed by `?`.

## Recall grading

`[1] Missed it` → again today · `[2] Solved with Effort` → in 3 days · `[3] Instant Reflex` → next week
(SM-2 stretches the gap further as you keep getting it). "Instant" only counts if you revealed the
answer within 20 seconds. Drills are unlabeled and rotate across topics (interleaved, not blocked).

## Foundations first (36 problems, Days 3 onward)

Before the Google bank, the daily problem comes from 36 building blocks, in order, 16 of them easy:
arrays and hashing (Two Sum … Longest Consecutive Sequence), two pointers, sliding window, prefix sums,
binary search (including "binary search on the answer"), linked lists, recursion on trees (inorder,
Same/Symmetric/Balanced Tree), backtracking (Subsets, Permutations, Combination Sum, Generate
Parentheses), stacks, and LRU Cache. The warmup drills follow them. Once each one has been tried, the
app moves to the Google problems; Foundations you missed or copied come back through the re-solve queue.
32 of the 36 were asked at Google in the last 30 days, the rest in the last 3 months (see below).

## Arrays & Strings (25 frequent Google problems)

`data/google_frequent.json`: problems from Google's last-30-days list in
[snehasishroy/leetcode-companywise-interview-questions](https://github.com/snehasishroy/leetcode-companywise-interview-questions)
(snapshot 12 Jul 2026) that the bank did not have: Palindrome Number, Add Two Numbers, Majority Element,
Maximum Subarray, Spiral Matrix, Rotate Image, Jump Game, Next Permutation, Median of Two Sorted Arrays,
Minimum Window Substring and more. They join the weakest-topic rotation as their own topic. SQL and
JavaScript-only questions from that list are left out (not asked in SWE coding rounds).

## Daily pace

Settings › **Daily pace**, or `google-prep-cli setup --daily 3`:
- **Part-time** (default): morning warmup + 1 problem in the evening.
- **Full-time**: warmup + 3 problems a day, any time of day; each one is a new problem (a miss goes to
  the re-solve queue instead of coming straight back). Today shows "problem 2 of 3".

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

Arrays/hashing, two pointers, sliding window, binary search, linked lists and LRU Cache are in the
Foundations set above. Still not covered: harder design problems (e.g. Minimum Window Substring).

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

## Interview habit training (v1.4.0)

Today now connects your next study session to one small interview habit. Set an intention before practice, then save an honest reflection afterward. The six rotating habits are clarification, explaining your plan, recovering when stuck, hand tracing, adapting to follow-ups, and reviewing your work. Press **h** to open the habit screen.

The last-seven-days view uses your saved grades and reflections. Missing observations stay missing. Confidence is an optional self-report; the practice estimate includes its sample counts and is not an offer prediction.

The whiteboard hides global navigation for focus. Smaller windows use **Problem**, **Code**, and **Dry-run trace** tabs. **Regroup** provides a short reasoning prompt without stopping the clock or revealing a solution.

Autosave keeps pending edits until the corresponding save succeeds, serializes requests, and stores a session-specific local recovery draft when browser storage is available. Keep the window open when a connection warning appears. This release does not diagnose the previously reported background-server shutdown.

Verification:

```sh
python -m pytest -q
node --test tests/frontend-autosave.test.cjs
```

## Automatic Windows updates

Version 1.6 adds updates to **Settings → App updates**. Install this version manually once using `GoogleSWEPrep-Setup.exe`; older versions cannot update themselves. The installed Windows x64 app checks GitHub in the background, downloads a tested main-branch release and verifies its size and SHA-256 checksum. It installs after the owned app window closes, or on the next safe launch. It never interrupts a study session, kills another instance or restarts Windows.

Turn **Automatic updates** off to stop automatic downloads and installation. **Check now** still offers a manual download. Source runs, portable executables, macOS and Linux use manual downloads. Offline checks retry quietly; studying does not depend on GitHub. A successful CI build on a development branch does not publish an update.

Progress stays in your existing `.google_prep` folder (or `GOOGLE_PREP_HOME`). The helper makes a verified SQLite backup under `updates/backups` before upgrading. Failed installers retain their log and downloaded installer under `updates`; check again or run a fresh Setup manually. This preserves a backup, but does not promise automatic rollback of a partially installed program. Windows may require a restart; the updater reports that rather than restarting your computer.
