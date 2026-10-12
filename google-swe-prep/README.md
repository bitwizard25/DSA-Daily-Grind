# google-prep: Google SWE II Interview OS

Offline interview simulator and accountability coach for an 8-week Google SWE II plan.
Spec: [`google_swe_prep_spec.md`](./google_swe_prep_spec.md).

Two front ends, one database (`~/.google_prep/prep.db`):

| | Command | For |
|---|---|---|
| **CLI** | `google-prep` | Daily habit: zero-argument launch, 60-second recall drills, micro-mode, streaks |
| **UI** | `google-prep ui` | Full-screen cockpit: 45-min whiteboard + scorecard, roadmap gauge, STAR Studio |

## Get the app

- **Windows `.exe`:** GitHub → **Actions** → latest green **build-exe** run → download **google-prep-Windows**.
  Run `google-prep.exe` in Windows Terminal or PowerShell.
- **macOS / Linux:** artifacts **google-prep-macOS** / **google-prep-Linux** from the same run.
- **Build it yourself** (Python 3.11+): `cd google-swe-prep`, `pip install -r requirements.txt`, `python build.py`
  → `dist/google-prep(.exe)`. `python build.py --onedir` builds a folder instead; it starts about 40%
  faster (≈0.9 s vs ≈1.5 s measured on Linux) because nothing is unpacked on each launch.
- **From source:** `pip install -e .` then `google-prep`.

## Daily use

| When | Run | What happens |
|---|---|---|
| Any time | `google-prep` | Starts the next best action: AM → recall warmup, PM → timed solve (mock in weeks 7–8), micro-mode on a heavy day. Offers to resume a crashed session first. |
| Exhausted | `m` on any screen, or `google-prep today --worked 11` | 10-minute Micro-Mode (1 pattern drill + 1 bug-hunt); counts toward the streak. |
| Mock | `google-prep mock --topic graphs`, or `w` | The 45-minute whiteboard below. |
| Weekly | `google-prep roadmap` | Circular readiness gauge, components, 14-day trend, pace, topic mastery, mistake profile. |
| STAR | `s`, or `google-prep star add/list/review` | Stories for Ambiguity, Ownership, Disagreement, Humility. |

Every screen shows the same frame: breadcrumb (`[Google SWE II OS] Phase 1 > Week 2 > Day 3 > …`, current
step in bold), the section bar `[t] Today [d] Recall [w] Mock [r] Roadmap [s] STAR`, a title, and the
keys for that screen in the footer. `b`/`Esc` goes back, `q` quits. The home screen shows one
**Next** card; press `Enter`.

Everything is logged automatically when a drill, solve or mock ends. Do off-app practice through
`google-prep solve <id>` so the timer captures it.

## The 45-minute whiteboard

Timer bar on top, problem on the left, plain-text editor on the right (no syntax highlighting, no
autocomplete, nothing runs), plus a **Trace table** tab where you step through your own code by hand.

| Time | Phase |
|---|---|
| 00:00–05:00 | Scoping: clarify constraints and edge cases (checklist F1–F2) |
| 05:00–35:00 | Code in plain text; trace a non-trivial input by hand (F6) |
| 35:00–40:00 | A Google follow-up banner drops in (streaming, memory caps, sharding) |
| 40:00–45:00 | 4-pillar scorecard: Analytical Ability, Coding & Craft, Verification & Rigor, System Scaling (25% each) |

Keys: `F1-F4` checklist · `F5` code · `F6` trace table · `Ctrl+F` another follow-up · `Ctrl+T` pause ·
`Ctrl+S` submit · `Ctrl+Q` twice to abandon.
Scorecard: first `[1] Missed it` / `[2] Solved with Effort` / `[3] Instant Reflex` (reschedules the problem
today / in 3 days / next week), then one `y`/`n` per pillar check; your checklist ticks are pre-noted, `b`
steps back. Every keystroke in the code and trace is saved to `~/.google_prep/sessions/`; after a crash,
`google-prep` opens with "Session restored: Course Schedule II (18m remaining). Press [Enter] to resume."
`--external` uses Notepad/TextEdit and a terminal timer instead.

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

`solve [id]` · `problems` · `patterns [n]` · `setup --name --target --interview-date` · `remind` (OS
scheduler commands for 07:30 / 19:30) · `resume`.

## Files

- `~/.google_prep/prep.db`: everything (override the folder with `GOOGLE_PREP_HOME`).
- `~/.google_prep/sessions/`: the auto-saved session (`active.json`) and every scratchpad and trace.
- Upgrades keep your progress: renamed fields are migrated in place, and progress on a retired problem
  moves to the same LeetCode problem's new id.

## Tests

`pip install -r requirements.txt` then `python -m pytest`.
