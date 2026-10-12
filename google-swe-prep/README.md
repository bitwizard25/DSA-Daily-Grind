# google-prep: Google SWE II Interview OS

Offline coach for an 8-week Google SWE II (L3) plan. One command picks your next task, times it,
and logs the result. Spec: [`google_swe_prep_spec.md`](./google_swe_prep_spec.md).

## Get the app

- **Windows `.exe`:** GitHub → **Actions** → latest **build-exe** run → download **google-prep-Windows**.
  Unzip and run `google-prep.exe` from PowerShell or Windows Terminal.
- **macOS / Linux:** same place, artifact **google-prep-macOS** / **google-prep-Linux**.
- **Build it yourself** (any OS, Python 3.11+):

  ```
  cd google-swe-prep
  pip install -r requirements.txt
  python build.py          # -> dist/google-prep(.exe)
  ```
- **Run from source:** `pip install -e .` then `google-prep`.

## Daily use

| When | Run | What happens |
|---|---|---|
| Any time | `google-prep` | Starts the single next best action: AM → recall warmup, PM → timed solve (mock in weeks 7–8). Resumes a crashed session first. |
| Exhausted | `google-prep today --worked 11` | 10+ hour day → 15-min micro-mode (1 pattern drill + 1 bug-hunt) for the rest of the day. |
| Check plan | `google-prep today` | Today's two cards, pace, streak, readiness. |
| Weekly | `google-prep roadmap` | Readiness breakdown, 14-day trend, pace, topic mastery, mistake profile. |
| Mock | `google-prep mock --topic graphs` | 45-min round: scoping → code in a plain editor → follow-up at 35:00 → rubric. |
| STAR | `google-prep star add` / `review` / `list` | Behavioural stories for the 4 Google dimensions. |

Everything is logged automatically when a drill, solve or mock finishes. There is no manual logging:
do off-app practice through `google-prep solve <id>` so the timer captures it.

### Keys

| Screen | Keys |
|---|---|
| Recall | `Space` reveal · `1` Missed · `2` Solved with Effort · `3` Instant Reflex · `q` stop |
| Mock / solve | `1-4` tick checklist · `p` pause · `f` follow-up · `s` submit · `q` abandon |
| Today | `Enter` start · `m` micro-mode · `r` roadmap · `s` STAR vault · `q` exit |
| Roadmap | `b` back · `m` mock · `s` add STAR story · `q` exit |
| Dashboard (`google-prep tui`) | `t` today · `r` roadmap · `s` STAR · `m` mock · `Enter` start · `q` exit |

## Readiness index

`Readiness = 0.40 Pattern Mastery + 0.30 Mock Speed + 0.15 Dry-Run Accuracy + 0.15 STAR Stories`

- **Pattern Mastery:** SM-2 progress on the 75 problems (70%) and 50 recall drills (30%).
- **Mock Speed:** last 5 mocks; full marks for a correct solution by 35:00, unsolved scores 0.
- **Dry-Run Accuracy:** last 10 attempts where your hand trace matched the expected output.
- **STAR Stories:** 2 stories in each of 4 dimensions (Ambiguity, Ownership, Collaboration, Humility).

## How problems are picked

1. **Weakest topic first** across 8 topics.
2. Inside it: failed problems due for a re-solve, then unseen problems **asked at Google most recently**
   (30 days > 3 months > 6 months), Easy last.
3. All 75 bank problems appear in Google's last-6-months data (snapshot of 16 Aug 2026 from
   [liquidslr/leetcode-company-wise-problems](https://github.com/liquidslr/leetcode-company-wise-problems)).
4. `google-prep sync` refreshes that data (the only command that needs internet).
   `google-prep hot` lists trending problems you don't have; `google-prep add <slug>` imports one.

Caveat: this frequency data is what candidates *reported*, and it is noisy. Google retires leaked
questions, so use it to prioritise patterns, not to memorise answers.

## Self-grading guards

The app cannot check your code, so it limits how much you can flatter yourself:

- "Instant Reflex" only counts if you revealed the answer within 20 seconds; slower counts as effort.
- Answers revealed after the 60-second timer count as effort at best.
- After a failed or slow solve you must name the mistake. Your top mistake reshapes the next warmup
  (bugs → more bug-hunts, missed patterns → only pattern drills).

## Other commands

`solve [id]` timed practice · `problems` bank status · `patterns [n]` cheat sheet and vault notes ·
`setup --name --target --interview-date` · `remind` prints OS scheduler commands for 07:30 / 19:30 ·
`resume` continue a crashed session.

## Files

- Data: `~/.google_prep/prep.db` (override with `GOOGLE_PREP_HOME`).
- Auto-saved session: `~/.google_prep/sessions/active.json` (removed when a session finishes or you quit with `q`).
- Scratchpads: `~/.google_prep/scratchpads/`. `GOOGLE_PREP_EDITOR` sets the editor (default: Notepad / TextEdit).

## Tests

```
pip install -r requirements.txt
python -m pytest
```
