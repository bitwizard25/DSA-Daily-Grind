# Master PRD & Technical Architecture: Google Interview OS (`google-swe-prep`)
### Designed by the Google Engineering Hiring & Candidate Experience Product Team

> **Instruction for Claude Code**: This document represents the definitive Product Requirement Document (PRD), Technical Specification, and UI/UX Design Guide. Build the repository as a **Hybrid UI + CLI Application** sharing a single local SQLite database (`~/.google_prep/prep.db`). Package into a standalone executable (`dist/google-prep.exe` on Windows / `dist/google-prep` on macOS/Linux) as specified in Section 8.

---

## 1. Executive Summary & Problem Discovery (The Google PM Lens)

### The Problem Space
Every year, thousands of capable software engineers—including those who have solved 600+ LeetCode problems—fail the Google SWE II (L3/L4) hiring loop. At Google, Hiring Committees (HC) reject candidates not because they lack raw intelligence, but because of three structural preparation failures:

1. **Compiler Addiction:** Candidates practice on platforms with instant "Run Code" feedback. During real Google interviews (conducted in a Google Doc / non-compiling plain-text sandbox), they freeze when required to mentally execute code line-by-line and trace edge cases.
2. **Blocked vs. Interleaved Practice:** Solving 20 graph problems in a row creates the "illusion of competence." In an interview, problems arrive without topic labels. Candidates waste 10–15 minutes trying to identify the underlying pattern.
3. **The Follow-Up Blindspot:** Google interviewers never stop at the base solution. The real hiring signal is generated in the final 10 minutes when the interviewer alters constraints: *"What if memory is capped at 64MB?", "What if this is a distributed stream of $10^7$ QPS?"*

### The Product Vision
`google-swe-prep` is an **Internal-Calibrated Google Interview Simulator & Accountability OS**. It unifies a **keyboard-first CLI** for daily habit retention with a **Desktop GUI** for full 45-minute interview simulations, powered by Google's actual 4-pillar evaluation rubric and distributed systems follow-up engine.

---

## 2. Core Architecture: The "UI + CLI Combo"

The system runs 100% locally and offline from a unified SQLite engine (`~/.google_prep/prep.db`):

```text
                                  ┌──────────────────────────────┐
                                  │      LOCAL SQLITE ENGINE     │
                                  │    (~/.google_prep/prep.db)  │
                                  └──────────────▲───────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                                                 │
          ┌─────────────┴──────────────┐                   ┌──────────────┴─────────────┐
          │     KEYBOARD-FIRST CLI     │                   │     RICH DESKTOP GUI / TUI │
          │      (`google-prep`)       │                   │    (`google-prep ui`)      │
          ├────────────────────────────┤                   ├────────────────────────────┤
          │ • Instant 60s Pattern Drills│                   │ • 45-Min Split-Screen Mock │
          │ • Zero-Argument Daily Action│                   │ • Visual 8-Week Roadmap    │
          │ • Emergency Micro-Mode     │                   │ • STAR Behavioral Studio   │
          │ • Fast Terminal Logging    │                   │ • Variable Trace Sandbox   │
          └────────────────────────────┘                   └────────────────────────────┘
```

* **CLI (`google-prep`):** Designed for low-friction daily execution. Zero-argument launch automatically starts your 20-minute Morning Warmup or 10-minute Emergency Micro-Mode.
* **Desktop UI (`google-prep ui`):** Full-screen interview cockpit featuring side-by-side problem description, plain-text editor, phased countdown timer, and live dry-run variable tracing tables.

---

## 3. Google-Calibrated Evaluation Pillars & Interview Rubric

The application evaluates candidates using the exact 4 scoring dimensions utilized by Google Hiring Committees:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GOOGLE HIRING COMMITTEE SCORING MATRIX                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Analytical Ability (25%)  │ Proactive constraint clarification (N, range)│
│                              │ Identifies edge cases before coding          │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ 2. Coding & Craft (25%)      │ Idiomatic, modular code (clean variable names)│
│                              │ Proper abstraction without spaghetti logic   │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ 3. Verification & Rigor (25%)│ Line-by-line manual test case tracing        │
│                              │ Catches own bugs without compiler feedback   │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ 4. System Scaling (25%)      │ Handles dynamic Google follow-up constraints │
│                              │ Trade-offs: Caching, streaming, sharding     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Role-Specific Focus: AI/ML & Distributed Systems (SWE II / L3)

The curated 75-problem bank is explicitly weighted toward backend, distributed systems, and AI infrastructure:

1. **Graphs & DAG Execution (35%):** Course Schedule I/II (Kahn's Algo), Evaluate Division (Weighted Graph/Disjoint Set), Network Delay Time (Dijkstra), Alien Dictionary.
2. **Concurrency, Queues & Intervals (25%):** Meeting Rooms II (Min-Heap / Sweep-Line), Merge Intervals, Sliding Window Maximum (Monotonic Deque), Task Scheduler.
3. **Trees, Tries & Hierarchical Retrieval (20%):** Lowest Common Ancestor (Post-Order DFS & Binary Lifting), Word Search II (Trie + DFS), Binary Tree Maximum Path Sum.
4. **Dynamic Programming & Sequence Search (20%):** Word Break I/II, Longest Increasing Subsequence, Trapping Rain Water (Two Pointers).

---

## 5. Feature Specifications & User Experience

### Feature 1: `prep today` — The Daily Command Center
* **CLI:** `google-prep today [--mode normal|micro]`
* **Zero-Argument Default:** Running `google-prep` without flags launches today's priority task automatically.
* **Routine Engine:**
  * **Morning (< 12:00 PM):** 20-minute Morning Warmup (1 rapid pattern drill + 1 review of yesterday's code).
  * **Evening (>= 5:00 PM):** 60–90 minute Evening Deep Work (1 targeted Google problem or 1 timed mock).
  * **Emergency Micro-Mode (`--mode micro`):** A 10-minute mental exercise to preserve streaks when exhausted from overtime work.

### Feature 2: `prep recall` — The Confidence-Rebuilding Engine
* **CLI:** `google-prep recall`
* **Mechanism:**
  * Displays an unseen problem statement with title and tags hidden.
  * 60-second countdown prompts:
    1. *What is the primary pattern/data structure?*
    2. *What is the optimal time & space complexity?*
    3. *What is the primary edge case to clarify?*
  * Unambiguous 3-button grading (Krug Law #2):
    * `[1] Missed it` *(Rescheduled today)*
    * `[2] Solved with Effort` *(Rescheduled in 3 days)*
    * `[3] Instant Reflex` *(Rescheduled next week)*

### Feature 3: `prep mock` — 45-Minute Google Whiteboard Simulator
* **CLI / UI:** `google-prep mock [--topic graphs|trees|intervals|dp|all]`
* **Phased Countdown:**
  * **00:00–05:00:** Problem scoping & constraint clarification checklist.
  * **05:00–35:00:** Plain-text implementation (**syntax checking and autocomplete disabled**).
  * **35:00–40:00:** **Dynamic Google Follow-Up Constraint** injected (e.g., *"What if queries stream at $10^7$ events/sec?"*).
  * **40:00–45:00:** 4-Pillar Google Rubric self-evaluation scorecard.

### Feature 4: `prep roadmap` — The 8-Week North Star Tracker
* **CLI / UI:** `google-prep roadmap`
* Visualizes progress across 3 Phases and computes the real-time **Readiness Index (0–100%)**:
  $$\text{Readiness Index} = 0.40(\text{Pattern Mastery}) + 0.30(\text{Mock Speed}) + 0.15(\text{Dry-Run Accuracy}) + 0.15(\text{STAR Stories})$$

### Feature 5: `prep star` — The Googlyness & Behavioral Studio
* **CLI / UI:** `google-prep star {add|list|review}`
* Structured builder mapping real engineering experiences (Hapminds, NNIIT, Projects) to Google's 4 dimensions:
  * *Ambiguity:* Leading ambiguous architecture decisions
  * *Ownership:* Handling production outages and bug fixes
  * *Disagreement:* Navigating technical architectural debates
  * *Humility:* Learning from technical miscalculations

---

## 6. Database Schema Specifications (`models.py`)

```python
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class Problem(SQLModel, table=True):
  id: str = Field(primary_key=True)
  title: str
  leetcode_id: int
  difficulty: str  # Medium, Hard
  category: str  # Graphs, Trees, Intervals, DP, etc.
  key_pattern: str
  time_target: str
  space_target: str
  google_frequency: str  # High, Very High
  key_insight: str
  edge_cases: str  # JSON-serialized string
  google_follow_up: str


class UserProgress(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  problem_id: str = Field(foreign_key="problem.id", index=True)
  status: str = "unseen"  # unseen, learning, mastered
  reps: int = 0
  easiness_factor: float = 2.5
  interval_days: int = 1
  next_review_date: datetime = Field(default_factory=datetime.utcnow)
  last_solved_at: Optional[datetime] = None


class DailySchedule(SQLModel, table=True):
  date_str: str = Field(primary_key=True)  # YYYY-MM-DD
  mode: str = "normal"  # normal, micro, rest
  morning_completed: bool = False
  evening_completed: bool = False
  total_minutes: int = 0


class MockSession(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  problem_id: str
  date: datetime = Field(default_factory=datetime.utcnow)
  duration_seconds: int
  clarified_constraints: bool
  stated_complexity: bool
  clean_code: bool
  dry_run_passed: bool
  follow_up_handled: bool
  score: float


class StarStory(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  dimension: str  # Ambiguity, Leadership, Ownership, Humility
  title: str
  situation: str
  task: str
  action: str
  result: str
  created_at: datetime = Field(default_factory=datetime.utcnow)
```

---

## 7. Implementation Roadmap for Claude Code

* **Phase 1: Shared Core & Storage**
  * Set up `pyproject.toml` with `typer`, `rich`, `textual`, `sqlmodel`, `pydantic`.
  * Create `google_prep/models.py` and `google_prep/db.py` with automatic SQLite initialization.
* **Phase 2: Data Seeding**
  * Seed `data/problems.json` with 75 curated Google questions covering Graphs, Trees, Heaps, Intervals, DP, and Monotonic Queues.
  * Seed `data/recall_drills.json` with 50 quick pattern recognition prompts.
* **Phase 3: Daily Routine & Scheduling Modules**
  * Implement `google_prep/core/scheduler.py` and `google_prep/modules/today.py` (with Normal and Micro modes).
  * Implement `google_prep/core/sm2.py` for spaced repetition calculation.
* **Phase 4: Simulation & Interview Tools**
  * Implement `google_prep/modules/recall.py` (60s countdown flash-challenges).
  * Implement `google_prep/modules/mock.py` (Plain-text editor launcher + Google 4-pillar rubric).
  * Implement `google_prep/modules/star.py` (Interactive STAR story manager).
* **Phase 5: Desktop TUI Application**
  * Implement `google_prep/ui/tui_app.py` using `textual` for the full-screen split-screen interview whiteboard.
* **Phase 6: Standalone Executable Packaging**
  * Implement `build.py` with PyInstaller to compile `dist/google-prep.exe`.

---

## 8. Standalone Executable Packaging (`build.py`)

```python
import platform
import subprocess
import sys


def build():
  print(f"Building google-swe-prep for {platform.system()}...")
  cmd = [
      sys.executable,
      "-m",
      "PyInstaller",
      "--name=google-prep",
      "--onefile",
      "--add-data=data:data",
      "--add-data=patterns_vault:patterns_vault",
      "--clean",
      "google_prep/cli.py",
  ]
  subprocess.run(cmd, check=True)
  print("\nSuccess! Standalone binary generated in dist/ directory.")


if __name__ == "__main__":
  build()
```

---

## 9. Comprehensive UI/UX Design & Interaction Guide

The user interface follows Steve Krug’s usability principles from *Don’t Make Me Think, Revisited*:
1. **Google Theme:** Primary Blue (`#4285F4`), Green (`#34A853`), Yellow (`#FBBC05`), Red (`#EA4335`) on dark slate (`#1E1E2E`).
2. **Persistent Breadcrumbs:** Every screen displays `[Google SWE II OS] Week 2 > Day 3 > Graphs > Course Schedule II`.
3. **Zero Cognitive Overhead:** Zero-argument launch automatically defaults to today's priority task.
4. **Auto-Save & Session Recovery:** All scratchpad code auto-saves to `~/.google_prep/sessions/` with one-key resume.
---

## 10. Usability Master Guide (Steve Krug, *Don't Make Me Think, Revisited*, Ch. 1–13)

| Chapter | Principle | Implementation in `google-swe-prep` |
|---|---|---|
| 1. Don't Make Me Think | Every screen self-evident; no "question marks" | Zero-argument launch runs today's single priority task; plain names (Mock Interview, Pattern Recall) |
| 2. How We Really Use the Web | Users scan, satisfice, muddle through | One dominant action per screen; no instruction manuals |
| 3. Billboard Design 101 | Conventions, visual hierarchy, low noise, text formatted for scanning | 4-colour Google palette on #1E1E2E; anchored headings; ≤ 2-line paragraphs; bold technical terms |
| 4. Animal, Vegetable, or Mineral? | Mindless, unambiguous choices; just-in-time guidance | `[1] Missed it` (today) · `[2] Solved with Effort` (3 days) · `[3] Instant Reflex` (next week) for recall and mocks; follow-up banner at minute 35 |
| 5. Omit Needless Words | Happy talk and instructions must die | No greetings or filler; bullets and tables |
| 6. Street Signs and Breadcrumbs | Persistent navigation, page names, breadcrumbs, the Trunk Test | `[Google SWE II OS] Phase 1 > Week 2 > Graphs > Course Schedule II` (current step bold); section bar Today · Recall · Mock · Roadmap · STAR; footer keys; `[b]` Back / `[q]` Quit |
| 7. The Big Bang Theory | Home screen answers "Where do I start?" | One hero card: `[Enter] Start Today's 20-min Warmup` |
| 8. The Farmer and the Cowman | No average user; test real workflows | Features tied to Google's hiring signals |
| 9. Usability Testing on 10 Cents a Day | Fix serious blockers first; take things away | No settings maze, no manual logging |
| 10. Mobile & Small Screens | Visible affordances; speed is usability | Bordered cards and keys; fast offline start |
| 11. Usability as Common Courtesy | Reservoir of goodwill | Auto-save on every keystroke; "Session restored: … Press [Enter] to resume."; 1-key `[m]` Micro-Mode (10 min) |
| 12. Accessibility | High contrast, keyboard only | WCAG AAA text colours (≥ 7:1); every feature on Tab / Enter / Space / Esc / 1–3 / y / n / q |
| 13. Guide for the Perplexed | No dark patterns, no false precision | Readiness shown with its exact formula and inputs |
