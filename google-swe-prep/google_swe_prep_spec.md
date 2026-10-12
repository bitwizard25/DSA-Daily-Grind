# Product Requirements Document (PRD) & Architecture Plan: Google SWE Interview OS (`google-swe-prep`)

> **Instruction for Claude Code**: This document serves as the complete Product Requirement Document (PRD), Technical Specification, and UI/UX Design Guide. Scaffold, implement, test, document, and **package into a standalone executable (`.exe` on Windows / binary on macOS/Linux)** following the modular phases in Section 7, packaging instructions in Section 8, and UI/UX design specifications in Section 9.

## 1. Executive Summary & Problem Discovery (The PM Lens)

### The User Problem

* **User Profile:** Early-career Backend / Distributed Systems Engineer balancing a full-time job while preparing for Google SWE II (L3).

* **The "Zero Confidence" Trap:** The candidate previously solved hundreds of algorithmic problems, but due to time away, experiences acute retention anxiety: *"I feel like I forgot everything and I'm starting from scratch."*

* **The Daily Schedule Collision:** The user does not fail because of lack of intellect; they fail because daily work exhaustion, context-switching, and unpredictable work hours break study streaks.

* **The Disconnect:** Long-term goals (passing a Google interview 8 weeks away) feel abstract, leading to procrastination or panic-cramming without structural progression.

### The Product Vision

`google-swe-prep` is an **offline, 100% local, standalone laptop application** (packaged as a single `.exe` / binary executable). It is an **Autonomous Interview OS & Accountability Coach** that bridges **Day-to-Day Time-Boxing** with the **8-Week Google Offer North Star**, systematically reactivating dormant algorithmic intuition while keeping the engineer disciplined regardless of work fatigue.

## 2. Core Product Pillars & User Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NORTH STAR: GOOGLE SWE II OFFER                       │
│           (Readiness Index: Pattern Mastery + Mock Speed + Googlyness)      │
└──────────────────────────────────────▲──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │  PHASE 3: ONSITE & GOOGLYNESS LOOP  │ (Weeks 7–8)
                    ├─────────────────────────────────────┤
                    │  PHASE 2: GOOGLE HIGH-FREQ & HARD   │ (Weeks 4–6)
                    ├─────────────────────────────────────┤
                    │  PHASE 1: FOUNDATION & RECALL SPRINT│ (Weeks 1–3)
                    └──────────────────▲──────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                    DAILY TIME-BOXING & SCHEDULE ENGINE                      │
│   Morning Warmup (20m) ──► Work Day ──► Evening Deep Mock/Solve (60-90m)   │
│   Adaptive Fallback: Low-Energy Mode for Overtime / Busy Workdays           │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Pillar 1: Day-to-Day Schedule & Energy-Adaptive Timeboxing

* **Morning Anchor (20–30 min):** Low-friction cognitive warmup (flashcards, pattern recognition drill, mental dry-run) before work starts.

* **Evening Anchor (60–90 min):** Deep problem solving or timed 45-minute mock simulation.

* **Energy-Adaptive Fallback ("Never Zero" Protocol):** When a user logs a stressful 12-hour workday, the application automatically toggles to **Emergency Micro-Mode** (1 pattern review + 1 bug-hunt drill in 15 minutes) to protect the habit loop without inducing burnout.

### Pillar 2: Fast-Recall Engine (Reactivating Intuition)

* Shows a problem statement for 60 seconds with title and tags hidden.

* Prompts: *"What is the core pattern? What is the optimal time complexity? What edge cases to clarify?"*

* Restores problem-solving reflexes and rebuilds confidence quickly.

### Pillar 3: North Star Milestone Roadmap & Readiness Index

* Translates the 8-week hiring timeline into a quantifiable **Readiness Score (0–100%)**: 

  $$
  \text{Readiness Index} = 0.40(\text{Pattern Mastery}) + 0.30(\text{Mock Speed}) + 0.15(\text{Dry-Run Accuracy}) + 0.15(\text{STAR Stories})
  $$

### Pillar 4: Google-Specific Simulation

* Terminal-based plain-text scratchpad with **no compiler and no autocomplete** (simulating Google's interview environment).

* Dynamic Google follow-up generator ($O(N)$ to streaming, memory constraints, scale).

* Built-in Googlyness STAR story manager.

## 3. Technology Stack & Packaging

* **Core Runtime:** Python 3.11+ (bundled internally into binary)

* **CLI Framework:** `typer` + `rich` (tables, progress bars, panels, syntax formatting)

* **Interactive Terminal UI:** `textual` (full-screen terminal dashboard for daily standup & schedule)

* **Data Layer:** Local SQLite (`prep.db`) + `sqlmodel` / `pydantic` v2

* **Offline Packaging:** `pyinstaller` (packages entire Python interpreter, libraries, problem JSONs, and SQLite into a single standalone `.exe` on Windows or Unix binary on macOS/Linux)

## 4. Repository Structure

```
google-swe-prep/
├── README.md
├── pyproject.toml
├── requirements.txt
├── build.py                     # Automated PyInstaller packaging script
├── data/
│   ├── problems.json            # 75 Curated Google high-frequency questions with follow-ups
│   ├── patterns.json            # 15 Core patterns with templates & visual cues
│   ├── recall_drills.json       # 50 Rapid-fire pattern recognition flash-challenges
│   └── googlyness_prompts.json   # Situational behavioral prompts for Google
├── patterns_vault/              # Reference markdown templates
│   ├── 01_two_pointers.md
│   ├── 02_sliding_window.md
│   ├── 03_graphs_bfs_dfs.md
│   ├── 04_topological_sort.md
│   ├── 05_union_find.md
│   ├── 06_heaps_priority_queue.md
│   ├── 07_binary_search_range.md
│   ├── 08_trie_prefix_trees.md
│   └── 09_dynamic_programming.md
├── google_prep/
│   ├── __init__.py
│   ├── cli.py                   # Main Typer CLI router
│   ├── db.py                    # SQLite engine & database migrations
│   ├── models.py                # Pydantic & SQLModel schema models
│   ├── core/
│   │   ├── scheduler.py         # Daily schedule manager & calendar timeboxer
│   │   ├── sm2.py               # Spaced repetition engine for retention
│   │   ├── timer.py             # 45-minute Google interview countdown timer
│   │   └── readiness.py         # 0-100% readiness calculation engine
│   ├── modules/
│   │   ├── today.py             # Daily command center (morning / evening / micro modes)
│   │   ├── recall.py            # 60-second pattern identification drill
│   │   ├── mock.py              # 45-min simulated Google whiteboard round
│   │   ├── roadmap.py           # Long-term 8-week phase tracker & milestones
│   │   └── star.py              # Googlyness behavioral story builder
│   └── ui/
│       ├── theme.py             # Google-inspired color palette & typography
│       ├── formatters.py        # Rich terminal styling, cards, and ASCII banners
│       └── tui_app.py           # Textual interactive full-screen dashboard
└── tests/
    ├── test_scheduler.py
    ├── test_sm2.py
    └── test_readiness.py

```

## 5. Feature Breakdown & Command Reference

### Feature 1: `prep today` — The Daily Command Center

* **Title / Command:** `google-prep today [--mode normal|micro|review]`

* **Purpose:** Removes daily decision fatigue by automatically diagnosing the time of day and serving the right cognitive task.

* **How It Works:**

  * **Morning Mode (< 12:00 PM):** Launches a 20-minute **Morning Warmup** (1 rapid pattern recognition drill + review of yesterday's code).

  * **Evening Mode (>= 5:00 PM):** Launches **Evening Deep Work** (1 targeted Google problem or 1 timed 45-minute mock).

  * **Emergency Micro-Mode (`--mode micro`):** Triggers when the user is exhausted from overtime work. Assigns a single 10-minute mental exercise to preserve the habit streak.

### Feature 2: `prep recall` — The Confidence-Rebuilding Engine

* **Title / Command:** `google-prep recall`

* **Purpose:** Overcomes the *"I forgot everything"* fear by building instant pattern-spotting reflexes.

* **How It Works:**

  * Displays an unseen problem statement with the title, tags, and difficulty masked.

  * Starts a 60-second timer prompting:

    1. *What is the primary pattern/data structure?*

    2. *What is the target time & space complexity?*

    3. *What is the key edge case to clarify?*

  * Reveals the optimal pattern and lets the user self-grade (1–5). Updates the local SM-2 spaced repetition queue.

### Feature 3: `prep mock` — The 45-Minute Google Whiteboard Simulator

* **Title / Command:** `google-prep mock [--topic graphs|trees|dp|all]`

* **Purpose:** Simulates Google's exact interview conditions so there are zero surprises during the live interview.

* **How It Works:**

  * **00:00–05:00:** Problem scoping & constraint clarification.

  * **05:00–35:00:** Launches a local plain-text scratchpad with **no compiler and no autocomplete** (simulating Google Docs).

  * **35:00–40:00:** Surfaces a dynamic **Google Follow-Up Constraint** (e.g., *"What if the input stream has* $10^9$ *events?"*, *"What if memory is constrained to 64MB?"*).

  * **40:00–45:00:** Interactive 4-Pillar Google Rubric self-evaluation.

### Feature 4: `prep roadmap` — The 8-Week North Star Tracker

* **Title / Command:** `google-prep roadmap`

* **Purpose:** Bridges today's 45 minutes of work with the distant goal of a Google offer.

* **How It Works:**

  * Tracks progress across 3 distinct preparation phases:

    * *Phase 1: Foundation & Screening Sprint* (Weeks 1–3)

    * *Phase 2: Google Core Patterns & Hard Follow-ups* (Weeks 4–6)

    * *Phase 3: Final Onsite Loop & Googlyness* (Weeks 7–8)

  * Displays the live **Readiness Score (**$0–100\%$**)** across all 8 problem categories.

### Feature 5: `prep star` — The Googlyness & Behavioral Vault

* **Title / Command:** `google-prep star {add|list|review}`

* **Purpose:** Ensures the non-technical round is locked down using real engineering experiences.

* **How It Works:**

  * Prompts you to draft structured STAR stories mapped to Google's evaluation dimensions:

    * *Thriving in Ambiguity*

    * *Ownership & Bias for Action*

    * *Collaboration & Navigating Disagreement*

    * *Technical Humility & Continuous Learning*

## 6. Database Schema Specifications (`models.py`)

```
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class Problem(SQLModel, table=True):
  id: str = Field(primary_key=True)
  title: str
  leetcode_id: int
  difficulty: str  # Easy, Medium, Hard
  category: str  # Graphs, DP, Trees, Intervals, etc.
  key_pattern: str
  time_target: str
  space_target: str
  google_frequency: str  # High, Very High
  follow_up_prompt: str


class UserProgress(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  problem_id: str = Field(foreign_key="problem.id")
  status: str  # unseen, learning, mastered
  reps: int = 0
  easiness_factor: float = 2.5
  interval_days: int = 1
  next_review_date: datetime
  last_solved_at: Optional[datetime] = None


class DailySchedule(SQLModel, table=True):
  date: str = Field(primary_key=True)  # YYYY-MM-DD
  mode: str = "normal"  # normal, micro, rest
  morning_completed: bool = False
  evening_completed: bool = False
  total_minutes_spent: int = 0


class StarStory(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  dimension: str  # Ambiguity, Leadership, Ownership, Humility
  title: str
  situation: str
  task: str
  action: str
  result: str
  updated_at: datetime = Field(default_factory=datetime.utcnow)

```

## 7. Phased Implementation Roadmap for Claude Code

* **Phase 1: Project Scaffolding & Database**

* **Phase 2: Data Seeding (75 Google Questions & 50 Recall Drills)**

* **Phase 3: Daily Routine & Scheduling Modules**

* **Phase 4: Simulation & Interview Tools**

* **Phase 5: Roadmap & Readiness Analytics**

* **Phase 6: Verification & Test Suite**

## 8. Standalone Executable Packaging (`.exe` / Binary)

### Automated Build Script (`build.py`)

```
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
  print("\nSuccess! Binary generated in dist/ directory.")


if __name__ == "__main__":
  build()

```

## 9. Comprehensive UI/UX Design & Interaction Guide

The user interface is designed for **maximum clarity, zero visual clutter, and instant keyboard navigation**. The interface runs entirely in the terminal using `rich` formatting and `textual` TUI components.

### A. Design Principles & Aesthetic Guidelines

1. **Google Brand Alignment (Subtle Terminal Palette):**

   * **Primary Accent (Google Blue):** `#4285F4` (Commands, active tabs, progress bars)

   * **Success / Ready (Google Green):** `#34A853` (Mastered problems, passed rubrics, streaks)

   * **Warning / Timer (Google Yellow):** `#FBBC05` (Countdowns, due reviews, medium difficulty)

   * **Danger / Hard (Google Red):** `#EA4335` (Hard problems, time expired, missed edge cases)

   * **Surface / Background:** `#1E1E2E` (Dark Slate / Terminal Default)

   * **Text Contrast:** High contrast (`#FFFFFF` for headers, `#D9E0EE` for body text, `#6C7086` for muted hints).

2. **Cognitive Load Reduction:**

   * One screen, one purpose. Never display more than 3 primary actions at any given moment.

   * Single-key shortcuts for rapid operations (`[Space]` to flip, `[1-5]` to grade, `[q]` to quit, `[Enter]` to submit).

### B. ASCII Screen Wireframes

#### Screen 1: The Daily Standup (`google-prep today`)

```
╭─────────────────────────────── GOOGLE SWE II OS ────────────────────────────────╮
│  Phase 1: Foundation (Day 4/21)  │  Streak: 🔥 5 Days  │  Readiness: 38% [====----]│
╰─────────────────────────────────────────────────────────────────────────────────╯

  Good morning, Raj! Current Time: 07:30 AM (Target: Google Hyderabad SWE II)

  ┌─ 🌅 MORNING WARMUP [20 Min] ─────────────────────────────────────────────────┐
  │  Task: Rapid Pattern Recall Drill (Graphs / BFS)                             │
  │  Goal: Identify core pattern & time complexity in under 60 seconds           │
  │  Action: Press [Enter] to start drill, or [m] for Emergency Micro-Mode       │
  └──────────────────────────────────────────────────────────────────────────────┘

  ┌─ 🌙 EVENING DEEP WORK [Scheduled for 7:30 PM] ───────────────────────────────┐
  │  Task: Course Schedule II (Topological Sort / Cycle Detection)               │
  │  Target Complexity: O(V + E) Time, O(V + E) Space                            │
  │  Status: [Pending Evening]                                                   │
  └──────────────────────────────────────────────────────────────────────────────┘

  [Enter] Start Warmup    [m] Switch to Micro-Mode    [r] View Roadmap    [q] Exit

```

#### Screen 2: Rapid Pattern Recall Flash Drill (`google-prep recall`)

```
╭──────────────────────── PATTERN RECALL DRILL ────────────────────── ⏱ 00:42 ──╮
│ Problem ID: #G-14 (Title Hidden)                                                │
╰─────────────────────────────────────────────────────────────────────────────────╯

  "Given an array of meeting time intervals consisting of start and end times
   [[s1,e1],[s2,e2],...], find the minimum number of conference rooms required."

  ---------------------------------------------------------------------------------
  Think quickly:
  1. What is the optimal Data Structure / Pattern?
  2. What is the target Time Complexity?
  3. What is the primary Edge Case?
  ---------------------------------------------------------------------------------

  [Space] Reveal Optimal Pattern & Solution Blueprint

```

*Upon pressing Space:*

```
  ✅ OPTIMAL PATTERN: Min-Heap (Priority Queue) OR Two Pointers on Sorted Starts/Ends
  ⏱ TIME COMPLEXITY: O(N log N) Time, O(N) Space
  ⚠️ KEY EDGE CASE: Meetings ending at the exact same minute another starts (e.g. [1, 5] and [5, 10])

  How well did you recall this pattern?
  [1] Blanked out   [2] Wrong Pattern   [3] Slow / Hesitated   [4] Good   [5] Instant Reflex

```

#### Screen 3: The 45-Minute Mock Interview Whiteboard (`google-prep mock`)

```
╭── MOCK INTERVIEW: Google Coding Round 1 ────────── ⏱ 24:18 Remaining ──────────╮
│ Problem: Evaluate Division (Medium)                Focus: Graph BFS / Union-Find │
╰─────────────────────────────────────────────────────────────────────────────────╯

  [Phase 2: Live Plain-Text Implementation]
  Editing: scratchpad_evaluate_division.py (External Editor Launched)
  * Autocomplete: DISABLED
  * Compiler / Test Runner: DISABLED (Manual mental dry-run required)

  ┌─ LIVE INTERVIEW CHECKLIST ───────────────────────────────────────────────────┐
  │  [✓] Clarified constraints (N <= 10^4, non-zero divisors)                    │
  │  [✓] Stated brute force O(M * N) vs optimal Union-Find approach              │
  │  [ ] Wrote clean, modular code with descriptive variable names               │
  │  [ ] Traced test case line-by-line manually                                  │
  └──────────────────────────────────────────────────────────────────────────────┘

  [p] Pause Timer    [f] Trigger Google Follow-up Constraint    [s] Submit & Grade

```

#### Screen 4: 8-Week Readiness Roadmap (`google-prep roadmap`)

```
╭─────────────────────── 8-WEEK NORTH STAR READINESS ROADMAP ──────────────────────╮
│ Overall Readiness Score: 44% / 100%    │ Target Interview Date: Nov 30, 2026     │
╰─────────────────────────────────────────────────────────────────────────────────╯

  MILESTONE PHASES
  ─────────────────────────────────────────────────────────────────────────────────
  [====------] Phase 1: Foundation & Recall Sprint       (Days 1–21)  - IN PROGRESS
  [----------] Phase 2: Google Core Patterns & Hard Grids (Days 22–42) - UPCOMING
  [----------] Phase 3: Live Onsite Mock Loops & STAR    (Days 43–56) - UPCOMING

  TOPIC MASTERY BREAKDOWN
  ─────────────────────────────────────────────────────────────────────────────────
  Graphs & Union-Find       [========------------] 42% (6/14 solved)
  Binary Trees & BST        [============--------] 60% (6/10 solved)
  Heaps & Priority Queues   [======--------------] 30% (3/10 solved)
  Intervals & Scheduling    [==========----------] 50% (4/8 solved)
  Two Pointers & Slid. Win  [================----] 80% (8/10 solved)
  Dynamic Programming       [====----------------] 20% (2/10 solved)
  Googlyness STAR Stories   [============--------] 60% (3/5 completed)

  [b] Back to Today    [m] Start Mock    [s] Add STAR Story    [q] Exit

```

### C. Keyboard Shortcuts & User Flow State

* **Global Navigation:**

  * `t` $\rightarrow$ Open Daily Command Center (`today`)

  * `r` $\rightarrow$ Open 8-Week Roadmap (`roadmap`)

  * `m` $\rightarrow$ Launch Mock Interview (`mock`)

  * `s` $\rightarrow$ Open STAR Story Vault (`star`)

  * `q` $\rightarrow$ Safe Exit (saves all SQLite progress)

* **Zero Cognitive Overhead:** When opening the terminal, simply running `google-prep` defaults to running `google-prep today`, immediately presenting the next single action to take.