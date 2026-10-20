# Personal study support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Put confidence evidence, a daily starting commitment, gap recovery and practical give-up rescue into real study flows first, following the user's latest priority.

**Architecture:** A small support module persists focus blocks and rescue events alongside existing SQLModel history. A separate frontend module adds a Today panel and a focused support page; the whiteboard receives a rescue panel without changing mock timing, grading or saved code.

**Tech Stack:** Existing Python/SQLModel/SQLite and plain JavaScript/CSS, no new dependencies.

**Spec:** Private workspace outputs/Personalized-Study-Support-Design.md sections 1–3, 5–6, 8, plus the user's 20 October request to address confidence, discipline, inconsistency and giving up first.

**Accepted follow-up:** Preserve Google mode and add a separate General mode for transferable skills and a changing role/company target. Existing profiles default to Google. A mode switch preserves the Google schedule, active coding session and shared practice evidence. General routines and role targets have separate preferences.

## Global Constraints
- Preserve code, trace, timer, hints, grading and active session identity.
- Five-minute starts are distinct from ten-minute Micro-Mode and independently solved problems.
- Show self-reports as self-reports; no diagnosis or hiring prediction.
- Daily commitment and reading size are user-controlled; no silent goal changes.
- Reflections optional, at most 240 characters for a useful-learning note.
- This prioritized increment does not install notifications, publish a release or implement the separately deferred Python runner. Native reminders and the broader day/week roadmap remain the next work after support is usable.

## Review Focus
- Restarted/parallel windows must not create overlapping focus blocks or duplicate completion evidence.
- Midnight and stale pages must not overwrite a different local day.
- A failed save must retain text and must not show success.
- Timer starts, paused time and rescue clicks must not fabricate completion/solves.
- Unavailable coaching must not block the existing Today or whiteboard route.

### Task 1: Persist focus support and truthful evidence
Files: google_prep/models.py; google_prep/core/support.py; tests/test_support.py.
Interfaces: state(session, when); preferences(session, changes=None); start(session, when, day, minutes, task); transition(session, when, block_id, action, note=''); rescue(session, when, day, blocker, outcome).
- [ ] Write behavioral tests for start/restart, pause/resume elapsed, completed vs attempted evidence, useful notes, stale day, concurrent active blocks, invalid input and rescue outcomes.
- [ ] Run tests and observe missing-feature failures.
- [ ] Implement models, validation, state aggregation and idempotent transitions.
- [ ] Run focused tests and existing training tests; commit.

### Task 2: API and real Today/support UI
Files: google_prep/gui/api.py; google_prep/gui/static/support.js; support.css; index.html; app.js; tests/test_support.py; tests/frontend-support.test.cjs.
Interfaces: support_state, support_preferences, support_start, support_transition, support_rescue RPCs; Support.today(ctx), Support.page(ctx), Support.rescue(ctx, options).
- [ ] Add failing API/JS tests for persistence, action payloads and failed-save recovery.
- [ ] Implement a Today next-step panel, focus timer with pause/extend/park/complete, short optional learning note, weekly evidence, editable cue/preferences and five-minute return flow.
- [ ] Mount optional Today support without making existing route depend on it; add labeled navigation and responsive styling.
- [ ] Verify failures, refresh persistence and all existing frontend tests; commit.

### Task 3: Give-up rescue in actual practice
Files: app.js; support.js; tests/frontend-support.test.cjs; tests/frontend-flows.test.cjs.
- [ ] Add failing tests for tailored blocker actions and save-before-park behavior.
- [ ] Add a non-blocking rescue panel with specific next steps, optional two-minute retry, helpful outcome acknowledgement and safe save-and-return. Keep existing hint unlock and mock clock unchanged.
- [ ] Run Python and Node suites, inspect desktop/small-window behavior using a separate test profile and preview port; commit.

### Task 4: Verify and review
- [ ] Run full Python and Node suites, source GUI selftest and isolated browser workflows.
- [ ] Dispatch one fresh whole-change review and fix important findings with regression tests.
- [ ] Report actual code, test results and remaining native notification/day-roadmap scope; no installed upgrade claim.
