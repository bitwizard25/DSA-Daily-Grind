"""The desktop app's backend: every action the window can take, as plain JSON-in / JSON-out
methods over the same database and logic the CLI uses. Nothing here knows about HTTP."""
import json
import time
from datetime import date, datetime, timedelta

from sqlmodel import select

from google_prep import __version__, db
from google_prep.core import clock, readiness, rubric, scheduler, session_store, support, training
from google_prep.db import get_session
from google_prep.models import LessonProgress, Problem, RecallDrill, StarStory, UserProgress
from google_prep.modules import fasttrack, mock, recall, star, today

PHASE_SHORT = {0: "Python Fast-Track", 1: "Foundation", 2: "Core Patterns", 3: "Onsite Loop"}


class ApiError(Exception):
    """A user-facing problem (bad input, missing item). Shown in the window as a short message."""


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _crumbs(session, day, *parts: str) -> list[str]:
    path = scheduler.breadcrumb(session, day, *parts)
    rest = path.removeprefix(scheduler.APP_NAME).strip()
    return [scheduler.APP_NAME.strip("[]")] + rest.split(" > ")


class Api:
    """Method names are the RPC names the window calls (POST /api/<name>)."""

    def __init__(self, now=clock.now, updater=None):
        self.now = now
        self.updater = updater

    def update_status(self) -> dict:
        if self.updater is not None:
            return self.updater.status()
        return {'supported':False, 'enabled':False, 'state':'unsupported', 'version':__version__, 'build_number':0,
                'message':'Automatic installation requires the installed Windows app.',
                'release_url':'https://github.com/bitwizard25/DSA-Daily-Grind/releases'}

    def update_check(self) -> dict:
        if self.updater is None:
            raise ApiError('Updates can be checked from the desktop app. Manual downloads are on GitHub Releases.')
        return self.updater.check()

    def update_preferences(self, enabled: bool) -> dict:
        if type(enabled) is not bool:
            raise ApiError('Automatic updates must be on or off.')
        if self.updater is None:
            raise ApiError('Automatic updates require the installed Windows app.')
        try:
            return self.updater.set_enabled(enabled)
        except (OSError, ValueError) as exc:
            raise ApiError('Could not save your update preference. Try again.') from exc

    def _plain(self, s, kind: str, item_id: str, field: str, original):
        """In simple language mode, the plain-English version from data/simple_text.json (if there is one)."""
        if fasttrack.language(s) != "simple":
            return original
        try:
            texts = db.load_json("simple_text.json")
        except (OSError, ValueError):
            return original
        return texts.get(kind, {}).get(item_id, {}).get(field) or original

    # ---- app frame ----------------------------------------------------------------
    def state(self) -> dict:
        t = self.now()
        with get_session() as s:
            day = scheduler.day_number(s, t.date())
            phase = scheduler.phase_for_day(day)
            week, dow = scheduler.week_and_day(s, t.date())
            r = readiness.record_snapshot(s, t.date())
            change = readiness.weekly_change(s, t.date())
            pace = scheduler.pace(s, t.date())
            return {
                "version": __version__,
                "build_number": self.update_status().get('build_number', 0),
                "app": scheduler.APP_NAME,
                "now": t.isoformat(),
                "day": day, "week": week, "dow": dow, "plan_days": scheduler.PLAN_DAYS,
                "phase": {"number": phase, "name": scheduler.PHASES[phase][0], "short": PHASE_SHORT[phase],
                          "last_day": scheduler.PHASES[phase][2]},
                "streak": scheduler.streak(s, t.date()),
                "readiness": round(r.index, 1), "readiness_evidence": r.evidence,
                "readiness_provisional": r.is_provisional, "weekly_change": None if change is None else round(change, 1),
                "interview_date": scheduler.interview_date(s).isoformat(),
                "days_left": (scheduler.interview_date(s) - t.date()).days,
                "pace": pace,
                "session": self._session_summary(s),
                "data_dir": str(db.home_dir()),
                "settings": {
                    "preparation_mode": scheduler.get_setting(s, "preparation_mode", "google"),
                    "name": scheduler.get_setting(s, "name", ""),
                    "target": scheduler.get_setting(s, "target", "Google SWE II"),
                    "start_date": scheduler.start_date(s, t.date()).isoformat(),
                    "interview_date": scheduler.interview_date(s).isoformat(),
                    "language": fasttrack.language(s),
                    "daily_problems": scheduler.daily_problems(s),
                },
                "fasttrack": {"unfinished": fasttrack.unfinished(s),
                              "pending": fasttrack.pending_day(s, day)},
            }

    def crumbs(self, parts: list[str]) -> list[str]:
        with get_session() as s:
            return _crumbs(s, self.now().date(), *parts)

    def heartbeat(self) -> dict:
        return {"ok": True}

    # ---- interview habits ----------------------------------------------------------
    def preparation_mode(self, mode: str) -> dict:
        if not isinstance(mode, str) or mode not in ('general', 'google'):
            raise ApiError('Choose General mode or Google mode.')
        with get_session() as s:
            if mode == 'google':
                active = support.state(s, self.now())['active']
                if active and active['status'] == 'active':
                    support.transition(s, self.now(), active['id'], 'pause', active['note'])
            scheduler.set_setting(s, 'preparation_mode', mode)
        return self.state()

    def support_state(self) -> dict:
        with get_session() as s:
            return support.state(s, self.now())

    def support_draft(self, slot: str, value: dict | None) -> dict:
        with get_session() as s:
            try:
                support.draft(s, slot, value)
            except ValueError as exc:
                raise ApiError(str(exc)) from exc
        return {'ok': True}

    def support_preferences(self, changes: dict) -> dict:
        with get_session() as s:
            try:
                support.preferences(s, changes)
            except ValueError as exc:
                raise ApiError(str(exc)) from exc
        return self.support_state()

    def support_start(self, date: str, minutes: int, task: str) -> dict:
        with get_session() as s:
            try:
                support.start(s, self.now(), date, minutes, task)
            except ValueError as exc:
                raise ApiError(str(exc)) from exc
        return self.support_state()

    def support_transition(self, block_id: str, action: str, note: str = '') -> dict:
        with get_session() as s:
            try:
                support.transition(s, self.now(), block_id, action, note)
            except ValueError as exc:
                raise ApiError(str(exc)) from exc
        return self.support_state()

    def support_rescue(self, date: str, blocker: str, outcome: str, request_id: str) -> dict:
        with get_session() as s:
            try:
                support.rescue(s, self.now(), date, blocker, outcome, request_id)
            except ValueError as exc:
                raise ApiError(str(exc)) from exc
        return self.support_state()

    def support_outside(self, date: str, minutes: int, task: str, request_id: str) -> dict:
        with get_session() as s:
            try:
                support.outside(s, self.now(), date, minutes, task, request_id)
            except ValueError as exc:
                raise ApiError(str(exc)) from exc
        return self.support_state()

    def training_state(self) -> dict:
        with get_session() as s:
            return training.state(s, self.now())

    def training_checkin(self, date: str, habit_id: str, intention: str | None = None,
                         reflection: str | None = None, confidence_before: int | None = None,
                         confidence_after: int | None = None) -> dict:
        t = self.now()
        with get_session() as s:
            try:
                training.save_checkin(s, t, date, habit_id, intention, reflection,
                                      confidence_before, confidence_after)
            except ValueError as exc:
                raise ApiError(str(exc)) from exc
            return training.state(s, t)

    def design_prep(self) -> dict:
        data = db.load_json("design_prep.json")
        with get_session() as s:
            tracks = []
            for track in data["tracks"]:
                known = {item["id"] for group in track["learning"] for item in group["items"]}
                try:
                    saved = json.loads(scheduler.get_setting(s, f"design:{track['id']}:learned", "[]"))
                except (ValueError, TypeError):
                    saved = []
                learned = [key for key in saved if isinstance(key, str) and key in known] if isinstance(saved, list) else []
                learned = sorted(set(learned))
                tracks.append({**track, "learned": learned, "learned_count": len(learned), "learning_total": len(known)})
            exercises = [{**item, "notes": scheduler.get_setting(s, f"design:{item['id']}:notes", ""),
                          "practised": scheduler.get_setting(s, f"design:{item['id']}:practised", "0") == "1"}
                         for item in data["exercises"]]
            return {"tracks": tracks, "exercises": exercises, "references": data.get("references", []), "intro": data["intro"],
                    "completed": sum(item["practised"] for item in exercises), "total": len(exercises)}

    def design_learning_save(self, track_id: str, completed: list[str]) -> dict:
        data = db.load_json("design_prep.json")
        track = next((item for item in data["tracks"] if item["id"] == track_id), None)
        if track is None:
            raise ApiError("Choose a known learning track.")
        known = {item["id"] for group in track["learning"] for item in group["items"]}
        if not isinstance(completed, list) or any(not isinstance(key, str) or key not in known for key in completed):
            raise ApiError("Choose learning topics from this track's checklist.")
        with get_session() as s:
            scheduler.set_setting(s, f"design:{track_id}:learned", json.dumps(sorted(set(completed))))
        return self.design_prep()

    def design_prep_save(self, exercise_id: str, notes: str | None = None, practised: bool | None = None) -> dict:
        data = db.load_json("design_prep.json")
        if not isinstance(exercise_id, str) or exercise_id not in {item["id"] for item in data["exercises"]}:
            raise ApiError("Choose a known design exercise.")
        if notes is not None and (not isinstance(notes, str) or len(notes) > 12000):
            raise ApiError("Keep design notes to 12,000 characters or fewer.")
        if practised is not None and type(practised) is not bool:
            raise ApiError("Practice status must be true or false.")
        with get_session() as s:
            saved_notes = scheduler.get_setting(s, f"design:{exercise_id}:notes", "") if notes is None else notes
            if practised and not saved_notes.strip():
                raise ApiError("Write brief notes from your practice before marking it practised.")
            if notes is not None:
                scheduler.set_setting(s, f"design:{exercise_id}:notes", notes)
            if practised is not None:
                scheduler.set_setting(s, f"design:{exercise_id}:practised", "1" if practised else "0")
        return self.design_prep()

    # ---- today --------------------------------------------------------------------
    def today(self, mode: str = "normal") -> dict:
        t = self.now()
        with get_session() as s:
            sched = scheduler.schedule_for(s, t.date())
            if mode == "full":  # "Back to the full plan"
                mode = "normal"
            elif mode == "normal" and sched.mode == "micro" and not (sched.morning_completed or sched.evening_completed):
                mode = "micro"  # a logged 10+ hour workday, nothing done yet (same rule as the CLI)
            d = today.decide(s, t, mode)
            d["status"] = [{"tag": tag, "text": text} for tag, text in d["status"]]
            d["crumbs"] = _crumbs(s, t.date(), "Today")
            return d

    def worked(self, hours: float) -> dict:
        """Log a long workday: 10+ hours switches today to Micro-Mode (Never Zero)."""
        t = self.now()
        with get_session() as s:
            if scheduler.needs_micro(float(hours)):
                sched = scheduler.schedule_for(s, t.date())
                sched.mode = "micro"
                s.add(sched)
                s.commit()
        return self.today()

    # ---- recall drills ---------------------------------------------------------------
    def recall_start(self, mode: str = "warmup", count: int = 5, resume: bool = False, discard: bool = False) -> dict:
        t = self.now()
        with get_session() as s:
            stored = session_store.load()
            mine = bool(stored and stored.get("kind") == "recall" and (resume or (
                stored.get("mode", "warmup") == mode and stored.get("day") == t.date().isoformat())))
            if stored and not mine:
                if not discard:  # the window asks: continue the unfinished one, or discard it
                    return {"conflict": self._session_summary(s)}
                self._credit_and_clear(s, stored)
            state = stored if mine else None
            if state:
                drills = [d for d in (s.get(RecallDrill, i) for i in state["drill_ids"]) if d]
                index, counts, seconds = state["index"], state["counts"], state["seconds"]
                mode = state.get("mode", mode)
            else:
                drills = (today.plan(s, t, "micro")["micro"] if mode == "micro"
                          else scheduler.pick_warmup(s, t, int(count)))
                index, counts, seconds = 0, {"1": 0, "2": 0, "3": 0}, 0.0
            ids = [d.id for d in drills]
            if ids:
                session_store.save({"kind": "recall", "mode": mode, "drill_ids": ids, "index": index,
                                    "counts": counts, "seconds": seconds,
                                    "day": (state or {}).get("day") or t.date().isoformat(),
                                    "started": (state or {}).get("started") or t.isoformat()})
            return {"mode": mode, "index": index, "counts": counts, "seconds": seconds,
                    "seconds_per_drill": recall.DRILL_SECONDS, "instant_seconds": recall.INSTANT_SECONDS,
                    "crumbs": _crumbs(s, t.date(), "Recall"),
                    "grades": [{"key": k, "label": label.split(" · ")[0], "again": label.split(" · ")[1]}
                               for k, label in recall.GRADES],
                    "drills": [{"id": d.id, "type": d.type, "prompt": self._plain(s, "drills", d.id, "prompt", d.prompt)}
                               for d in drills]}

    def recall_reveal(self, drill_id: str) -> dict:
        with get_session() as s:
            d = s.get(RecallDrill, drill_id)
            if d is None:
                raise ApiError(f"Unknown drill {drill_id}")
            out = {"label": "Bug & fix" if d.type == "bug_hunt" else "Pattern",
                   "pattern": self._plain(s, "drills", d.id, "pattern", d.pattern), "complexity": d.complexity,
                   "edge_cases": [self._plain(s, "drills", d.id, "edge_case", d.edge_case)], "key_insight": None,
                   "problem": None}
            if d.problem_id and (p := s.get(Problem, d.problem_id)):
                out.update(edge_cases=p.edge_list(), key_insight=self._plain(s, "problems", p.id, "key_insight", p.key_insight),
                           problem={"id": p.id, "title": p.title, "leetcode_id": p.leetcode_id})
            return out

    def recall_grade(self, drill_id: str, choice: str, seconds: float, timed_out: bool = False) -> dict:
        t = self.now()
        q, min_days, note = recall.quality(str(choice), float(seconds), bool(timed_out))
        state = session_store.load()
        in_session = bool(state and state.get("kind") == "recall" and drill_id in state["drill_ids"])
        with get_session() as s:
            prog = scheduler.grade_drill(s, drill_id, q, t, min_days)
            day = self._session_day(state) if in_session else t.date()
            micro = in_session and state.get("mode") == "micro"
            scheduler.mark_done(s, day, "morning", 0, mode="micro" if micro else None)
        counted = "2" if note else str(choice)  # a slow "Instant" is logged as Effort
        if in_session:
            state["index"] = state["drill_ids"].index(drill_id) + 1
            state["counts"][counted] = state["counts"].get(counted, 0) + 1
            state["seconds"] = state.get("seconds", 0) + float(seconds)
            session_store.save(state)
        return {"again_today": q < 3, "next_review": _iso(prog.next_review_date), "note": note, "counted_as": counted}

    def recall_finish(self, mode: str, counts: dict, seconds: float) -> dict:
        t = self.now()
        done = sum(int(v) for v in counts.values())
        minutes = max(1, round(float(seconds) / 60)) if done else 0
        state = session_store.load()
        ours = bool(state and state.get("kind") == "recall")
        day = self._session_day(state) if ours else t.date()  # a session past midnight counts for the day it began
        with get_session() as s:
            if done:
                scheduler.mark_done(s, day, "morning", minutes, mode="micro" if mode == "micro" else None)
            if ours:
                session_store.clear()
            return {"done": done, "minutes": minutes, "counts": counts, "streak": scheduler.streak(s, t.date())}

    # ---- problems & mock whiteboard ---------------------------------------------------
    def problems(self) -> dict:
        with get_session() as s:
            progress = {p.problem_id: p for p in s.exec(select(UserProgress)).all()}
            overrides = scheduler.recency_overrides(s)
            rows = []
            for p in s.exec(select(Problem).order_by(Problem.category, Problem.phase, Problem.id)).all():
                window, _ = scheduler.recency(p, overrides)
                prog = progress.get(p.id)
                rows.append({"id": p.id, "title": p.title, "difficulty": p.difficulty, "category": p.category,
                             "category_name": scheduler.CATEGORIES[p.category], "recency": window,
                             "status": prog.status if prog else "unseen",
                             "due": bool(prog and prog.next_review_date <= self.now())})
            return {"categories": [{"key": k, "name": v} for k, v in scheduler.CATEGORIES.items()], "problems": rows}

    def mock_target(self, topic: str = "all") -> dict:
        t = self.now()
        with get_session() as s:
            phase = scheduler.phase_for_day(scheduler.day_number(s, t.date()))
            p = scheduler.pick_target_problem(s, max(phase, 1), t, None if topic == "all" else [topic])
            return {"problem": self._problem_card(p), "phase": phase,
                    "crumbs": _crumbs(s, t.date(), "Mock")}

    def mock_start(self, problem_id: str | None = None, practice: bool = False, minutes: int | None = None,
                   resume: bool = False, discard: bool = False) -> dict:
        t = self.now()
        stored = session_store.load()
        mine = bool(stored and stored.get("kind") == "mock" and (resume or (
            stored.get("problem_id") == problem_id and bool(stored.get("practice")) == bool(practice))))
        if stored and not mine:
            with get_session() as s:
                if not discard:  # the window asks: continue the unfinished one, or discard it
                    return {"conflict": self._session_summary(s)}
                self._credit_and_clear(s, stored)
        state = stored if mine else None
        if state:
            problem_id, practice, minutes = state["problem_id"], state["practice"], state["minutes"]
        minutes = int(minutes or (60 if practice else 45))
        with get_session() as s:
            if problem_id is None:
                phase = scheduler.phase_for_day(scheduler.day_number(s, t.date()))
                problem_id = scheduler.pick_target_problem(s, max(phase, 1), t).id
            p = s.get(Problem, problem_id)
            if p is None:
                raise ApiError(f"Unknown problem {problem_id}")
            elapsed = float(state["elapsed"]) if state else 0.0
            ticks = list(state["ticks"]) if state else [False] * len(mock.CHECKLIST)
            shown = int(state["shown"]) if state else -1
            submitted = bool(state.get("submitted")) if state else False
            started = (state or {}).get("started") or t.isoformat()
            used_hints = int((state or {}).get("hints", 0))
            session_store.save({"kind": "mock", "problem_id": p.id, "minutes": minutes, "practice": bool(practice),
                                "elapsed": elapsed, "ticks": ticks, "shown": shown, "submitted": submitted,
                                "started": started, "submitted_at": (state or {}).get("submitted_at"),
                                "hints": used_hints})
            return {"problem": self._problem_card(p, full=True), "minutes": minutes, "practice": bool(practice),
                    "session_started": started,
                    "elapsed": elapsed, "ticks": ticks, "shown": shown, "submitted": submitted,
                    "hints": self._hints(s, p)[:used_hints], "hint_unlock": mock.hint_unlock_seconds(bool(practice)),
                    "hint_total": len(mock.hints(p)), "stuck_tips": mock.STUCK_TIPS,
                    "follow_ups": self._follow_ups(s, p), "checklist": mock.CHECKLIST,
                    "code": mock.scratchpad(p).read_text(encoding="utf-8"),
                    "trace": mock.trace_pad(p).read_text(encoding="utf-8"),
                    "phases": [{"label": label, "start": a, "end": b} for label, a, b in (
                        ("Scoping", 0, 5), ("Code", 5, 35), ("Follow-up", 35, 40), ("Scorecard", 40, 45))],
                    "crumbs": _crumbs(s, t.date(), "Solve" if practice else "Mock",
                                      scheduler.CATEGORIES[p.category], p.title)}

    def mock_save(self, problem_id: str, practice: bool, minutes: int, elapsed: float, ticks: list,
                  shown: int, code: str | None = None, trace: str | None = None, submitted: bool = False,
                  hints: int = 0) -> dict:
        """Called on every keystroke (debounced) and every timer tick: nothing is ever lost."""
        with get_session() as s:
            p = s.get(Problem, problem_id)
            if p is None:
                raise ApiError(f"Unknown problem {problem_id}")
            if code is not None:
                mock.scratchpad(p).write_text(code, encoding="utf-8")
            if trace is not None:
                mock.trace_pad(p).write_text(trace, encoding="utf-8")
        prev = session_store.load() or {}
        same = prev.get("kind") == "mock" and prev.get("problem_id") == problem_id
        submitted = bool(submitted) or bool(same and prev.get("submitted"))  # a late autosave never un-submits
        submitted_at = (prev.get("submitted_at") if same else None) or (self.now().isoformat() if submitted else None)
        session_store.save({"kind": "mock", "problem_id": problem_id, "minutes": int(minutes),
                            "practice": bool(practice), "elapsed": max(0.0, float(elapsed)), "ticks": list(ticks),
                            "shown": int(shown), "submitted": submitted,
                            "started": (prev.get("started") if same else None) or self.now().isoformat(),
                            "submitted_at": submitted_at, "hints": max(int(hints), int(prev.get("hints", 0)) if same else 0)})
        return {"saved_at": time.time()}

    def mock_hint(self, problem_id: str, elapsed: float) -> dict:
        """'I'm stuck'. Before the struggle time is up: what to try first. After it: the next hint
        (the pattern's name, then the key idea), counted, so the scorecard knows a hint was used."""
        state = session_store.load()
        if not (state and state.get("kind") == "mock" and state.get("problem_id") == problem_id):
            raise ApiError("Start the round first.")
        wait = mock.hint_unlock_seconds(bool(state.get("practice"))) - max(float(elapsed), float(state.get("elapsed", 0)))
        with get_session() as s:
            p = s.get(Problem, problem_id)
            if p is None:
                raise ApiError(f"Unknown problem {problem_id}")
            levels = self._hints(s, p)
        used = int(state.get("hints", 0))
        if wait > 0:
            return {"locked": True, "wait": round(wait), "tips": mock.STUCK_TIPS, "hints": levels[:used]}
        used = min(len(levels), used + 1)
        session_store.save({**state, "hints": used})
        return {"locked": False, "hints": levels[:used], "more": used < len(levels)}

    def mock_abandon(self) -> dict:
        session_store.clear()
        return {"ok": True}

    def mock_scorecard(self, problem_id: str, practice: bool, minutes: int, elapsed: float, ticks: list,
                       shown: int) -> dict:
        """40:00: the questions for the 4-pillar scorecard (or the short self-grade after a solve)."""
        self.mock_save(problem_id, practice, minutes, elapsed, ticks, shown, submitted=True)
        with get_session() as s:
            p = s.get(Problem, problem_id)
            fups = self._follow_ups(s, p)
            used = max(1, int(elapsed))
            ticked = rubric.from_ticks(ticks)
            trace = mock.trace_pad(p).read_text(encoding="utf-8")
            questions = [{"field": f, "pillar": pillar, "pillar_name": rubric.PILLARS[pillar], "question": q,
                          "ticked": ticked[f]}
                         for f, pillar, q in rubric.CHECKS if not practice or f in rubric.PRACTICE_CHECKS]
            state = session_store.load() or {}
            return {"practice": bool(practice), "used": used,
                    "over_time": used > int(minutes) * 60 * readiness.CODING_SHARE,
                    "outcomes": [{"key": k, "label": v, "again": rubric.AGAIN[k]} for k, v in rubric.OUTCOMES.items()],
                    "help": [{"key": k, "value": v, "label": rubric.HELP[v]} for k, v in rubric.HELP_KEYS.items()],
                    "hints_used": int(state.get("hints", 0)) if state.get("problem_id") == problem_id else 0,
                    "again_answer": rubric.AGAIN_ANSWER,
                    "questions": questions, "trace_rows": mock.trace_rows(trace),
                    "follow_up": fups[shown % len(fups)] if shown >= 0 else "",
                    "mistakes": [{"key": k, "label": scheduler.MISTAKES[k]} for k in scheduler.MISTAKE_CHOICES],
                    "pillars": rubric.PILLARS,
                    "crumbs": _crumbs(s, self.now().date(), "Solve" if practice else "Mock",
                                      scheduler.CATEGORIES[p.category], p.title,
                                      "Self-grade" if practice else "Scorecard")}

    def problem_card(self, problem_id: str) -> dict:
        """The pre-start card for a problem picked from the bank table."""
        with get_session() as s:
            p = s.get(Problem, problem_id)
            if p is None:
                raise ApiError(f"Unknown problem {problem_id}")
            return {"problem": self._problem_card(p)}

    def mock_needs_mistake(self, outcome: str, dry_run_passed: bool, elapsed: float, minutes: int) -> dict:
        return {"needs": mock.needs_mistake(outcome != "1", bool(dry_run_passed), int(elapsed), int(minutes))}

    def mock_grade(self, problem_id: str, practice: bool, minutes: int, elapsed: float, outcome: str,
                   checks: dict, mistake: str | None = None, shown: int = -1, help: str = "none") -> dict:  # noqa: A002
        if help not in rubric.HELP:
            raise ApiError("Say whether you used help: none, a hint, or the answer.")
        if help == "answer":
            outcome = "1"
        if str(outcome) not in rubric.OUTCOMES:
            raise ApiError("Pick Missed it, Solved with Effort or Instant Reflex.")
        state = session_store.load()
        ours = bool(state and state.get("kind") == "mock" and state.get("problem_id") == problem_id)
        if ours and help == "none" and int(state.get("hints", 0)):
            help = "hint"  # noqa: A001  (the app's own hint was opened)
        t = self._sitting_time(state, int(minutes)) if ours else self.now()
        with get_session() as s:
            p = s.get(Problem, problem_id)
            if p is None:
                raise ApiError(f"Unknown problem {problem_id}")
            full = rubric.from_ticks([False] * 4)
            full.update({k: bool(v) for k, v in checks.items() if k in full})
            fups = mock.follow_ups(p)
            result = mock.record(s, p, int(minutes), max(1, int(elapsed)), str(outcome), full,
                                 fups[shown % len(fups)] if shown >= 0 else "", bool(practice), mistake, t, help)
        if ours or not state:
            session_store.clear()
        pillars = rubric.pillar_scores(result["checks"])
        label = "Saw the answer: not solved yet" if result["help"] == "answer" else rubric.OUTCOMES[result["outcome"]]
        return {**result, "pillars": {rubric.PILLARS[k]: round(v * 25, 1) for k, v in pillars.items()},
                "outcome_label": label, "again": result["again"][:1].upper() + result["again"][1:],
                "help_label": rubric.HELP[result["help"]],
                "mistake_label": scheduler.MISTAKES.get(result["mistake"] or ""),
                "problem": {"id": p.id, "title": p.title}}

    # ---- session recovery -----------------------------------------------------------------
    def session(self) -> dict:
        with get_session() as s:
            return {"session": self._session_summary(s)}

    def session_discard(self) -> dict:
        state = session_store.load()
        if state:
            with get_session() as s:
                self._credit_and_clear(s, state)
        return {"ok": True}

    def _credit_and_clear(self, s, state: dict) -> None:
        """Drop an unfinished session. Drills already graded still count for the day they were done."""
        if state.get("kind") == "recall" and sum(int(v) for v in state.get("counts", {}).values()):
            minutes = max(1, round(float(state.get("seconds", 0)) / 60))
            scheduler.mark_done(s, self._session_day(state), "morning", minutes,
                                mode="micro" if state.get("mode") == "micro" else None)
        session_store.clear()

    SITTING_GRACE = timedelta(hours=6)

    def _session_day(self, state: dict | None) -> date:
        """The day drills count for: the day the sitting began, if it began within the last few hours
        (so 23:50 drills finished at 00:05 count for the evening before). Work done after a long gap,
        such as a page left open overnight, counts for the day it is done."""
        now = self.now()
        try:
            started = datetime.fromisoformat(state["started"])
            return started.date() if timedelta(0) <= now - started <= self.SITTING_GRACE else now.date()
        except (KeyError, TypeError, ValueError):
            pass
        try:  # sessions saved by older versions have only the day
            return date.fromisoformat(state["day"])
        except (KeyError, TypeError, ValueError):
            return now.date()

    def _sitting_time(self, state: dict | None, minutes: int) -> datetime:
        """When a mock or solve began, so one that runs past midnight is logged on the day it started.
        The round ends at the 40:00 hand-over, so a scorecard graded the next morning still counts for the
        evening it was coded. A sitting recovered much later (a crash, then the next evening) counts for
        the day it is finished."""
        now = self.now()
        try:
            started = datetime.fromisoformat(state["started"])
        except (KeyError, TypeError, ValueError):
            return now
        try:
            end = datetime.fromisoformat(state["submitted_at"]) if state.get("submitted_at") else now
        except (TypeError, ValueError):
            end = now
        return started if timedelta(0) <= end - started <= timedelta(minutes=minutes) + self.SITTING_GRACE else now

    def _session_summary(self, s) -> dict | None:
        state = session_store.load()
        if not state:
            return None
        if state.get("kind") == "recall":
            left = len(state["drill_ids"]) - state["index"]
            return {"kind": "recall", "mode": state.get("mode", "warmup"),
                    "text": f"Session restored: Pattern recall ({left} drill{'s' if left != 1 else ''} left)."}
        p = s.get(Problem, state.get("problem_id")) if state.get("problem_id") else None
        title = p.title if p else state.get("problem_id") or "a mock"
        if state.get("submitted"):
            text = f"Session restored: {title} (scorecard not finished)."
        else:
            left = max(0, int(state.get("minutes", 45)) * 60 - float(state.get("elapsed", 0)))
            text = f"Session restored: {title} ({round(left / 60)}m remaining)."
        return {"kind": "mock", "practice": state.get("practice", False), "problem_id": state.get("problem_id"),
                "submitted": bool(state.get("submitted")), "text": text}

    # ---- roadmap ----------------------------------------------------------------------------
    def roadmap(self) -> dict:
        t = self.now()
        with get_session() as s:
            r = readiness.record_snapshot(s, t.date())
            day = scheduler.day_number(s, t.date())
            comps = [("Pattern Mastery", "pattern_mastery", 40), ("Mock Speed", "mock_speed", 30),
                     ("Dry-Run Accuracy", "dry_run", 15), ("STAR Stories", "star", 15)]
            explain = {  # what each score means, in plain words (shown under it)
                "pattern_mastery": "How much of the problem bank and the drills you have under control.",
                "mock_speed": "Last 5 mocks: solved, and coded by minute 35 of 45.",
                "dry_run": "Last 10 solves: when you traced your code by hand, did it match?",
                "star": "Behaviour stories ready: 2 for each of 4 topics.",
            }
            phases = []
            for num, (name, first, last) in scheduler.PHASES.items():
                span = last - first + 1
                phases.append({"number": num, "name": name, "first": first, "last": last,
                               "progress": min(1.0, max(0.0, (day - first + 1) / span)),
                               "state": "done" if day > last else "now" if day >= first else "upcoming"})
            counts = readiness.star_counts(s)
            mistakes = scheduler.mistake_profile(s)
            return {
                "crumbs": _crumbs(s, t.date(), "Roadmap"),
                "index": round(r.index, 1), "readiness_evidence": r.evidence,
                "readiness_provisional": r.is_provisional, "day": day, "plan_days": scheduler.PLAN_DAYS,
                "components": [{"name": n, "key": k, "weight": w, "value": round(getattr(r, k), 4), "explain": explain[k]}
                               for n, k, w in comps],
                "formula": " + ".join(f"{w / 100:.2f}×{getattr(r, k) * 100:.0f}%" for _, k, w in comps)
                           + f" = {r.index:.0f}%",
                "history": [round(v, 1) for v in readiness.history(s, t.date())],
                "weekly_change": readiness.weekly_change(s, t.date()),
                "phases": phases,
                "topics": [{"key": k, "name": scheduler.CATEGORIES[k], "score": round(v["score"], 4),
                            "solved": v["solved"], "total": v["total"]}
                           for k, v in scheduler.category_mastery(s).items()],
                "star": {"have": sum(min(c, readiness.STARS_PER_DIMENSION) for c in counts.values()),
                         "target": readiness.STARS_PER_DIMENSION * len(counts), "value": round(r.star, 4)},
                "pace": scheduler.pace(s, t.date()),
                "mistakes": [{"key": k, "label": scheduler.MISTAKES[k], "count": n} for k, n in mistakes.items()],
                "interview_date": scheduler.interview_date(s).isoformat(),
            }

    # ---- STAR studio ------------------------------------------------------------------------
    # ---- role prep ----------------------------------------------------------------------------
    def role_prep(self) -> dict:
        with get_session() as s:
            data = db.load_json("role_prep.json")
            saved = self._role_state(s)
            items = [{**x, "done": bool(saved.get(x["id"], {}).get("done")), "notes": saved.get(x["id"], {}).get("notes", "")}
                     for x in data["items"]]
            return {"crumbs": _crumbs(s, self.now().date(), "Role prep"), "intro": data["intro"], "items": items,
                    "done": sum(x["done"] for x in items), "total": len(items)}

    def role_prep_save(self, item_id: str, done: bool | None = None, notes: str | None = None) -> dict:
        if item_id not in {x["id"] for x in db.load_json("role_prep.json")["items"]}:
            raise ApiError(f"Unknown item {item_id}")
        with get_session() as s:
            saved = self._role_state(s)
            row = saved.setdefault(item_id, {})
            if done is not None:
                row["done"] = bool(done)
            if notes is not None:
                row["notes"] = str(notes)[:5000]
            scheduler.set_setting(s, "role_prep", json.dumps(saved))
        return {"ok": True, "done": sum(bool(v.get("done")) for v in saved.values())}

    @staticmethod
    def _role_state(s) -> dict:
        try:
            return dict(json.loads(scheduler.get_setting(s, "role_prep", "{}") or "{}"))
        except (TypeError, ValueError):
            return {}

    def star_list(self) -> dict:
        with get_session() as s:
            counts = readiness.star_counts(s)
            stories = s.exec(select(StarStory).order_by(StarStory.dimension, StarStory.id)).all()
            return {"crumbs": _crumbs(s, self.now().date(), "STAR"),
                    "need": readiness.STARS_PER_DIMENSION,
                    "dimensions": [{"key": k, "label": v, "count": counts[k]} for k, v in star.DIMENSIONS.items()],
                    "stories": [{"id": x.id, "dimension": x.dimension, "title": x.title, "situation": x.situation,
                                 "task": x.task, "action": x.action, "result": x.result,
                                 "created_at": _iso(x.created_at)} for x in stories]}

    def star_save(self, dimension: str, title: str, situation: str, task: str, action: str, result: str,
                  id: int | None = None) -> dict:  # noqa: A002 - matches the JSON field
        dim = star.resolve_dimension(dimension or "")
        fields = {"title": title, "situation": situation, "task": task, "action": action, "result": result}
        if dim is None or not all((v or "").strip() for v in fields.values()):
            raise ApiError("Pick a dimension and fill all five fields.")
        with get_session() as s:
            if id is not None:
                story = s.get(StarStory, int(id))
                if story is None:
                    raise ApiError("That story no longer exists.")
                story.dimension = dim
                for k, v in fields.items():
                    setattr(story, k, v.strip())
                s.add(story)
                s.commit()
            else:
                star.add(s, dim, **{k: v.strip() for k, v in fields.items()})
        return self.star_list()

    def star_delete(self, id: int) -> dict:  # noqa: A002
        with get_session() as s:
            story = s.get(StarStory, int(id))
            if story:
                s.delete(story)
                s.commit()
        return self.star_list()

    def star_practice(self, dimension: str | None = None) -> dict:
        with get_session() as s:
            dim, prompt = star.review_prompt(s, star.resolve_dimension(dimension) if dimension else None)
            return {"dimension": dim, "label": star.DIMENSIONS[dim], "prompt": prompt, "seconds": 120,
                    "stories": [{"id": x.id, "title": x.title} for x in star.stories_for(s, dim)]}

    # ---- Phase 0: C++ -> Python fast-track -------------------------------------------------
    def fasttrack(self) -> dict:
        t = self.now()
        with get_session() as s:
            day_no = scheduler.day_number(s, t.date())
            answered = {r.item_id: r.correct for r in s.exec(select(LessonProgress)).all() if r.kind == "exercise"}
            days = []
            for lesson in fasttrack.load_days():
                days.append({
                    **{k: lesson.get(k) for k in ("day", "title", "title_simple", "subtitle", "subtitle_simple",
                                                  "minutes", "vault")},
                    "glossary": lesson.get("glossary", []),
                    "sections": lesson["sections"],
                    "exercises": [{**{k: e.get(k) for k in ("id", "section", "topic", "prompt", "code", "options")},
                                   "result": answered.get(e["id"])}  # right/wrong so far; answers stay on the server
                                  for e in lesson["exercises"]],
                    "progress": fasttrack.progress(s, lesson),
                })
            pending = fasttrack.pending_day(s, day_no)
            return {"days": days, "current": pending or (fasttrack.unfinished(s) or [1])[0], "plan_day": day_no,
                    "unclear": fasttrack.unclear_steps(s), "language": fasttrack.language(s),
                    "crumbs": _crumbs(s, t.date(), "Fast-Track")}

    def lesson_unclear(self, section_id: str, unclear: bool = True) -> dict:
        """'I did not get this step yet': listed again at the end of the day and on Today."""
        known = {sec["id"] for lesson in fasttrack.load_days() for sec in lesson["sections"]}
        if section_id not in known:
            raise ApiError(f"Unknown step {section_id}")
        with get_session() as s:
            fasttrack.mark_unclear(s, section_id, bool(unclear), self.now())
            return {"unclear": fasttrack.unclear_steps(s)}

    def set_language(self, mode: str) -> dict:
        """Standard (technical English, every term explained) or simple (plain English + Hindi hints)."""
        if mode not in fasttrack.LANGUAGES:
            raise ApiError("Language must be standard or simple")
        with get_session() as s:
            scheduler.set_setting(s, "language", mode)
        return {"language": mode}

    def hints(self) -> dict:
        return {"hints": fasttrack.load_hints(), "code_hints": fasttrack.load_hints("code_hints")}

    def fasttrack_answer(self, exercise_id: str, choice: int) -> dict:
        with get_session() as s:
            try:
                return fasttrack.answer(s, exercise_id, int(choice), self.now())
            except KeyError as exc:
                raise ApiError(f"Unknown exercise {exc.args[0]}") from exc

    def fasttrack_complete(self, day: int, minutes: int = 0) -> dict:
        with get_session() as s:
            fasttrack.complete(s, int(day), self.now(), int(minutes))
            return {"ok": True, "streak": scheduler.streak(s, self.now().date())}

    # ---- concept vault --------------------------------------------------------------------
    def vault_list(self) -> dict:
        folder = db.resource_dir() / "patterns_vault"
        notes = []
        for path in sorted(folder.glob("*.md")):
            first = path.read_text(encoding="utf-8").splitlines()[0] if path.stat().st_size else path.stem
            notes.append({"file": path.name, "title": first.lstrip("# ").strip() or path.stem})
        with get_session() as s:
            return {"notes": notes, "crumbs": _crumbs(s, self.now().date(), "Vault")}

    def vault_note(self, file: str) -> dict:
        from markdown_it import MarkdownIt

        folder = (db.resource_dir() / "patterns_vault").resolve()
        path = (folder / file).resolve()
        if path.parent != folder or path.suffix != ".md" or not path.exists():
            raise ApiError("Unknown note")
        text = path.read_text(encoding="utf-8")
        html = MarkdownIt("commonmark", {"html": False}).enable("table").render(text)
        title = text.splitlines()[0].lstrip("# ").strip() if text else file
        with get_session() as s:
            return {"file": file, "title": title, "html": html,
                    "crumbs": _crumbs(s, self.now().date(), "Vault", title)}

    # ---- links --------------------------------------------------------------------------------
    def open_link(self, url: str) -> dict:
        """Open a LeetCode or reference link in the normal browser, never inside the app window."""
        if not str(url).startswith("https://"):
            raise ApiError("Only https links can be opened")
        import webbrowser

        webbrowser.open(str(url))
        return {"ok": True}

    # ---- settings -------------------------------------------------------------------------
    def settings_save(self, name: str = "", target: str = "", interview_date: str = "", start_date: str = "",
                      language: str = "", daily_problems: str = "") -> dict:
        if daily_problems and str(daily_problems) not in scheduler.DAILY_PROBLEMS:
            raise ApiError("Daily problems must be 1 or 3.")
        if language:
            self.set_language(language)
        for label, value in (("Interview date", interview_date), ("Plan start", start_date)):
            if value:
                try:
                    datetime.fromisoformat(value)
                except ValueError as exc:
                    raise ApiError(f"{label} must be YYYY-MM-DD") from exc
        with get_session() as s:
            for key, value in (("name", name), ("target", target), ("interview_date", interview_date),
                               ("start_date", start_date), ("daily_problems", str(daily_problems))):
                if value:
                    scheduler.set_setting(s, key, value.strip())
        return self.state()

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _foundations_total() -> int:
        return len(db.load_json("foundations.json")["problems"])

    def _hints(self, s, p: Problem) -> list[str]:
        return mock.hints(p, self._plain(s, "problems", p.id, "key_insight", p.key_insight))

    def _follow_ups(self, s, p: Problem) -> list[str]:
        fups = mock.follow_ups(p)
        return [self._plain(s, "problems", p.id, "follow_up", fups[0])] + fups[1:]

    def _problem_card(self, p: Problem, full: bool = False) -> dict:
        with get_session() as s:
            statement = self._plain(s, "problems", p.id, "statement", p.statement)
        label = scheduler.WINDOW_LABEL.get(p.recency, "")
        if p.category == scheduler.FOUNDATIONS:
            label = f"building block {p.sequence} of {self._foundations_total()}" + (f" · {label}" if p.recency else "")
        out = {"id": p.id, "title": p.title, "difficulty": p.difficulty, "category": p.category,
               "category_name": scheduler.CATEGORIES[p.category], "leetcode_id": p.leetcode_id,
               "statement": statement, "recency": p.recency, "recency_label": label,
               "url": f"https://leetcode.com/problems/{p.slug}/"}
        if not full:  # the pre-start card may show targets; the whiteboard hides the pattern
            out.update(time_target=p.time_target, space_target=p.space_target)
        return out


RPC = frozenset(name for name, value in vars(Api).items() if callable(value) and not name.startswith("_"))


def call(api: Api, name: str, payload: dict | None) -> dict:
    """Dispatch one RPC by name; only the public methods of Api are callable."""
    if name not in RPC:
        raise ApiError(f"Unknown action {name}")
    try:
        return getattr(api, name)(**(payload or {}))
    except TypeError as exc:  # wrong or missing fields from the window
        raise ApiError(f"Bad request for {name}: {exc}") from exc


def dumps(data) -> bytes:
    return json.dumps(data, default=str).encode("utf-8")
