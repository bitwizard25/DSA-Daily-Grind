"""The desktop app's backend: every action the window can take, as plain JSON-in / JSON-out
methods over the same database and logic the CLI uses. Nothing here knows about HTTP."""
import json
import time
from datetime import datetime

from sqlmodel import select

from google_prep import __version__, db
from google_prep.core import clock, readiness, rubric, scheduler, session_store
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

    def __init__(self, now=clock.now):
        self.now = now

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
                "app": scheduler.APP_NAME,
                "now": t.isoformat(),
                "day": day, "week": week, "dow": dow, "plan_days": scheduler.PLAN_DAYS,
                "phase": {"number": phase, "name": scheduler.PHASES[phase][0], "short": PHASE_SHORT[phase],
                          "last_day": scheduler.PHASES[phase][2]},
                "streak": scheduler.streak(s, t.date()),
                "readiness": round(r.index, 1), "weekly_change": None if change is None else round(change, 1),
                "interview_date": scheduler.interview_date(s).isoformat(),
                "days_left": (scheduler.interview_date(s) - t.date()).days,
                "pace": pace,
                "session": self._session_summary(s),
                "data_dir": str(db.home_dir()),
                "settings": {
                    "name": scheduler.get_setting(s, "name", ""),
                    "target": scheduler.get_setting(s, "target", "Google SWE II"),
                    "start_date": scheduler.start_date(s, t.date()).isoformat(),
                    "interview_date": scheduler.interview_date(s).isoformat(),
                },
                "fasttrack": {"unfinished": fasttrack.unfinished(s),
                              "pending": fasttrack.pending_day(s, day)},
            }

    def crumbs(self, parts: list[str]) -> list[str]:
        with get_session() as s:
            return _crumbs(s, self.now().date(), *parts)

    def heartbeat(self) -> dict:
        return {"ok": True}

    # ---- today --------------------------------------------------------------------
    def today(self, mode: str = "normal") -> dict:
        t = self.now()
        with get_session() as s:
            sched = scheduler.schedule_for(s, t.date())
            if mode == "normal" and sched.mode == "micro":
                mode = "micro"
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
    def recall_start(self, mode: str = "warmup", count: int = 5, resume: bool = False) -> dict:
        t = self.now()
        with get_session() as s:
            state = session_store.load() if resume else None
            if state and state.get("kind") == "recall":
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
                                    "counts": counts, "seconds": seconds, "day": t.date().isoformat()})
            return {"mode": mode, "index": index, "counts": counts, "seconds": seconds,
                    "seconds_per_drill": recall.DRILL_SECONDS, "instant_seconds": recall.INSTANT_SECONDS,
                    "crumbs": _crumbs(s, t.date(), "Recall"),
                    "grades": [{"key": k, "label": label.split(" · ")[0], "again": label.split(" · ")[1]}
                               for k, label in recall.GRADES],
                    "drills": [{"id": d.id, "type": d.type, "prompt": d.prompt} for d in drills]}

    def recall_reveal(self, drill_id: str) -> dict:
        with get_session() as s:
            d = s.get(RecallDrill, drill_id)
            if d is None:
                raise ApiError(f"Unknown drill {drill_id}")
            out = {"label": "Bug & fix" if d.type == "bug_hunt" else "Pattern", "pattern": d.pattern,
                   "complexity": d.complexity, "edge_cases": [d.edge_case], "key_insight": None, "problem": None}
            if d.problem_id and (p := s.get(Problem, d.problem_id)):
                out.update(edge_cases=p.edge_list(), key_insight=p.key_insight,
                           problem={"id": p.id, "title": p.title, "leetcode_id": p.leetcode_id})
            return out

    def recall_grade(self, drill_id: str, choice: str, seconds: float, timed_out: bool = False) -> dict:
        t = self.now()
        q, min_days, note = recall.quality(str(choice), float(seconds), bool(timed_out))
        with get_session() as s:
            prog = scheduler.grade_drill(s, drill_id, q, t, min_days)
        counted = "2" if note else str(choice)  # a slow "Instant" is logged as Effort
        state = session_store.load()
        if state and state.get("kind") == "recall" and drill_id in state["drill_ids"]:
            state["index"] = state["drill_ids"].index(drill_id) + 1
            state["counts"][counted] = state["counts"].get(counted, 0) + 1
            state["seconds"] = state.get("seconds", 0) + float(seconds)
            session_store.save(state)
        return {"again_today": q < 3, "next_review": _iso(prog.next_review_date), "note": note, "counted_as": counted}

    def recall_finish(self, mode: str, counts: dict, seconds: float) -> dict:
        t = self.now()
        done = sum(int(v) for v in counts.values())
        minutes = max(1, round(float(seconds) / 60)) if done else 0
        with get_session() as s:
            if done:
                scheduler.mark_done(s, t.date(), "morning", minutes, mode="micro" if mode == "micro" else None)
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
                   resume: bool = False) -> dict:
        t = self.now()
        state = session_store.load() if resume else None
        if state and state.get("kind") == "mock":
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
            session_store.save({"kind": "mock", "problem_id": p.id, "minutes": minutes, "practice": bool(practice),
                                "elapsed": elapsed, "ticks": ticks, "shown": shown, "submitted": submitted})
            return {"problem": self._problem_card(p, full=True), "minutes": minutes, "practice": bool(practice),
                    "elapsed": elapsed, "ticks": ticks, "shown": shown, "submitted": submitted,
                    "follow_ups": mock.follow_ups(p), "checklist": mock.CHECKLIST,
                    "code": mock.scratchpad(p).read_text(encoding="utf-8"),
                    "trace": mock.trace_pad(p).read_text(encoding="utf-8"),
                    "phases": [{"label": label, "start": a, "end": b} for label, a, b in (
                        ("Scoping", 0, 5), ("Code", 5, 35), ("Follow-up", 35, 40), ("Scorecard", 40, 45))],
                    "crumbs": _crumbs(s, t.date(), "Solve" if practice else "Mock",
                                      scheduler.CATEGORIES[p.category], p.title)}

    def mock_save(self, problem_id: str, practice: bool, minutes: int, elapsed: float, ticks: list,
                  shown: int, code: str | None = None, trace: str | None = None, submitted: bool = False) -> dict:
        """Called on every keystroke (debounced) and every timer tick: nothing is ever lost."""
        with get_session() as s:
            p = s.get(Problem, problem_id)
            if p is None:
                raise ApiError(f"Unknown problem {problem_id}")
            if code is not None:
                mock.scratchpad(p).write_text(code, encoding="utf-8")
            if trace is not None:
                mock.trace_pad(p).write_text(trace, encoding="utf-8")
        session_store.save({"kind": "mock", "problem_id": problem_id, "minutes": int(minutes),
                            "practice": bool(practice), "elapsed": float(elapsed), "ticks": list(ticks),
                            "shown": int(shown), "submitted": bool(submitted)})
        return {"saved_at": time.time()}

    def mock_abandon(self) -> dict:
        session_store.clear()
        return {"ok": True}

    def mock_scorecard(self, problem_id: str, practice: bool, minutes: int, elapsed: float, ticks: list,
                       shown: int) -> dict:
        """40:00: the questions for the 4-pillar scorecard (or the short self-grade after a solve)."""
        self.mock_save(problem_id, practice, minutes, elapsed, ticks, shown, submitted=True)
        with get_session() as s:
            p = s.get(Problem, problem_id)
            fups = mock.follow_ups(p)
            used = max(1, int(elapsed))
            ticked = rubric.from_ticks(ticks)
            trace = mock.trace_pad(p).read_text(encoding="utf-8")
            questions = [{"field": f, "pillar": pillar, "pillar_name": rubric.PILLARS[pillar], "question": q,
                          "ticked": ticked[f]}
                         for f, pillar, q in rubric.CHECKS if not practice or f in rubric.PRACTICE_CHECKS]
            return {"practice": bool(practice), "used": used,
                    "over_time": used > int(minutes) * 60 * readiness.CODING_SHARE,
                    "outcomes": [{"key": k, "label": v, "again": {"1": "back in today's re-solve queue",
                                                                  "2": "re-solve in 3 days",
                                                                  "3": "re-solve next week"}[k]}
                                 for k, v in rubric.OUTCOMES.items()],
                    "questions": questions, "trace_rows": mock.trace_rows(trace),
                    "follow_up": fups[shown % len(fups)] if shown >= 0 else "",
                    "mistakes": [{"key": k, "label": v} for k, v in scheduler.MISTAKES.items()],
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
                   checks: dict, mistake: str | None = None, shown: int = -1) -> dict:
        t = self.now()
        with get_session() as s:
            p = s.get(Problem, problem_id)
            if p is None:
                raise ApiError(f"Unknown problem {problem_id}")
            full = rubric.from_ticks([False] * 4)
            full.update({k: bool(v) for k, v in checks.items() if k in full})
            fups = mock.follow_ups(p)
            result = mock.record(s, p, int(minutes), max(1, int(elapsed)), str(outcome), full,
                                 fups[shown % len(fups)] if shown >= 0 else "", bool(practice), mistake, t)
        session_store.clear()
        pillars = rubric.pillar_scores(result["checks"])
        again = {"1": "Back in today's re-solve queue", "2": "Re-solve in 3 days", "3": "Re-solve next week"}
        return {**result, "pillars": {rubric.PILLARS[k]: round(v * 25, 1) for k, v in pillars.items()},
                "outcome_label": rubric.OUTCOMES[result["outcome"]], "again": again[result["outcome"]],
                "mistake_label": scheduler.MISTAKES.get(result["mistake"] or ""),
                "problem": {"id": p.id, "title": p.title}}

    # ---- session recovery -----------------------------------------------------------------
    def session(self) -> dict:
        with get_session() as s:
            return {"session": self._session_summary(s)}

    def session_discard(self) -> dict:
        session_store.clear()
        return {"ok": True}

    def _session_summary(self, s) -> dict | None:
        state = session_store.load()
        if not state:
            return None
        if state.get("kind") == "recall":
            left = len(state["drill_ids"]) - state["index"]
            return {"kind": "recall", "mode": state.get("mode", "warmup"),
                    "text": f"Session restored: Pattern recall ({left} drill{'s' if left != 1 else ''} left)."}
        p = s.get(Problem, state.get("problem_id"))
        title = p.title if p else state.get("problem_id")
        if state.get("submitted"):
            text = f"Session restored: {title} (scorecard not finished)."
        else:
            left = max(0, state["minutes"] * 60 - state["elapsed"])
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
                "index": round(r.index, 1), "day": day, "plan_days": scheduler.PLAN_DAYS,
                "components": [{"name": n, "key": k, "weight": w, "value": round(getattr(r, k), 4)} for n, k, w in comps],
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
                    **{k: lesson[k] for k in ("day", "title", "subtitle", "minutes", "vault")},
                    "sections": lesson["sections"],
                    "exercises": [{**{k: e[k] for k in ("id", "topic", "prompt", "code", "options")},
                                   "result": answered.get(e["id"])}  # right/wrong so far; answers stay on the server
                                  for e in lesson["exercises"]],
                    "progress": fasttrack.progress(s, lesson),
                })
            pending = fasttrack.pending_day(s, day_no)
            return {"days": days, "current": pending or (fasttrack.unfinished(s) or [1])[0], "plan_day": day_no,
                    "crumbs": _crumbs(s, t.date(), "Fast-Track")}

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
    def settings_save(self, name: str = "", target: str = "", interview_date: str = "", start_date: str = "") -> dict:
        for label, value in (("Interview date", interview_date), ("Plan start", start_date)):
            if value:
                try:
                    datetime.fromisoformat(value)
                except ValueError as exc:
                    raise ApiError(f"{label} must be YYYY-MM-DD") from exc
        with get_session() as s:
            for key, value in (("name", name), ("target", target), ("interview_date", interview_date),
                               ("start_date", start_date)):
                if value:
                    scheduler.set_setting(s, key, value.strip())
        return self.state()

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _problem_card(p: Problem, full: bool = False) -> dict:
        out = {"id": p.id, "title": p.title, "difficulty": p.difficulty, "category": p.category,
               "category_name": scheduler.CATEGORIES[p.category], "leetcode_id": p.leetcode_id,
               "statement": p.statement, "recency": p.recency, "recency_label": scheduler.WINDOW_LABEL.get(p.recency, ""),
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
