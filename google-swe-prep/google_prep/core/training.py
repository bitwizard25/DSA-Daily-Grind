"""Short, repeatable interview habits and honest evidence from saved practice.

The app records self-grades and reflections, not hiring probabilities or a mental
fitness score. Missing observations stay missing rather than becoming zeroes.
"""
from datetime import date, datetime, time, timedelta

from sqlmodel import Session, select

from google_prep.core import scheduler
from google_prep.models import Attempt, DailySchedule, MockSession, TrainingCheckin


HABITS = (
    {
        "id": "clarify", "title": "Clarify before coding",
        "cue": "Spend the first minute understanding the problem.",
        "practice_steps": ["Restate the goal in your own words.",
                           "Ask about input size and one ambiguous edge case.",
                           "Work through a small example before writing code."],
        "reflection_prompt": "Which question changed or confirmed your understanding?",
    },
    {
        "id": "explain", "title": "Explain your plan out loud",
        "cue": "Make your reasoning easy for another person to follow.",
        "practice_steps": ["Describe a simple approach before optimizing.",
                           "Name the data structure and why it fits.",
                           "State time and space complexity, then check your reasoning."],
        "reflection_prompt": "Where did your explanation become unclear, and how would you say it next time?",
    },
    {
        "id": "reset", "title": "Recover when you get stuck",
        "cue": "Use a short pause to choose your next useful step.",
        "practice_steps": ["Pause for 20 seconds and take a comfortable breath.",
                           "Say what you know and name the exact obstacle.",
                           "Try a smaller example, a brute-force solution, or one focused question."],
        "reflection_prompt": "When you got stuck, what did you try next? If it did not happen, say so.",
    },
    {
        "id": "dry_run", "title": "Trace before you submit",
        "cue": "Test your idea by hand rather than guessing that it works.",
        "practice_steps": ["Choose one normal case and one edge case.",
                           "Track variables line by line next to the code.",
                           "Compare the result with the expected output and explain any fix."],
        "reflection_prompt": "What did your trace catch or confirm?",
    },
    {
        "id": "follow_up", "title": "Adapt to a follow-up",
        "cue": "A changed constraint is a chance to show how you reason.",
        "practice_steps": ["Repeat the new constraint to confirm it.",
                           "Explain what breaks in your current solution.",
                           "Propose a change and describe its trade-off before editing."],
        "reflection_prompt": "Which constraint changed, and what trade-off did you explain?",
    },
    {
        "id": "reflect", "title": "Review honestly and choose one fix",
        "cue": "Turn one practice session into a clear next action.",
        "practice_steps": ["Record any hint or answer you used.",
                           "Name one specific thing that worked and one obstacle.",
                           "Choose one behavior to try in the next session."],
        "reflection_prompt": "What will you repeat, and what one thing will you change next time?",
    },
)
HABIT_BY_ID = {habit["id"]: habit for habit in HABITS}


def _checkin(row: TrainingCheckin | None) -> dict | None:
    if row is None:
        return None
    return {"date": row.date_str, "habit_id": row.habit_id,
            "intention": row.intention, "reflection": row.reflection,
            "confidence_before": row.confidence_before, "confidence_after": row.confidence_after,
            "completed": bool(row.reflection.strip()), "updated_at": row.updated_at.isoformat()}


def habit_for_day(session: Session, day: date) -> dict:
    saved = session.get(TrainingCheckin, day.isoformat())
    # Editing the plan start date must not silently change an intention already saved.
    if saved and saved.habit_id in HABIT_BY_ID:
        return HABIT_BY_ID[saved.habit_id]
    return HABITS[(scheduler.day_number(session, day) - 1) % len(HABITS)]


def _count(habit_id: str, observed: int, total: int, source: str, empty: str) -> dict:
    return {"id": habit_id, "title": HABIT_BY_ID[habit_id]["title"],
            "observed": observed if total else None, "total": total,
            "source": source, "status": "recorded" if total else "no_evidence",
            "message": f"{observed} of {total} {source}" if total else empty}


def practice_progress(session: Session, when: datetime) -> dict:
    """Activity and modest practice goals, never a hiring or mastery prediction."""
    from google_prep.modules import fasttrack

    day = when.date()
    key = day.isoformat()
    day_number = scheduler.day_number(session, day)
    schedule = session.get(DailySchedule, key) or DailySchedule(date_str=key)
    checkin = session.get(TrainingCheckin, key)
    if day_number <= 2:
        lesson = fasttrack.pending_day(session, day_number) or max(1, min(day_number, 2))
        lang_label = fasttrack.CODING_LANGUAGES[fasttrack.coding_language(session)]
        items = [{"id": "lesson", "title": f"{lang_label} lesson · Day {lesson}",
                  "done": fasttrack.is_done(session, lesson), "route": f"#/fasttrack/{lesson}",
                  "detail": "One small step at a time.", "optional": False}]
    else:
        practised = len(scheduler.attempts_on(session, day))
        goal = scheduler.daily_problems(session)
        items = [{"id": "warmup", "title": "Recall warmup", "done": schedule.morning_completed,
                  "route": "#/recall?mode=warmup",
                  "detail": "Micro-Mode completed" if schedule.mode == "micro" and schedule.morning_completed
                            else "Recall the pattern before seeing the answer.", "optional": False},
                 {"id": "coding", "title": f"Practise {goal} problem{'s' if goal != 1 else ''}",
                  "done": practised >= goal, "route": "#/mock", "current": practised, "target": goal,
                  "detail": f"{practised} of {goal} distinct problems attempted", "optional": False}]
    items.append({"id": "reflection", "title": "Quick reflection", "done": bool(checkin and checkin.reflection.strip()),
                  "route": "#/training", "detail": "Optional · one or two lines", "optional": True})
    required = [item for item in items if not item["optional"]]

    attempts = list(session.exec(select(Attempt).where(Attempt.day <= key)).all())
    mocks = list(session.exec(select(MockSession).where(
        MockSession.date < datetime.combine(day + timedelta(days=1), time.min))).all())
    checkins = list(session.exec(select(TrainingCheckin).where(TrainingCheckin.date_str <= key)).all())
    first = (day - timedelta(days=6)).isoformat()
    previous_first = (day - timedelta(days=13)).isoformat()
    current = [a for a in attempts if a.day >= first]
    previous = [a for a in attempts if previous_first <= a.day < first]

    def solo_problems(rows):
        return len({a.problem_id for a in rows if a.solved and a.help == "none"})

    current_solo, previous_solo = solo_problems(current), solo_problems(previous)
    explained = sum(m.clarified_constraints is True and m.stated_complexity is True for m in mocks)
    reflections = sum(bool(row.reflection.strip()) for row in checkins)
    milestones = [{"id": "coding", "title": "Coding", "label": "5 problems solved without help",
                   "current": solo_problems(attempts), "target": 5, "source": "Saved solve grades"},
                  {"id": "reasoning", "title": "Reasoning", "label": "Clarify and explain in 3 mocks",
                   "current": explained, "target": 3, "source": "Self-graded clarification and complexity"},
                  {"id": "habits", "title": "Habits", "label": "Reflect on 3 days",
                   "current": reflections, "target": 3, "source": "Daily reflections · self-report"}]
    for milestone in milestones:
        milestone["reached"] = milestone["current"] >= milestone["target"]
    return {"daily": {"items": items, "completed": sum(item["done"] for item in required), "total": len(required),
                      "micro_completed": schedule.mode == "micro" and schedule.morning_completed},
            "comparison": {"current_start": first, "previous_start": previous_first,
                           "current_solo": current_solo, "previous_solo": previous_solo,
                           "solo_delta": current_solo - previous_solo,
                           "current_attempts": len(current), "previous_attempts": len(previous)},
            "milestones": milestones}


def state(session: Session, when: datetime) -> dict:
    day = when.date()
    first = day - timedelta(days=6)
    day_key, first_key = day.isoformat(), first.isoformat()
    checkins = list(session.exec(select(TrainingCheckin).where(
        TrainingCheckin.date_str >= first_key, TrainingCheckin.date_str <= day_key
    ).order_by(TrainingCheckin.date_str)).all())
    attempts = list(session.exec(select(Attempt).where(
        Attempt.day >= first_key, Attempt.day <= day_key
    )).all())
    mocks = list(session.exec(select(MockSession).where(
        MockSession.date >= datetime.combine(first, time.min),
        MockSession.date < datetime.combine(day + timedelta(days=1), time.min)
    )).all())
    today_checkin = session.get(TrainingCheckin, day_key)
    clarifications = [mock for mock in mocks if mock.clarified_constraints is not None]
    explanations = [mock for mock in mocks if mock.stated_complexity is not None]
    traces = [attempt for attempt in attempts if attempt.dry_run_ok is not None]
    followups = [mock for mock in mocks if (mock.follow_up or "").strip() and mock.follow_up_handled is not None]
    reflections = [checkin for checkin in checkins if checkin.reflection.strip()]
    resets = [checkin for checkin in reflections if checkin.habit_id == "reset"]
    evidence = [
        _count("clarify", sum(bool(mock.clarified_constraints) for mock in clarifications), len(clarifications),
               "self-graded mocks marked clarified", "No clarification grades recorded in the last 7 days."),
        _count("explain", sum(bool(mock.stated_complexity) for mock in explanations), len(explanations),
               "self-graded mocks marked complexity stated", "No complexity grades recorded in the last 7 days."),
        {"id": "reset", "title": HABIT_BY_ID["reset"]["title"], "observed": len(resets), "total": None,
         "source": "self-report", "status": "self_report" if resets else "no_evidence",
         "message": f"{len(resets)} reset-habit reflection(s) saved; recovery is self-reported." if resets
                    else "Recovery is not measured by the timer. Record what you tried in a reflection."},
        _count("dry_run", sum(attempt.dry_run_ok is True for attempt in traces), len(traces),
               "attempts marked dry-run matched", "No dry-run grades recorded in the last 7 days."),
        _count("follow_up", sum(bool(mock.follow_up_handled) for mock in followups), len(followups),
               "shown follow-ups marked handled", "No shown follow-up graded in the last 7 days."),
        {"id": "reflect", "title": HABIT_BY_ID["reflect"]["title"], "observed": len(reflections),
         "total": len(checkins), "source": "self-report", "status": "self_report" if reflections else "no_evidence",
         "message": f"{len(reflections)} daily reflection(s) saved." if reflections
                    else "No daily reflections saved in the last 7 days."},
    ]
    days = []
    checkin_by_day = {row.date_str: row for row in checkins}
    for offset in range(7):
        key = (first + timedelta(days=offset)).isoformat()
        days.append({"date": key, "attempts": sum(attempt.day == key for attempt in attempts),
                     "mocks": sum(mock.date.date().isoformat() == key for mock in mocks),
                     "checkin": _checkin(checkin_by_day.get(key))})
    return {"date": day_key, "day": scheduler.day_number(session, day),
            "today": {"habit": habit_for_day(session, day), "checkin": _checkin(today_checkin)},
            "habits": list(HABITS), "confidence_label": "Optional confidence self-report (1-5)",
            "progress": practice_progress(session, when),
            "week": {"start": first_key, "end": day_key, "checkins": len(checkins),
                     "completed_checkins": len(reflections), "practice_attempts": len(attempts),
                     "practice_solves": sum(attempt.source == "solve" for attempt in attempts),
                     "mock_sessions": len(mocks), "habits": evidence, "days": days,
                     "message": "Saved practice and self-reports from the last 7 days. These are not hiring predictions."}}


def _text(value: str | None, name: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name.capitalize()} must be text.")
    value = value.strip()
    if not value:
        raise ValueError(f"Write a short {name} before saving.")
    if len(value) > limit:
        raise ValueError(f"Keep the {name} to {limit} characters or fewer.")
    return value


def _confidence(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or not 1 <= value <= 5):
        raise ValueError("Confidence is an optional whole number from 1 to 5.")
    return value


def save_checkin(session: Session, when: datetime, date_str: str, habit_id: str,
                 intention: str | None = None, reflection: str | None = None,
                 confidence_before: int | None = None, confidence_after: int | None = None) -> None:
    try:
        parsed = date.fromisoformat(date_str)
    except (TypeError, ValueError) as exc:
        raise ValueError("Check-in date must be YYYY-MM-DD.") from exc
    if parsed.isoformat() != date_str:
        raise ValueError("Check-in date must be YYYY-MM-DD.")
    if parsed != when.date():
        raise ValueError("The day has changed. Refresh today's habit before saving your check-in.")
    if not isinstance(habit_id, str) or habit_id not in HABIT_BY_ID:
        raise ValueError("Choose a known interview habit.")
    intention = _text(intention, "intention", 500)
    reflection = _text(reflection, "reflection", 2000)
    confidence_before, confidence_after = _confidence(confidence_before), _confidence(confidence_after)
    if all(value is None for value in (intention, reflection, confidence_before, confidence_after)):
        raise ValueError("Save an intention or a reflection for today's practice.")
    existing = session.get(TrainingCheckin, date_str)
    if existing and existing.habit_id != habit_id:
        raise ValueError("Today's intention is already attached to a different habit. Refresh before saving.")
    row = existing or TrainingCheckin(date_str=date_str, habit_id=habit_id, created_at=when, updated_at=when)
    for name, value in (("intention", intention), ("reflection", reflection),
                        ("confidence_before", confidence_before), ("confidence_after", confidence_after)):
        if value is not None:
            setattr(row, name, value)
    row.updated_at = when
    session.add(row)
    session.commit()
