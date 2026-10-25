"""Code Red: the exam-condition sprint for the last 2-3 weeks before the interview.

It reuses the problem bank, drills, STAR stories and recruiter sheet, but serves only
Medium and Hard problems asked at Google in the last 3 months, and plans one fixed daily
mission. The day's plan is stored once, so it does not shift while you work through it.

Realistic-load rules on top of the original Code Red brief:
- Queue first: with QUEUE_FIRST or more re-solves due, the second mock becomes another
  re-solve, so failed problems are fixed before new ones pile up.
- Taper: in the last TAPER_DAYS before the interview no new problems are served; the
  mocks become re-solves of problems you already solved. Interview day is warmup only.
- Stories first: until STAR_TARGET stories exist, the behavioral block asks you to write one,
  because a 2-minute prompt needs a story to tell.
- Mixed topics: each new problem of the day comes from a different topic while one is left,
  so you practise recognising the pattern, not repeating it.
"""
import json
from datetime import date, datetime

from sqlmodel import Session, col, select

from google_prep.core import readiness, scheduler
from google_prep.models import Attempt, MockSession, Problem, StarStory, UserProgress

MODE = "code_red"
POOL_WINDOWS = ("30d", "3m")
POOL_LEVELS = ("Medium", "Hard")
WARMUP_CARDS = 5
QUEUE_FIRST = 4
TAPER_DAYS = 2
STAR_TARGET = 8
TREND_DAYS = 7
PLAN_KEY = "code_red:plan:"
BEHAVIORAL_KEY = "code_red:behavioral:"


def active(session: Session) -> bool:
    return scheduler.get_setting(session, "preparation_mode", "google") == MODE


# ---- which problems Code Red uses -----------------------------------------------------------

def in_scope(p: Problem) -> bool:
    """Problems Code Red works with at all: no Foundations, no Easy."""
    return p.category != scheduler.FOUNDATIONS and p.difficulty in POOL_LEVELS


def pool(session: Session) -> list[Problem]:
    """New Code Red problems: in scope and asked at Google in the last 3 months."""
    overrides = scheduler.recency_overrides(session)
    return [p for p in session.exec(select(Problem).order_by(Problem.id)).all()
            if in_scope(p) and scheduler.recency(p, overrides)[0] in POOL_WINDOWS]


def pool_categories(session: Session) -> list[str]:
    cats = {p.category for p in pool(session)}
    return [c for c in scheduler.CATEGORIES if c in cats]


def due(session: Session, now: datetime) -> list[Problem]:
    """Re-solves that are due now, oldest first (in scope only)."""
    return [p for p in scheduler.due_problems(session, now) if in_scope(p)]


def pick_new(session: Session, now: datetime, exclude: set[str] | None = None,
             categories: list[str] | None = None) -> Problem | None:
    """An unseen pool problem from the weakest Code Red topic (most recent at Google first)."""
    exclude = exclude or set()
    tried = set(session.exec(select(UserProgress.problem_id)).all())
    overrides = scheduler.recency_overrides(session)
    fresh = [p for p in pool(session) if p.id not in tried and p.id not in exclude]
    cats = [c for c in (categories or pool_categories(session)) if any(p.category == c for p in fresh)]
    for cat in scheduler.weakest_categories(session, cats):
        options = [p for p in fresh if p.category == cat]
        if options:
            def rank(p: Problem) -> tuple:
                window, freq = scheduler.recency(p, overrides)
                return (scheduler.WINDOW_RANK.get(window, 4), -freq, scheduler.DIFF_RANK.get(p.difficulty, 1), p.id)
            return min(options, key=rank)
    return None


def pick(session: Session, now: datetime, exclude: set[str] | None = None,
         categories: list[str] | None = None) -> Problem | None:
    """Overdue re-solves first, then an unseen problem from the weakest topic. Once every pool
    problem has been tried, the one whose next review is soonest (Practice never runs dry)."""
    exclude = exclude or set()
    for p in due(session, now):
        if p.id not in exclude and (categories is None or p.category in categories):
            return p
    new = pick_new(session, now, exclude, categories)
    if new is not None:
        return new
    progress = {u.problem_id: u for u in session.exec(select(UserProgress)).all()}
    left = [p for p in pool(session) if p.id not in exclude and (categories is None or p.category in categories)]
    return min(left, key=lambda p: (progress[p.id].next_review_date if p.id in progress else now, p.id), default=None)


def warmup(session: Session, now: datetime, count: int = WARMUP_CARDS):
    """Five 60-second cards from the Code Red topics, tuned to your top mistake."""
    return scheduler.pick_warmup(session, now, count, set(pool_categories(session)), basics_first=False)


# ---- the daily mission ----------------------------------------------------------------------

def _solved_before(session: Session, exclude: set[str]) -> list[Problem]:
    """Problems you already solved, least recently solved first (taper confidence reps)."""
    rows = session.exec(select(Problem, UserProgress)
                        .join(UserProgress, col(UserProgress.problem_id) == col(Problem.id))
                        .where(col(UserProgress.last_solved_at).is_not(None))
                        .order_by(UserProgress.last_solved_at)).all()
    return [p for p, _ in rows if in_scope(p) and p.id not in exclude]


def _new_plan(session: Session, now: datetime, days_left: int) -> dict:
    waiting = due(session, now)
    if days_left <= 0:
        return {"phase": "interview_day", "coding": []}
    if days_left <= TAPER_DAYS:
        reps = [p.id for p in (waiting + _solved_before(session, {p.id for p in waiting}))][:3]
        return {"phase": "taper", "coding": [{"slot": f"rep{i + 1}", "problem_id": pid, "kind": "resolve"}
                                             for i, pid in enumerate(reps)]}
    coding = []
    taken: set[str] = set()
    used_topics: set[str] = set()

    def fresh() -> Problem | None:
        """A new problem, from a topic not used yet today when one is left (mixed practice)."""
        other = [c for c in pool_categories(session) if c not in used_topics]
        return pick_new(session, now, taken, other) or pick_new(session, now, taken)

    def take(p: Problem) -> None:
        taken.add(p.id)
        used_topics.add(p.category)

    strike = waiting[0] if waiting else fresh()
    if strike:
        coding.append({"slot": "strike", "problem_id": strike.id, "kind": "resolve" if waiting else "new"})
        take(strike)
    first = fresh()
    if first:
        coding.append({"slot": "mock1", "problem_id": first.id, "kind": "mock"})
        take(first)
    if len(waiting) >= QUEUE_FIRST:
        second = waiting[1]
        coding.append({"slot": "mock2", "problem_id": second.id, "kind": "resolve", "queue_first": True})
    else:
        second = fresh()
        if second:
            coding.append({"slot": "mock2", "problem_id": second.id, "kind": "mock"})
    return {"phase": "sprint", "coding": coding}


def day_plan(session: Session, now: datetime) -> dict:
    """Today's fixed plan: made on the first visit of the day, then kept."""
    key = PLAN_KEY + now.date().isoformat()
    raw = scheduler.get_setting(session, key)
    try:
        plan = json.loads(raw) if raw else None
    except ValueError:
        plan = None
    if not isinstance(plan, dict) or "coding" not in plan:
        days_left = (scheduler.interview_date(session) - now.date()).days
        plan = _new_plan(session, now, days_left)
        scheduler.set_setting(session, key, json.dumps(plan))
    plan["coding"] = [c for c in plan["coding"] if session.get(Problem, c.get("problem_id", "")) is not None]
    return plan


def _attempted_today(session: Session, day: date) -> set[str]:
    return set(session.exec(select(Attempt.problem_id).where(Attempt.day == day.isoformat())).all())


def star_count(session: Session) -> int:
    return len(session.exec(select(StarStory.id)).all())


def behavioral_done(session: Session, day: date) -> bool:
    if star_count(session) < STAR_TARGET:
        return any(s.created_at.date() == day for s in session.exec(select(StarStory)).all())
    return scheduler.get_setting(session, BEHAVIORAL_KEY + day.isoformat()) == "1"


def mark_behavioral(session: Session, day: date) -> None:
    scheduler.set_setting(session, BEHAVIORAL_KEY + day.isoformat(), "1")


SLOT_TITLE = {"strike": "Re-Solve Strike", "mock1": "Live mock 1", "mock2": "Live mock 2",
              "rep1": "Confidence re-solve 1", "rep2": "Confidence re-solve 2", "rep3": "Confidence re-solve 3"}


def _coding_block(session: Session, item: dict, done_ids: set[str]) -> dict:
    p = session.get(Problem, item["problem_id"])
    if item["slot"] == "strike":
        detail = ("Due re-solve: a problem that beat you. Code it again from memory."
                  if item["kind"] == "resolve" else "Nothing is due yet, so this is a new pool problem.")
    elif item.get("queue_first"):
        detail = f"{QUEUE_FIRST}+ re-solves are waiting, so this slot clears one instead of adding a new problem."
    elif item["kind"] == "resolve":
        detail = "Taper day: re-solve a problem you know, no new material."
    else:
        detail = "Unseen problem from your weakest topic, Part 1 + Part 2 follow-up."
    return {"id": item["slot"], "title": SLOT_TITLE.get(item["slot"], item["slot"]), "minutes": 45,
            "kind": item["kind"], "detail": detail, "done": p.id in done_ids,
            "problem": {"id": p.id, "title": p.title, "difficulty": p.difficulty,
                        "category": scheduler.CATEGORIES[p.category]},
            "action": {"type": "mock", "problem_id": p.id}}


def today(session: Session, now: datetime) -> dict:
    day = now.date()
    plan = day_plan(session, now)
    days_left = (scheduler.interview_date(session) - day).days
    sched = scheduler.schedule_for(session, day)
    done_ids = _attempted_today(session, day)
    blocks = [{"id": "warmup", "title": "5-minute mistake warmup", "minutes": 5, "done": sched.morning_completed,
               "detail": f"{WARMUP_CARDS} rapid 60-second cards, tuned to what cost you points.",
               "action": {"type": "recall", "mode": MODE}}]
    blocks += [_coding_block(session, item, done_ids) for item in plan["coding"]]
    stories = star_count(session)
    if stories < STAR_TARGET:
        behavioral = {"id": "behavioral", "title": "Write one STAR story", "minutes": 15,
                      "detail": f"{stories}/{STAR_TARGET} stories ready. A 2-minute answer needs a story first.",
                      "action": {"type": "star"}}
    else:
        behavioral = {"id": "behavioral", "title": "2-minute Googleyness drill", "minutes": 2,
                      "detail": "One random behavioral prompt, answered out loud against a 2-minute timer.",
                      "action": {"type": "behavioral"}}
    behavioral["done"] = behavioral_done(session, day)
    blocks.append(behavioral)
    r = readiness.compute(session)
    notes = []
    if plan["phase"] == "sprint" and len([c for c in plan["coding"] if c["slot"].startswith("mock")]) < 2:
        notes.append("The Code Red pool is running out of unseen problems. Clear the Re-Solve Queue next.")
    if plan["phase"] == "taper":
        notes.append("Taper: no new problems in the last two days. Sleep counts more than one more problem.")
    if plan["phase"] == "interview_day":
        notes.append("Interview day: a light warmup only. No new problems.")
    if not active(session):
        notes.append("Code Red is not your current mode.")
    return {"phase": plan["phase"], "days_left": days_left,
            "interview_date": scheduler.interview_date(session).isoformat(),
            "readiness": round(r.index, 1), "readiness_provisional": r.is_provisional,
            "queue_due": len(due(session, now)), "blocks": blocks,
            "completed": sum(b["done"] for b in blocks), "total": len(blocks), "notes": notes}


# ---- re-solve queue -------------------------------------------------------------------------

def _beat_you(a: Attempt) -> str | None:
    if a.help == "answer":
        return "Saw the answer"
    if a.help == "hint":
        return "Needed a hint"
    if not a.solved:
        return "Missed it"
    if a.mistake == "slow":
        return "Ran out of time"
    return None


def queue(session: Session, now: datetime) -> dict:
    """Every in-scope problem that beat you on its latest attempt, or is due again; most urgent first."""
    latest: dict[str, Attempt] = {}
    for a in session.exec(select(Attempt).order_by(col(Attempt.id))).all():
        latest[a.problem_id] = a
    rows = []
    for prog in session.exec(select(UserProgress)).all():
        p = session.get(Problem, prog.problem_id)
        if p is None or not in_scope(p):
            continue
        last = latest.get(p.id)
        reason = _beat_you(last) if last else None
        is_due = prog.next_review_date <= now
        if not reason and not is_due:
            continue
        rows.append({"id": p.id, "title": p.title, "difficulty": p.difficulty,
                     "category": scheduler.CATEGORIES[p.category], "reason": reason or "Review due",
                     "due": is_due, "next_review": prog.next_review_date.isoformat(timespec="minutes"),
                     "overdue_days": max(0, (now.date() - prog.next_review_date.date()).days) if is_due else 0,
                     "last_day": last.day if last else None})
    rows.sort(key=lambda r: (not r["due"], r["next_review"], r["id"]))
    return {"rows": rows, "due_count": sum(r["due"] for r in rows)}


# ---- progress -------------------------------------------------------------------------------

def solved_solo(session: Session) -> set[str]:
    ids = {p.id for p in pool(session)}
    rows = session.exec(select(Attempt).where(col(Attempt.solved).is_(True))).all()
    return {a.problem_id for a in rows if a.problem_id in ids and (a.help or "none") == "none"}


def progress(session: Session, now: datetime) -> dict:
    total = len(pool(session))
    trend = []
    mocks = session.exec(select(MockSession)).all()
    for back in range(TREND_DAYS - 1, -1, -1):
        day = now.date().toordinal() - back
        scores = [m.score for m in mocks if m.date.date().toordinal() == day]
        trend.append({"day": date.fromordinal(day).isoformat(), "mocks": len(scores),
                      "average": round(sum(scores) / len(scores), 1) if scores else None})
    mistakes = scheduler.mistake_profile(session)
    r = readiness.compute(session)
    return {"solved_solo": len(solved_solo(session)), "pool_total": total, "trend": trend,
            "mistakes": [{"key": k, "label": scheduler.MISTAKES[k], "count": n} for k, n in mistakes.items()],
            "readiness": round(r.index, 1), "readiness_provisional": r.is_provisional,
            "days_left": (scheduler.interview_date(session) - now.date()).days}
