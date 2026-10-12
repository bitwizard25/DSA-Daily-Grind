"""Day/phase arithmetic, daily schedule records, streaks and task selection."""
from datetime import date, datetime, timedelta

from sqlmodel import Session, col, func, select

from google_prep.core import clock, sm2
from google_prep.models import (
    Attempt,
    DailySchedule,
    DrillProgress,
    Problem,
    RecallDrill,
    Setting,
    TrendingProblem,
    UserProgress,
)

PLAN_DAYS = 56
PHASES = {  # phase: (name, first day, last day)
    0: ("C++ to Python Fast-Track", 1, 2),
    1: ("Foundation & Recall Sprint", 3, 21),
    2: ("Google Core Patterns & Hard Follow-ups", 22, 42),
    3: ("Final Onsite Loop & Googlyness", 43, 56),
}
CATEGORIES = {  # the six Google categories, in plan order
    "graphs": "Graphs",
    "trees": "Trees",
    "heaps": "Heaps",
    "intervals": "Intervals",
    "dp": "DP",
    "monotonic": "Monotonic Queues",
}
MICRO_HOURS = 10  # a workday this long switches today to Emergency Micro-Mode
WINDOW_RANK = {"30d": 0, "3m": 1, "6m": 2, "6m+": 3, "": 4}
WINDOW_LABEL = {"30d": "asked at Google in the last 30 days", "3m": "asked in the last 3 months",
                "6m": "asked in the last 6 months", "6m+": "not reported in the last 6 months",
                "": "no recent frequency data"}
DIFF_RANK = {"Easy": 0, "Medium": 1, "Hard": 2}
MISTAKES = {
    "pattern": "Did not spot the pattern",
    "implementation": "Knew the pattern, could not code it",
    "bug_edge": "Bug or missed edge case",
    "slow": "Too slow",
    "follow_up": "Stuck on the follow-up",
}
PROBLEM_TARGET_DAY = PLAN_DAYS  # mocks in weeks 7-8 also solve bank problems
PROBLEM_START_DAY = PHASES[1][1]  # the bank starts after the 2-day Python fast-track


# ---- settings -------------------------------------------------------------

def get_setting(session: Session, key: str, default: str | None = None) -> str | None:
    row = session.get(Setting, key)
    return row.value if row else default


def set_setting(session: Session, key: str, value: str) -> None:
    session.merge(Setting(key=key, value=value))
    session.commit()


def start_date(session: Session, today: date | None = None) -> date:
    value = get_setting(session, "start_date")
    if value is None:
        value = (today or clock.now().date()).isoformat()
        set_setting(session, "start_date", value)
    return date.fromisoformat(value)


def interview_date(session: Session) -> date:
    value = get_setting(session, "interview_date")
    if value:
        return date.fromisoformat(value)
    return start_date(session) + timedelta(days=PLAN_DAYS)


# ---- calendar -------------------------------------------------------------

def day_number(session: Session, today: date) -> int:
    return max(1, (today - start_date(session, today)).days + 1)


def week_and_day(session: Session, today: date) -> tuple[int, int]:
    """(week number, day within that week), e.g. plan day 10 -> (2, 3)."""
    day = day_number(session, today)
    return (day - 1) // 7 + 1, (day - 1) % 7 + 1


APP_NAME = "[Google SWE II OS]"


def breadcrumb(session: Session, today: date, *parts: str) -> str:
    """Trunk test: app name, then Phase > Week > Day > section > topic > problem."""
    week, dow = week_and_day(session, today)
    phase = phase_for_day(day_number(session, today))
    return f"{APP_NAME} " + " > ".join([f"Phase {phase}", f"Week {week}", f"Day {dow}", *[p for p in parts if p]])


def phase_for_day(day: int) -> int:
    for phase, (_, first, last) in PHASES.items():
        if first <= day <= last:
            return phase
    return 3


def time_slot(now: datetime) -> str:
    if now.hour < 12:
        return "morning"
    if now.hour >= 17:
        return "evening"
    return "midday"


def schedule_for(session: Session, day: date) -> DailySchedule:
    key = day.isoformat()
    row = session.get(DailySchedule, key)
    if row is None:
        row = DailySchedule(date_str=key)
        session.add(row)
        session.commit()
    return row


def mark_done(session: Session, day: date, slot: str, minutes: int = 0, mode: str | None = None) -> None:
    row = schedule_for(session, day)
    if slot == "morning":
        row.morning_completed = True
    elif slot == "evening":
        row.evening_completed = True
    row.total_minutes += max(0, minutes)
    if mode:
        row.mode = mode
    session.add(row)
    session.commit()


def streak(session: Session, today: date) -> int:
    done = {
        r.date_str
        for r in session.exec(select(DailySchedule)).all()
        if r.morning_completed or r.evening_completed
    }
    day = today if today.isoformat() in done else today - timedelta(days=1)
    count = 0
    while day.isoformat() in done:
        count += 1
        day -= timedelta(days=1)
    return count


def needs_micro(hours_worked: float | None) -> bool:
    return hours_worked is not None and hours_worked >= MICRO_HOURS


# ---- mastery per category (used for weak-spot targeting) --------------------

def category_mastery(session: Session) -> dict[str, dict]:
    """Per category: problem score (SM-2 reps), drill score, solved count, total."""
    progress = {p.problem_id: p for p in session.exec(select(UserProgress)).all()}
    drills = {d.drill_id: d for d in session.exec(select(DrillProgress)).all()}
    out = {c: {"problem": 0.0, "drill": 0.0, "solved": 0, "total": 0, "drills": 0} for c in CATEGORIES}
    for p in session.exec(select(Problem)).all():
        row = out[p.category]
        row["total"] += 1
        prog = progress.get(p.id)
        if prog:
            row["problem"] += sm2.score(prog.reps)
            row["solved"] += prog.last_solved_at is not None
    for d in session.exec(select(RecallDrill)).all():
        row = out[d.category]
        row["drills"] += 1
        if d.id in drills:
            row["drill"] += sm2.score(drills[d.id].reps)
    for row in out.values():
        p = row["problem"] / row["total"] if row["total"] else 0.0
        d = row["drill"] / row["drills"] if row["drills"] else 0.0
        row["score"] = 0.7 * p + 0.3 * d
    return out


def weakest_categories(session: Session, allowed: list[str] | None = None) -> list[str]:
    mastery = category_mastery(session)
    order = list(CATEGORIES)
    cats = [c for c in order if allowed is None or c in allowed]
    return sorted(cats, key=lambda c: (mastery[c]["score"], order.index(c)))


# ---- problem selection -------------------------------------------------------

def recency_overrides(session: Session) -> dict[str, tuple[str, float]]:
    """Fresh data from `google-prep sync`, keyed by LeetCode slug. Empty if never synced."""
    return {t.slug: (t.window, t.frequency) for t in session.exec(select(TrendingProblem)).all()}


def recency(p: Problem, overrides: dict[str, tuple[str, float]]) -> tuple[str, float]:
    if overrides:
        return overrides.get(p.slug, ("6m+" if p.slug else p.recency, 0.0))
    return p.recency, p.recent_frequency


def _rank(p: Problem, prog: UserProgress | None, phase: int, now: datetime,
          overrides: dict[str, tuple[str, float]]) -> tuple:
    window, freq = recency(p, overrides)
    if prog is None:
        bucket = 1 if p.phase <= phase else 2
    elif prog.next_review_date <= now and prog.status != "mastered":
        bucket = 0  # failed / due for review: fix it first
    elif prog.status != "mastered":
        bucket = 3
    else:
        bucket = 4
    easy_last = p.difficulty == "Easy"  # low yield for someone who has solved hundreds before
    return (bucket, WINDOW_RANK.get(window, 4), easy_last, -freq, p.phase, DIFF_RANK.get(p.difficulty, 1), p.id)


def pick_target_problem(session: Session, phase: int, now: datetime,
                        categories: list[str] | None = None) -> Problem | None:
    """Weakest category first. Inside it: due re-solves, then unseen problems asked at
    Google most recently (30 days > 3 months > 6 months), then by reported frequency."""
    progress = {p.problem_id: p for p in session.exec(select(UserProgress)).all()}
    problems = session.exec(select(Problem)).all()
    overrides = recency_overrides(session)
    for cat in weakest_categories(session, categories):
        pool = [p for p in problems if p.category == cat]
        if pool:
            return min(pool, key=lambda p: _rank(p, progress.get(p.id), phase, now, overrides))
    return None


def due_problems(session: Session, now: datetime) -> list[Problem]:
    rows = session.exec(
        select(Problem, UserProgress)
        .join(UserProgress, col(UserProgress.problem_id) == col(Problem.id))
        .where(UserProgress.next_review_date <= now)
        .order_by(UserProgress.next_review_date)
    ).all()
    return [p for p, _ in rows]


def record_problem_review(session: Session, problem_id: str, grade: int, solved: bool,
                          now: datetime, min_days: int | None = None) -> UserProgress:
    """SM-2 on the problem. `min_days` applies the 3-tier floors (0 = back in today's queue)."""
    prog = session.exec(select(UserProgress).where(UserProgress.problem_id == problem_id)).first()
    state = sm2.CardState(prog.reps, prog.easiness_factor, prog.interval_days) if prog else sm2.CardState()
    new = sm2.review(state, grade)
    if prog is None:
        prog = UserProgress(problem_id=problem_id, status="learning", next_review_date=now)
    prog.reps, prog.easiness_factor, prog.interval_days = new.reps, new.easiness_factor, new.interval_days
    prog.status = sm2.status_for(new.reps)
    prog.next_review_date = sm2.next_date(now, new)
    if min_days is not None:
        days = 0 if grade < 3 else max(new.interval_days, min_days)
        prog.interval_days, prog.next_review_date = max(1, days), now + timedelta(days=days)
    if solved:
        prog.last_solved_at = now
    session.add(prog)
    session.commit()
    return prog


def log_attempt(session: Session, problem_id: str, minutes: int, solved: bool,
                dry_run_ok: bool | None, grade: int, now: datetime, source: str = "solve",
                mistake: str | None = None, min_days: int | None = None) -> None:
    session.add(Attempt(problem_id=problem_id, day=now.date().isoformat(), minutes=minutes,
                        solved=solved, dry_run_ok=dry_run_ok, mistake=mistake, source=source))
    session.commit()
    record_problem_review(session, problem_id, grade, solved, now, min_days)
    mark_done(session, now.date(), "evening", minutes)


def attempts_on(session: Session, day: date) -> list[Problem]:
    ids = session.exec(select(Attempt.problem_id).where(Attempt.day == day.isoformat()).distinct()).all()
    return [p for p in (session.get(Problem, i) for i in ids) if p is not None]  # retired problems drop out


# ---- recall drill selection ------------------------------------------------

def pick_drills(session: Session, now: datetime, count: int, kind: str | None = None) -> list[RecallDrill]:
    """Due drills first (oldest due first), then unseen drills round-robin across categories."""
    drills = [d for d in session.exec(select(RecallDrill)).all() if kind is None or d.type == kind]
    progress = {p.drill_id: p for p in session.exec(select(DrillProgress)).all()}
    due = sorted((d for d in drills if d.id in progress and progress[d.id].next_review_date <= now),
                 key=lambda d: progress[d.id].next_review_date)
    order = list(CATEGORIES)
    seen_in_cat: dict[str, int] = {}
    unseen = []
    for d in sorted((d for d in drills if d.id not in progress), key=lambda d: d.id):
        idx = seen_in_cat.get(d.category, 0)
        seen_in_cat[d.category] = idx + 1
        unseen.append((idx, order.index(d.category), d))
    unseen.sort(key=lambda t: (t[0], t[1]))
    return (due + [d for _, _, d in unseen])[:count]


def mistake_profile(session: Session, last: int = 20) -> dict[str, int]:
    rows = session.exec(select(Attempt).where(col(Attempt.mistake).is_not(None))
                        .order_by(col(Attempt.id).desc()).limit(last)).all()
    counts = {k: 0 for k in MISTAKES}
    for a in rows:
        if a.mistake in counts:
            counts[a.mistake] += 1
    return counts


def dominant_mistake(session: Session) -> str | None:
    counts = mistake_profile(session)
    top = max(counts, key=lambda k: counts[k])
    return top if counts[top] >= 2 else None


def pick_warmup(session: Session, now: datetime, count: int = 5) -> list[RecallDrill]:
    """Morning drills shaped by your recent mistakes: bugs and missed edge cases get more
    bug-hunts, missed patterns get only pattern drills, otherwise one bug-hunt a day."""
    mistake = dominant_mistake(session)
    bugs_wanted = {"bug_edge": 2, "pattern": 0}.get(mistake or "", 1)
    bugs = pick_drills(session, now, bugs_wanted, "bug_hunt") if bugs_wanted else []
    patterns = pick_drills(session, now, count - len(bugs), "pattern")
    return patterns + bugs


def pace(session: Session, today: date) -> dict:
    """Problems solved vs. a straight-line plan from Day 3 (after the Python fast-track) to the
    last plan day. Only finished days count, so the first morning is never 'behind'."""
    total = count(session, Problem)
    solved = session.exec(select(func.count()).select_from(UserProgress)
                          .where(col(UserProgress.last_solved_at).is_not(None))).one()
    day = day_number(session, today)
    span = PROBLEM_TARGET_DAY - PROBLEM_START_DAY + 1
    done_days = min(max(day - PROBLEM_START_DAY, 0), span)
    expected = min(total, round(total * done_days / span))
    days_left = max(1, PROBLEM_TARGET_DAY - max(day, PROBLEM_START_DAY) + 1)
    return {"solved": solved, "total": total, "expected": expected, "behind": max(0, expected - solved),
            "per_day": max(0.0, (total - solved) / days_left)}


def grade_drill(session: Session, drill_id: str, grade: int, now: datetime, min_days: int = 0) -> DrillProgress:
    """SM-2 on the drill, with the recall screen's floors: a miss comes back today, effort in
    3+ days, an instant reflex in 7+ days (SM-2 stretches the gap further as reps build)."""
    prog = session.get(DrillProgress, drill_id)
    state = sm2.CardState(prog.reps, prog.easiness_factor, prog.interval_days) if prog else sm2.CardState()
    new = sm2.review(state, grade)
    if prog is None:
        prog = DrillProgress(drill_id=drill_id, next_review_date=now)
    days = 0 if grade < 3 else max(new.interval_days, min_days)
    prog.reps, prog.easiness_factor, prog.interval_days = new.reps, new.easiness_factor, max(1, days)
    prog.status = sm2.status_for(new.reps)
    prog.next_review_date = now + timedelta(days=days)
    prog.last_grade = grade
    session.add(prog)
    session.commit()
    return prog


def count(session: Session, model) -> int:
    return session.exec(select(func.count()).select_from(model)).one()
