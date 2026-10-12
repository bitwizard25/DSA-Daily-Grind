"""SQLModel tables. The first four follow spec section 6; the rest hold data the
readiness index needs (attempts, mocks, drill progress, settings)."""
from datetime import datetime
from typing import Optional

from pydantic import NaiveDatetime  # local wall-clock time; this app is single-user and offline
from sqlmodel import Field, SQLModel


class Problem(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str
    leetcode_id: int
    slug: str = ""  # leetcode.com/problems/<slug>
    difficulty: str  # Easy, Medium, Hard
    category: str  # arrays, pointers, intervals, heaps, trees, graphs, dp, design
    phase: int = 1  # plan phase the problem belongs to (1-3)
    key_pattern: str
    time_target: str
    space_target: str
    google_frequency: str  # Very High (asked in last 30 days), High (last 6 months), Moderate
    recency: str = ""  # most recent window it was reported at Google: 30d, 3m, 6m, 6m+
    recent_frequency: float = 0.0
    statement: str = ""
    edge_case: str = ""
    google_follow_up: str


class UserProgress(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    problem_id: str = Field(foreign_key="problem.id", index=True, unique=True)
    status: str  # unseen, learning, mastered
    reps: int = 0
    easiness_factor: float = 2.5
    interval_days: int = 1
    next_review_date: NaiveDatetime
    last_solved_at: Optional[NaiveDatetime] = None


class DailySchedule(SQLModel, table=True):
    date: str = Field(primary_key=True)  # YYYY-MM-DD
    mode: str = "normal"  # normal, micro, rest
    morning_completed: bool = False
    evening_completed: bool = False
    total_minutes_spent: int = 0


class StarStory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dimension: str  # Ambiguity, Ownership, Collaboration, Humility
    title: str
    situation: str
    task: str
    action: str
    result: str
    updated_at: NaiveDatetime = Field(default_factory=datetime.now)


class RecallDrill(SQLModel, table=True):
    id: str = Field(primary_key=True)
    type: str  # pattern, bug_hunt
    category: str
    problem_id: Optional[str] = None
    prompt: str
    pattern: str
    complexity: str
    edge_case: str


class DrillProgress(SQLModel, table=True):
    drill_id: str = Field(primary_key=True, foreign_key="recalldrill.id")
    status: str = "learning"
    reps: int = 0
    easiness_factor: float = 2.5
    interval_days: int = 1
    next_review_date: NaiveDatetime
    last_grade: int = 0


class Attempt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    problem_id: str = Field(foreign_key="problem.id", index=True)
    day: str
    minutes: int
    solved: bool
    dry_run_ok: Optional[bool] = None  # did the hand trace match the expected output?
    mistake: Optional[str] = None  # pattern, implementation, bug_edge, slow, follow_up
    source: str = "solve"  # solve, mock


class MockSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    problem_id: str = Field(foreign_key="problem.id")
    day: str
    limit_minutes: int
    used_seconds: int
    solved: bool
    follow_up: str = ""
    checklist_done: int = 0
    algorithms: int = 0  # Google rubric pillars, 1-4
    coding: int = 0
    communication: int = 0
    problem_solving: int = 0


class Setting(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


class TrendingProblem(SQLModel, table=True):
    """Company frequency data fetched by `google-prep sync` (overrides the bundled snapshot)."""
    slug: str = Field(primary_key=True)
    title: str
    difficulty: str
    topics: str = ""
    window: str  # 30d, 3m, 6m, 6m+
    frequency: float = 0.0


class ReadinessSnapshot(SQLModel, table=True):
    date: str = Field(primary_key=True)
    score: float
    pattern_mastery: float
    mock_speed: float
    dry_run: float
    star: float
