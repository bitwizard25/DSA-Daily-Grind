"""SQLModel tables. Problem, UserProgress, DailySchedule, MockSession and StarStory follow
spec section 6 (field names included); the rest hold what the readiness index needs.
Both the CLI and `google-prep ui` read and write this one database."""
import json
from datetime import datetime
from typing import Optional

from pydantic import NaiveDatetime  # local wall-clock time; this app is single-user and offline
from sqlmodel import Field, SQLModel

from google_prep.core.clock import now


class Problem(SQLModel, table=True):
    id: str = Field(primary_key=True)  # G-<LeetCode number>, stable across bank updates
    title: str
    leetcode_id: int
    slug: str = ""  # leetcode.com/problems/<slug>
    difficulty: str  # Medium, Hard
    category: str  # graphs, trees, heaps, intervals, dp, monotonic
    phase: int = 1  # plan phase the problem belongs to (1-3)
    key_pattern: str
    key_insight: str = ""
    time_target: str
    space_target: str
    google_frequency: str  # Very High (asked in last 30 days), High (last 6 months)
    recency: str = ""  # most recent window it was reported at Google: 30d, 3m, 6m
    recent_frequency: float = 0.0
    statement: str = ""
    edge_cases: str = "[]"  # JSON-serialized list
    google_follow_up: str

    def edge_list(self) -> list[str]:
        try:
            return list(json.loads(self.edge_cases or "[]"))
        except ValueError:
            return [self.edge_cases]


class UserProgress(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    problem_id: str = Field(foreign_key="problem.id", index=True, unique=True)
    status: str = "unseen"  # unseen, learning, mastered
    reps: int = 0
    easiness_factor: float = 2.5
    interval_days: int = 1
    next_review_date: NaiveDatetime = Field(default_factory=now)
    last_solved_at: Optional[NaiveDatetime] = None


class DailySchedule(SQLModel, table=True):
    date_str: str = Field(primary_key=True)  # YYYY-MM-DD
    mode: str = "normal"  # normal, micro, rest
    morning_completed: bool = False
    evening_completed: bool = False
    total_minutes: int = 0


class MockSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    problem_id: str
    date: NaiveDatetime = Field(default_factory=now)
    duration_seconds: int
    clarified_constraints: bool = False  # Analytical Ability
    stated_complexity: bool = False  # Analytical Ability
    clean_code: bool = False  # Coding & Craft
    dry_run_passed: bool = False  # Verification & Rigor
    follow_up_handled: bool = False  # System Scaling
    score: float = 0.0  # 0-100, the four pillars at 25% each
    solved: bool = False
    limit_minutes: int = 45
    follow_up: str = ""


class StarStory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dimension: str  # Ambiguity, Ownership, Disagreement, Humility
    title: str
    situation: str
    task: str
    action: str
    result: str
    created_at: NaiveDatetime = Field(default_factory=now)


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
