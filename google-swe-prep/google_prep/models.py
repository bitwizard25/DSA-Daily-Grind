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
    difficulty: str  # Easy, Medium, Hard
    category: str  # foundations, graphs, trees, heaps, intervals, dp, monotonic
    phase: int = 1  # plan phase the problem belongs to (1-3)
    sequence: int = 0  # Foundations only: the order they are served in (easy building blocks first)
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
    mistake: Optional[str] = None  # pattern, implementation (legacy), python_syntax, implementation_logic, bug_edge, slow, follow_up
    source: str = "solve"  # solve, mock
    help: Optional[str] = None  # none, hint (a nudge), answer (saw a solution, a video or AI code)


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


class LessonProgress(SQLModel, table=True):
    """Phase 0 fast-track: one row per answered exercise (latest answer wins) and one per finished day."""
    item_id: str = Field(primary_key=True)  # "d1-e3" or "day-1"
    kind: str  # exercise, day
    correct: Optional[bool] = None
    updated_at: NaiveDatetime = Field(default_factory=now)


class ReadinessSnapshot(SQLModel, table=True):
    date: str = Field(primary_key=True)
    score: float
    pattern_mastery: float
    mock_speed: float
    dry_run: float
    star: float


class TrainingCheckin(SQLModel, table=True):
    """One daily interview-habit intention and reflection, kept with real practice history."""
    date_str: str = Field(primary_key=True)  # local YYYY-MM-DD, matching Attempt.day
    habit_id: str
    intention: str = ""
    reflection: str = ""
    confidence_before: Optional[int] = None  # optional self-report, 1-5; never a readiness score
    confidence_after: Optional[int] = None
    created_at: NaiveDatetime = Field(default_factory=now)
    updated_at: NaiveDatetime = Field(default_factory=now)


class SupportBlock(SQLModel, table=True):
    """User-acknowledged focus time, never a coding grade."""
    id: str = Field(primary_key=True)
    day: str = Field(index=True)
    task: str
    target_minutes: int
    status: str = 'active'
    active_slot: Optional[str] = Field(default='focus', unique=True)
    elapsed_seconds: float = 0
    running_since: Optional[NaiveDatetime] = None
    created_at: NaiveDatetime = Field(default_factory=now)
    updated_at: NaiveDatetime = Field(default_factory=now)
    note: str = ''
    source: str = 'timer'


class SupportRescue(SQLModel, table=True):
    id: str = Field(primary_key=True)
    day: str = Field(index=True)
    blocker: str
    helpful: bool = False
    created_at: NaiveDatetime = Field(default_factory=now)


class ComprehensionAttempt(SQLModel, table=True):
    """Saved learning work, explicitly self-reported; never a DSA grade."""
    id: str = Field(primary_key=True)
    exercise_id: str = Field(index=True)
    language: str
    active_slot: Optional[str] = Field(default=None, unique=True)
    status: str = 'draft'
    revision: int = 0
    notes: str = '{}'
    code: str = ''
    help: str = 'none'
    created_at: NaiveDatetime = Field(default_factory=now)
    updated_at: NaiveDatetime = Field(default_factory=now)
