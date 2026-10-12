"""The 4-pillar scorecard from spec section 3, 25% each. Shared by the terminal flow and
the full-screen UI so both score a mock the same way."""

PILLARS = {
    "analytical": "Analytical Ability",
    "craft": "Coding & Craft",
    "verification": "Verification & Rigor",
    "scaling": "System Scaling",
}
# (MockSession field, pillar, question). The whiteboard checklist pre-fills the first four.
CHECKS = [
    ("clarified_constraints", "analytical", "Clarified constraints (N, ranges) and edge cases before coding"),
    ("stated_complexity", "analytical", "Stated time and space complexity before coding"),
    ("clean_code", "craft", "Idiomatic, modular code with descriptive names"),
    ("dry_run_passed", "verification", "Traced a non-trivial test case line by line, and it matched"),
    ("follow_up_handled", "scaling", "Handled the minute-35 follow-up with clear trade-offs"),
]
PRACTICE_CHECKS = ["dry_run_passed"]  # timed solves only self-grade verification

# The same three unambiguous outcomes as recall (Krug's second law). They also reschedule the
# problem: Missed -> today's re-solve queue, Effort -> 3 days, Instant -> next week.
OUTCOMES = {"1": "Missed it", "2": "Solved with Effort", "3": "Instant Reflex"}
OUTCOME_DAYS = {"1": 0, "2": 3, "3": 7}
OUTCOME_GRADE = {"1": 2, "2": 4, "3": 5}  # SM-2 quality


def outcome_for(choice: str, over_time: bool) -> str:
    """An over-time solve cannot be an instant reflex."""
    return "2" if choice == "3" and over_time else choice


def from_ticks(ticks: list[bool]) -> dict[str, bool]:
    checks = {field: False for field, _, _ in CHECKS}
    for (field, _, _), ticked in zip(CHECKS[:4], ticks):
        checks[field] = bool(ticked)
    return checks


def pillar_scores(checks: dict[str, bool]) -> dict[str, float]:
    """0-1 per pillar; Analytical averages its two checks."""
    out = {}
    for pillar in PILLARS:
        fields = [f for f, p, _ in CHECKS if p == pillar]
        out[pillar] = sum(bool(checks.get(f)) for f in fields) / len(fields)
    return out


def score(checks: dict[str, bool]) -> float:
    return 25 * sum(pillar_scores(checks).values())


def hire_signal(points: float, solved: bool) -> str:
    """An unsolved problem caps the signal: strong pillars rarely rescue a missing solution."""
    if points >= 87.5 and solved:
        return "Strong Hire"
    if points >= 62.5 and solved:
        return "Lean Hire"
    if points >= 50:
        return "Lean No Hire"
    return "No Hire"
