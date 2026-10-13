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


# The honesty question, asked before anything else. Copying answers is how hundreds of "solved"
# problems turn into no skill, so a seen answer never counts as solved.
HELP = {"none": "No help: all my own work", "hint": "A hint or a nudge",
        "answer": "I saw the answer (solution, video or AI code)"}
HELP_KEYS = {"1": "none", "2": "hint", "3": "answer"}
AGAIN = {"1": "back in today's re-solve queue", "2": "re-solve in 3 days", "3": "re-solve next week"}
AGAIN_ANSWER = "code it again from memory tomorrow, without looking"


def outcome_for(choice: str, over_time: bool, help: str = "none") -> str:  # noqa: A002
    """An over-time solve cannot be an instant reflex, nor can one that needed a hint.
    After seeing the answer it is a miss: it is not your solve yet."""
    if help == "answer":
        return "1"
    return "2" if choice == "3" and (over_time or help == "hint") else choice


def again_for(outcome: str, help: str = "none") -> str:  # noqa: A002
    return AGAIN_ANSWER if help == "answer" else AGAIN[outcome]


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
