import pytest

from google_prep.core import sm2


def test_first_three_good_reviews_follow_sm2_intervals():
    s = sm2.review(sm2.CardState(), 4)
    assert (s.reps, s.interval_days) == (1, 1)
    s = sm2.review(s, 4)
    assert (s.reps, s.interval_days) == (2, 6)
    s = sm2.review(s, 4)
    assert s.reps == 3 and s.interval_days == round(6 * s.easiness_factor)


def test_failure_resets_reps_and_interval():
    s = sm2.review(sm2.review(sm2.CardState(), 5), 5)
    failed = sm2.review(s, 2)
    assert (failed.reps, failed.interval_days) == (0, 1)
    assert failed.easiness_factor < s.easiness_factor


def test_easiness_factor_has_a_floor():
    s = sm2.CardState()
    for _ in range(20):
        s = sm2.review(s, 1)
    assert s.easiness_factor == sm2.MIN_EF


def test_perfect_grades_raise_easiness():
    assert sm2.review(sm2.CardState(), 5).easiness_factor > 2.5
    assert sm2.review(sm2.CardState(), 3).easiness_factor < 2.5


@pytest.mark.parametrize("grade", [0, 6])
def test_invalid_grade(grade):
    with pytest.raises(ValueError):
        sm2.review(sm2.CardState(), grade)


def test_status_and_score():
    assert sm2.status_for(2) == "learning"
    assert sm2.status_for(3) == "mastered"
    assert sm2.score(0) == 0 and sm2.score(3) == 1 and sm2.score(10) == 1
