import pytest

from google_prep.core import scheduler, trending
from google_prep.models import Problem, TrendingProblem

HEADER = "Difficulty,Title,Frequency,Acceptance Rate,Link,Topics\n"
CSV = {
    "1. Thirty Days": HEADER
    + 'HARD,Split Array Largest Sum,53.2,0.5,https://leetcode.com/problems/split-array-largest-sum,"Array, Binary Search, Dynamic Programming"\n'
    + 'EASY,Two Sum,100.0,0.5,https://leetcode.com/problems/two-sum,"Array, Hash Table"\n',
    "2. Three Months": HEADER
    + 'MEDIUM,Generate Parentheses,48.3,0.5,https://leetcode.com/problems/generate-parentheses,"String, Dynamic Programming, Backtracking"\n'
    + 'HARD,Split Array Largest Sum,40.0,0.5,https://leetcode.com/problems/split-array-largest-sum,"Array, Binary Search"\n',
    "3. Six Months": HEADER + 'EASY,Fizz Buzz,10,0.5,https://leetcode.com/problems/fizz-buzz,"Math"\n'
    + 'EASY,Last Stone Weight,20,0.5,https://leetcode.com/problems/last-stone-weight,"Array, Heap (Priority Queue)"\n'
    + 'MEDIUM,Add Two Numbers,70,0.5,https://leetcode.com/problems/add-two-numbers,"Linked List, Math"\n',
    "4. More Than Six Months": HEADER
    + 'MEDIUM,Number of Provinces,30,0.5,https://leetcode.com/problems/number-of-provinces,"Union-Find"\n',
}


def fake_fetch(url):
    for name, text in CSV.items():
        if name.replace(" ", "%20") in url:
            return text
    raise AssertionError(url)


def test_sync_keeps_most_recent_window(session):
    counts = trending.sync(session, fetch=fake_fetch)
    assert counts == {"6m+": 1, "6m": 3, "3m": 2, "30d": 2}
    split = session.get(TrendingProblem, "split-array-largest-sum")
    assert (split.window, split.frequency, split.difficulty) == ("30d", 53.2, "Hard")
    assert session.get(TrendingProblem, "number-of-provinces").window == "6m+"


def test_hot_only_lists_six_category_problems_outside_the_bank(session):
    trending.sync(session, fetch=fake_fetch)
    assert [t.title for t in trending.hot(session)] == ["Generate Parentheses"]  # Split Array is in the bank
    with_easy = [t.title for t in trending.hot(session, include_easy=True)]
    assert "Last Stone Weight" in with_easy
    assert "Fizz Buzz" not in with_easy and "Add Two Numbers" not in with_easy  # outside the six categories


def test_add_problem_imports_with_category(session):
    trending.sync(session, fetch=fake_fetch)
    p = trending.add_problem(session, "https://leetcode.com/problems/generate-parentheses/", phase=2)
    assert p.id == "X-01" and p.category == "dp" and p.recency == "3m"
    assert trending.add_problem(session, "generate-parentheses", phase=2).id == "X-01"  # no duplicates
    assert trending.add_problem(session, "split-array-largest-sum", phase=2).id == "G-410"  # already in the bank
    q = trending.add_problem(session, "last-stone-weight", phase=1)
    assert q.id == "X-02" and q.category == "heaps"
    assert session.get(Problem, "X-02").title == "Last Stone Weight"
    with pytest.raises(KeyError, match="outside the six"):
        trending.add_problem(session, "add-two-numbers", phase=1)
    with pytest.raises(KeyError, match="sync"):
        trending.add_problem(session, "not-a-problem", phase=1)


def test_failed_sync_keeps_old_data(session):
    trending.sync(session, fetch=fake_fetch)

    def broken(url):
        raise OSError("offline")

    with pytest.raises(OSError):
        trending.sync(session, fetch=broken)
    assert scheduler.count(session, TrendingProblem) == 7


def test_slug_parsing():
    assert trending.slug_of("https://leetcode.com/problems/two-sum/description/") == "two-sum"
    assert trending.slug_of("Two-Sum") == "two-sum"
