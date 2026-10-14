# Dynamic programming: solve each small question once

> In one line: write the plain recursion, notice it repeats the same questions, then save each answer once.

## When to use it
- "Number of ways" to do something, for example to climb stairs or walk through a grid.
- "Minimum cost", "fewest coins", "maximum money", "longest ..." over many choices.
- At each step you make a small choice (take or skip, step 1 or step 2) and the rest is a smaller problem.
- A simple recursive answer works on small inputs but is far too slow on big ones.

## The idea in plain words
**Dynamic programming** (DP) means: break a problem into smaller problems of the same kind,
solve each smaller problem only once, and save its answer.

Each smaller problem is called a **subproblem**. A **state** is the small set of numbers that names
one subproblem, for example "step i" or "cell (r, c)".

The usual path, at a whiteboard:
1. **Recursion**: a function that calls itself on smaller inputs. It is correct, but it can ask the
   same question again and again. That repeat is called **overlapping subproblems**.
2. **Memoization**: save the answer for each state the first time, and reuse it later.
   Python can do it for you (see the templates). This style is called **top-down**:
   start from the big question and go down to smaller ones.
3. **Bottom-up table**: a list `dp` where `dp[i]` holds the answer for state i.
   Fill it from the smallest state up, with a plain loop. No recursion is needed.

Four questions to answer before you code:
1. **State**: what does `dp[i]` mean, in words? Write it as a comment.
2. **Transition**: how does `dp[i]` come from smaller states? This is one formula.
3. **Base case**: the smallest states, which you know without any work.
4. **Answer**: which cell holds the final answer, for example `dp[n]`.

## Python template

### Climbing stairs: plain recursion
You climb n steps. Each move goes up 1 or 2 steps. How many different ways reach the top?
The last move was 1 step (from n - 1) or 2 steps (from n - 2), so `ways(n) = ways(n - 1) + ways(n - 2)`.

```python
calls = 0  # counts how many times ways() runs

def ways(n):
    global calls  # global: lets the function change calls
    calls += 1
    if n <= 1:
        return 1  # base case: 0 or 1 step left gives one way
    return ways(n - 1) + ways(n - 2)

print(ways(5))  # 8
# the same small questions are asked again and again
print(calls)  # 15
```

### Same idea with @lru_cache (memoization, top-down)
`@lru_cache` is a **decorator**: an `@name` line above a function that adds behaviour.
This one saves each answer, and returns the saved answer when the same `n` comes again.

```python
from functools import lru_cache

@lru_cache(maxsize=None)  # maxsize=None: no limit, keep every answer
def ways(n):
    if n <= 1:
        return 1
    return ways(n - 1) + ways(n - 2)

print(ways(5))  # 8
print(ways(40))  # 165580141
# misses = answers computed for the first time: each n from 0 to 40, once
print(ways.cache_info().misses)  # 41
```

### Same idea as a bottom-up table

```python
def ways(n):
    # dp[i] = number of ways to reach step i
    dp = [0] * (n + 1)
    dp[0] = 1  # base case: one way to stay at step 0 (do nothing)
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]
        if i >= 2:
            dp[i] += dp[i - 2]
    return dp[n]

print(ways(5))  # 8
print(ways(40))  # 165580141
```

### House Robber: take it or skip it
Houses in a row hold money. You cannot rob two neighbour houses. What is the most money?
At house i: skip it (keep `dp[i - 1]`) or rob it (`dp[i - 2]` plus its money).

```python
def rob(nums):
    n = len(nums)
    # dp[i] = most money from the first i houses
    # house number i is nums[i - 1], because list indexes start at 0
    dp = [0] * (n + 1)
    if n > 0:
        dp[1] = nums[0]
    for i in range(2, n + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
    return dp[n]

print(rob([2, 7, 9, 3, 1]))  # 12
```

Each `dp[i]` only needs the last two values, so two variables are enough:

```python
def rob(nums):
    prev2, prev1 = 0, 0  # best money two houses back, one house back
    for money in nums:
        best = max(prev1, prev2 + money)
        prev2 = prev1
        prev1 = best
    return prev1

print(rob([2, 7, 9, 3, 1]))  # 12
print(rob([]))  # 0
```

### Unique Paths: a 2D table for a grid
A robot starts at the top-left cell and moves only right or down. How many paths reach the bottom-right?
A cell is reached from the cell above or the cell on the left.

```python
def unique_paths(rows, cols):
    # dp[r][c] = number of paths from the start to cell (r, c)
    # first row and first column: only 1 path (straight right or straight down)
    dp = [[1] * cols for _ in range(rows)]
    for r in range(1, rows):
        for c in range(1, cols):
            dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
    return dp[rows - 1][cols - 1]

print(unique_paths(3, 3))  # 6
print(unique_paths(3, 7))  # 28
```

### Longest Increasing Subsequence (LIS)
A **subsequence** keeps some items in their order, with gaps allowed: `[2, 5, 7]` from `[2, 9, 5, 7]`.

```python
def lis(nums):
    # dp[i] = length of the longest increasing subsequence that ends at nums[i]
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp, default=0)  # default=0: answer for an empty list

# one answer is [2, 5, 7, 101]
print(lis([10, 9, 2, 5, 3, 7, 101, 18]))  # 4
print(lis([]))  # 0
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `vector<long long> dp(n + 1, 0);` | `dp = [0] * (n + 1)` | Python ints never overflow |
| `vector<vector<int>> dp(r, vector<int>(c, 1));` | `dp = [[1] * c for _ in range(r)]` | a new list for each row |
| `if (memo.count(n)) return memo[n];` | `@lru_cache(maxsize=None)` above the `def` | the decorator saves and checks answers for you |
| `dp[i] = max(dp[i], dp[j] + 1);` | `dp[i] = max(dp[i], dp[j] + 1)` | same, without the `;` |
| `*max_element(dp.begin(), dp.end())` | `max(dp)` | `max(dp, default=0)` if dp can be empty |
| `INT_MAX` for "not possible yet" | `float("inf")` | bigger than any number |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| Plain recursion for climbing stairs | O(2^n) | each call makes two more calls, so the work explodes as n grows |
| Memoization or table for climbing stairs | O(n) | each of the n states is solved once, with O(1) work |
| House Robber | O(n) time, O(1) memory with two variables | one pass over the houses |
| Unique Paths | O(rows × cols) | one cell, one addition |
| LIS with this table | O(n^2) | for each item, look at every item before it |
| Any DP | number of states × work per state | count the states first, then the work for one state |

## Common mistakes
- Mistake: starting to code before you know what `dp[i]` means. Fix: write the state as a comment first.
- Mistake: a missing or wrong base case, like `dp[0] = 0` for climbing stairs. Fix: check the smallest inputs (0, 1, 2) by hand.
- Mistake: `@lru_cache` without `maxsize=None`. It keeps only 128 answers and drops old ones, so they are computed again. Fix: `@lru_cache(maxsize=None)`, or `@cache` (the same thing, also from `functools`).
- Mistake: passing a list to a cached function. It raises `TypeError`: a list can change, so it cannot be a saved key. Fix: pass indexes or a tuple.
- Mistake: `[[0] * cols] * rows` for a 2D table. All rows are the same list. Fix: `[[0] * cols for _ in range(rows)]`.
- Mistake: very deep recursion on a big input raises `RecursionError`. Fix: use the bottom-up table.
- Mistake: in LIS, returning `dp[-1]`. The longest one may end earlier. Fix: return `max(dp)`.

## Practice in the app
- **G-198 House Robber**: the take-or-skip template above.
- **G-62 Unique Paths**: the 2D grid template above.
- **G-300 Longest Increasing Subsequence**: the LIS template; `dp[i]` ends at item i.
- **G-322 Coin Change**: `dp[a]` = fewest coins for amount a; try each coin as the last one. Start with `float("inf")` for "not possible yet".
- **G-139 Word Break**: `dp[i]` = True when the first i letters can be split into words from the list (`dp[0]` = True).
- **G-91 Decode Ways**: like climbing stairs: read one digit or two digits at each step.
- **G-1143 Longest Common Subsequence**: a 2D table; `dp[i][j]` uses the first i letters of one string and the first j of the other.
