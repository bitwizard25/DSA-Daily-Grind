# Dynamic Programming

**Use when:**
- "number of ways", "minimum cost / coins / edits", "maximum profit / sum", "longest ... subsequence".
- "can you reach / partition / segment / match": a yes/no answer over many choices.
- A brute-force recursion repeats the same subproblems (**overlapping subproblems**).
- Two strings and "edit / match / common": 2D DP over **prefixes** of both.

## Python template(s)

### The 4-step recipe

| Step | Question | House Robber (G-198) |
|---|---|---|
| 1. **State** | What does `dp[i]` mean, in words? | `dp[i]` = most money from the first `i` houses |
| 2. **Transition** | How does it follow from smaller states? | `dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])` |
| 3. **Base + order** | Smallest states? Loop so inputs are ready first | `dp[0] = 0`, `dp[1] = nums[0]`; `i` ascending |
| 4. **Answer** | Which cell holds it? Then compress space | `dp[n]`; only 2 values are needed |

- Write step 1 as a comment before coding. Check every transition against it.
- Top-down (`@cache`) visits only reachable states; bottom-up fills every cell. Know both.

```python
import random
from functools import cache


def rob_top_down(nums):              # G-198
    @cache                           # nested: a fresh cache on every call of rob_top_down
    def best(i):                     # most money from houses i..n-1
        if i >= len(nums):
            return 0
        return max(best(i + 1), nums[i] + best(i + 2))

    return best(0)


def rob_table(nums):
    n = len(nums)
    dp = [0] * (n + 1)               # dp[i]: most money from the first i houses
    if n:
        dp[1] = nums[0]
    for i in range(2, n + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
    return dp[n]


def rob(nums):                       # dp[i] reads only dp[i-1] and dp[i-2]
    prev2 = prev1 = 0
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)   # right side is evaluated first
    return prev1


assert rob([1, 2, 3, 1]) == 4
assert rob([2, 7, 9, 3, 1]) == 12
assert rob([]) == rob_table([]) == rob_top_down([]) == 0
random.seed(198)
for _ in range(200):
    nums = [random.randint(0, 50) for _ in range(random.randint(0, 12))]
    assert rob(nums) == rob_table(nums) == rob_top_down(nums)
```

### Top-down: `@cache` vs `@lru_cache`

- `@functools.cache` (3.9+) equals `@lru_cache(maxsize=None)`: unbounded, never evicts.
- Bare `@lru_cache` has **maxsize=128** and evicts states. LCS of two 200-char strings: 28 thousand calls
  with `@cache`, over 18 million with bare `@lru_cache` (measured).
- Arguments are the key: they must be **hashable**. Cache on indices, not on lists or slices.

```python
from functools import cache, lru_cache


@cache
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


assert fib(90) == 2880067194370816120
assert fib.cache_info().misses == 91          # each of fib(0..90) computed once
assert fib.cache_info().maxsize is None
fib.cache_clear()
assert fib.cache_info().currsize == 0


@lru_cache                                    # no parentheses: maxsize=128
def bounded(n):
    return n


@lru_cache(maxsize=None)                      # same as @cache
def unbounded(n):
    return n


assert bounded.cache_info().maxsize == 128
assert unbounded.cache_info().maxsize is None


@cache
def total(xs):
    return sum(xs)


assert total((1, 2, 3)) == 6                  # tuple: hashable
try:
    total([1, 2, 3])
except TypeError:                             # unhashable type: 'list'
    unhashable = True
else:
    unhashable = False
assert unhashable

words = {"a"}


@cache
def can_build(s):                             # BUG: reads the global "words"
    return s == "" or any(s[:k] in words and can_build(s[k:]) for k in range(1, len(s) + 1))


assert can_build("aa")
words = {"b"}                                 # next test case
assert can_build("aa")                        # stale: the cache still says True
```

### 1D DP: Coin Change, Decode Ways, Maximum Product, Word Break

- `dp[0]` is the empty case: 0 coins, 1 way to decode "", the empty prefix can be segmented.
- Mark "impossible" with `math.inf`. Map it back to `-1` only at the end.

```python
import math


def coin_change(coins, amount):      # G-322: fewest coins, unlimited supply
    dp = [0] + [math.inf] * amount   # dp[a]: fewest coins summing to a
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return -1 if dp[amount] == math.inf else dp[amount]


assert coin_change([1, 2, 5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
assert coin_change([1, 3, 4], 6) == 2          # greedy 4 + 1 + 1 uses 3 coins


def num_decodings(s):                # G-91: "1" -> A ... "26" -> Z
    prev2, prev1 = 1, int(s[0] != "0")         # dp[0], dp[1]
    for i in range(2, len(s) + 1):
        cur = 0
        if s[i - 1] != "0":
            cur += prev1                       # last digit alone
        if "10" <= s[i - 2:i] <= "26":         # 2-char digit strings compare like numbers
            cur += prev2                       # last two digits
        prev2, prev1 = prev1, cur
    return prev1


assert num_decodings("12") == 2
assert num_decodings("226") == 3
assert num_decodings("06") == 0
assert num_decodings("2101") == 1
assert num_decodings("11106") == 2


def max_product(nums):               # G-152
    hi = lo = best = nums[0]         # largest and smallest product ending here
    for i in range(1, len(nums)):
        x = nums[i]
        if x < 0:
            hi, lo = lo, hi          # a negative turns the min into the max
        hi = max(x, hi * x)
        lo = min(x, lo * x)
        best = max(best, hi)
    return best


assert max_product([2, 3, -2, 4]) == 6
assert max_product([-2, 0, -1]) == 0
assert max_product([-2, 3, -4]) == 24
assert max_product([-2]) == -2


def word_break(s, word_dict):        # G-139
    words = set(word_dict)
    longest = max(map(len, words), default=0)
    ok = [True] + [False] * len(s)   # ok[i]: s[:i] can be segmented
    for i in range(1, len(s) + 1):
        for j in range(max(0, i - longest), i):
            if ok[j] and s[j:i] in words:
                ok[i] = True
                break
    return ok[len(s)]


assert word_break("leetcode", ["leet", "code"])
assert word_break("applepenapple", ["apple", "pen"])
assert not word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])
```

### 2D DP over prefixes: Edit Distance, LCS, Unique Paths, Regex

- `dp[i][j]` is about `a[:i]` and `b[:j]`. Size `(m + 1) x (n + 1)`: row 0 and column 0 are empty prefixes.
- Build rows with a comprehension. `[[0] * (n + 1)] * (m + 1)` repeats **one** row object.

```python
import math
import random
import re
from functools import cache


def unique_paths(m, n):              # G-62
    dp = [[1] * n for _ in range(m)] # first row and first column: one way each
    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
    return dp[m - 1][n - 1]


assert unique_paths(3, 7) == 28 == math.comb(8, 2)
assert unique_paths(100, 100) == math.comb(198, 99)   # 59 digits: Python ints never overflow


def edit_distance(a, b):             # G-72
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]   # dp[i][j]: edits to turn a[:i] into b[:j]
    for i in range(m + 1):
        dp[i][0] = i                 # delete all i chars
    for j in range(n + 1):
        dp[0][j] = j                 # insert all j chars
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],       # delete a[i-1]
                                   dp[i][j - 1],       # insert b[j-1]
                                   dp[i - 1][j - 1])   # replace
    return dp[m][n]


assert edit_distance("horse", "ros") == 3
assert edit_distance("intention", "execution") == 5
assert edit_distance("", "abc") == 3


def lcs(a, b):                       # G-1143
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def lcs_memo(a, b):                  # same states, top-down
    @cache
    def go(i, j):                    # LCS of a[i:] and b[j:]: indices, not slices
        if i == len(a) or j == len(b):
            return 0
        if a[i] == b[j]:
            return 1 + go(i + 1, j + 1)
        return max(go(i + 1, j), go(i, j + 1))

    return go(0, 0)


assert lcs("abcde", "ace") == lcs_memo("abcde", "ace") == 3
assert lcs("abc", "def") == 0

bad = [[0] * 3] * 2
bad[0][0] = 9
assert bad == [[9, 0, 0], [9, 0, 0]]  # both "rows" changed


def is_match(s, p):                  # G-10: "." any char, "x*" zero or more x
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]   # dp[i][j]: s[:i] matches p[:j]
    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 2]  # "a*b*" matches ""
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                zero = dp[i][j - 2]                                # use "x*" zero times
                more = p[j - 2] in (s[i - 1], ".") and dp[i - 1][j]  # x* eats s[i-1]
                dp[i][j] = zero or more
            else:
                dp[i][j] = p[j - 1] in (s[i - 1], ".") and dp[i - 1][j - 1]
    return dp[m][n]


assert not is_match("aa", "a")
assert is_match("aa", "a*")
assert is_match("ab", ".*")
assert is_match("aab", "c*a*b")
assert not is_match("mississippi", "mis*is*p*.")
assert is_match("", "a*b*")

random.seed(10)
for _ in range(500):
    s = "".join(random.choice("ab") for _ in range(random.randint(0, 6)))
    p = "".join(random.choice("ab.") + random.choice(["", "*"]) for _ in range(random.randint(0, 4)))
    assert is_match(s, p) == (re.fullmatch(p, s) is not None)
```

### Knapsack: 0/1 vs unbounded loop direction

| Variant | Each item | Loops | Example |
|---|---|---|---|
| **0/1** | at most once | capacity **descending**: `range(cap, w - 1, -1)` | G-416 Partition Equal Subset Sum |
| **Unbounded** | any number of times | capacity **ascending**: `range(w, cap + 1)` | G-322 Coin Change |
| Count **combinations** | any number of times | items outer, capacity inner (ascending) | Coin Change II |
| Count **sequences** (order matters) | any number of times | capacity outer, items inner | Combination Sum IV |

- Descending reads `dp[c - w]` **before** this item's pass updates it: the item is used once.
- Ascending reads a value this pass already updated: the item can be reused.

```python
import itertools
import random


def can_partition(nums):             # G-416: 0/1 knapsack over sums
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    reach = [True] + [False] * target            # reach[s]: some subset sums to s
    for x in nums:
        for s in range(target, x - 1, -1):       # DOWN: x used at most once
            if reach[s - x]:
                reach[s] = True
    return reach[target]


def can_partition_wrong(nums):       # BUG: upward loop
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    reach = [True] + [False] * target
    for x in nums:
        for s in range(x, target + 1):           # UP: reach[s - x] may already use x
            if reach[s - x]:
                reach[s] = True
    return reach[target]


assert can_partition([1, 5, 11, 5])
assert not can_partition([1, 2, 3, 5])
assert not can_partition([1, 2, 5])              # sum 8: no subset makes 4
assert can_partition_wrong([1, 2, 5])            # wrong: 2 + 2 uses the 2 twice


def can_partition_bits(nums):        # a Python int as a growable bitset
    total = sum(nums)
    bits = 1                         # bit s set: sum s is reachable
    for x in nums:
        bits |= bits << x            # every old sum, plus x
    return total % 2 == 0 and (bits >> (total // 2)) & 1 == 1


assert can_partition_bits([1, 5, 11, 5]) and not can_partition_bits([1, 2, 5])
assert (6 & 1 == 0) is True          # Python: (6 & 1) == 0. C++ parses 6 & (1 == 0).


def knapsack_01(weights, values, cap):
    best = [0] * (cap + 1)           # best[c]: max value with capacity c
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):
            best[c] = max(best[c], best[c - w] + v)
    return best[cap]


def knapsack_unbounded(weights, values, cap):
    best = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(w, cap + 1):
            best[c] = max(best[c], best[c - w] + v)
    return best[cap]


assert knapsack_01([1, 3, 4], [15, 20, 30], 4) == 35         # items 0 and 1
assert knapsack_unbounded([1, 3, 4], [15, 20, 30], 4) == 60  # item 0 four times

random.seed(416)
for _ in range(200):
    n = random.randint(0, 6)
    ws = [random.randint(1, 6) for _ in range(n)]
    vs = [random.randint(0, 20) for _ in range(n)]
    cap = random.randint(0, 12)
    brute = max(sum(vs[i] for i in pick)
                for k in range(n + 1) for pick in itertools.combinations(range(n), k)
                if sum(ws[i] for i in pick) <= cap)
    assert knapsack_01(ws, vs, cap) == brute
    nums = [random.randint(1, 9) for _ in range(n)]
    assert can_partition(nums) == can_partition_bits(nums)


def count_combinations(coins, amount):           # Coin Change II
    ways = [1] + [0] * amount
    for c in coins:                              # items outer: each multiset once
        for a in range(c, amount + 1):
            ways[a] += ways[a - c]
    return ways[amount]


def count_sequences(coins, amount):              # Combination Sum IV
    ways = [1] + [0] * amount
    for a in range(1, amount + 1):               # capacity outer: every order counts
        for c in coins:
            if c <= a:
                ways[a] += ways[a - c]
    return ways[amount]


assert count_combinations([1, 2, 5], 5) == 4     # 5, 2+2+1, 2+1+1+1, 1+1+1+1+1
assert count_sequences([1, 2, 5], 5) == 9        # 1+2+2, 2+1+2, 2+2+1 are 3 different ones
```

### Interval DP: Burst Balloons

- `dp[l][r]` = best result for the open interval `(l, r)`. Fill by **increasing length**.
- Choose `k` = the **last** balloon burst in `(l, r)`. Its neighbours are then fixed: `l` and `r`.
- Choosing the **first** balloon fails: its neighbours change, so the two sides are not independent.

```python
import itertools
import random
from functools import cache


def max_coins(nums):                 # G-312
    vals = [1] + nums + [1]          # pad: bursting at an edge multiplies by 1
    n = len(vals)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):       # length = r - l; shorter intervals first
        for left in range(n - length):
            right = left + length
            dp[left][right] = max(
                dp[left][k] + vals[left] * vals[k] * vals[right] + dp[k][right]
                for k in range(left + 1, right))
    return dp[0][n - 1]


def max_coins_memo(nums):
    vals = [1] + nums + [1]

    @cache
    def best(left, right):
        return max((best(left, k) + vals[left] * vals[k] * vals[right] + best(k, right)
                    for k in range(left + 1, right)), default=0)   # empty interval: 0

    return best(0, len(vals) - 1)


def max_coins_brute(nums):           # try every burst order
    best = 0
    for order in itertools.permutations(range(len(nums))):
        alive, coins = list(range(len(nums))), 0
        for i in order:
            p = alive.index(i)
            left = nums[alive[p - 1]] if p > 0 else 1
            right = nums[alive[p + 1]] if p + 1 < len(alive) else 1
            coins += left * nums[i] * right
            alive.pop(p)
        best = max(best, coins)
    return best


assert max_coins([3, 1, 5, 8]) == max_coins_memo([3, 1, 5, 8]) == 167
assert max_coins([1, 5]) == 10
assert max_coins([]) == 0
random.seed(312)
for _ in range(60):
    nums = [random.randint(0, 9) for _ in range(random.randint(0, 6))]
    assert max_coins(nums) == max_coins_memo(nums) == max_coins_brute(nums)
```

### LIS in O(N log N) with `bisect`

- `tails[k]` = smallest tail of any increasing subsequence of length `k + 1`. `tails` stays sorted.
- Strictly increasing: `bisect_left`. Non-decreasing: `bisect_right`.
- `tails` gives the **length**, not the subsequence. Rebuild the sequence with parent links.

```python
import bisect
import random


def length_of_lis(nums):             # G-300
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)   # first tail >= x (std::lower_bound)
        if i == len(tails):
            tails.append(x)          # x extends the longest run
        else:
            tails[i] = x             # same length, smaller tail
    return len(tails)


def longest_non_decreasing(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_right(tails, x)  # first tail > x (std::upper_bound)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def lis_quadratic(nums):             # O(N^2) reference
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp, default=0)        # max([]) raises ValueError


def lis_sequence(nums):              # one actual LIS
    tail_vals, tail_idx = [], []
    parent = [-1] * len(nums)
    for i, x in enumerate(nums):
        k = bisect.bisect_left(tail_vals, x)
        if k > 0:
            parent[i] = tail_idx[k - 1]    # previous element of this run
        if k == len(tail_vals):
            tail_vals.append(x)
            tail_idx.append(i)
        else:
            tail_vals[k] = x
            tail_idx[k] = i
    out = []
    i = tail_idx[-1] if tail_idx else -1
    while i != -1:
        out.append(nums[i])
        i = parent[i]
    return out[::-1]


def is_subsequence(sub, seq):
    it = iter(seq)
    return all(x in it for x in sub)       # "in" advances the iterator


assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert length_of_lis([7, 7, 7, 7]) == 1
assert longest_non_decreasing([7, 7, 7, 7]) == 4
assert length_of_lis([]) == 0

tails = []
for x in [3, 4, 1]:
    i = bisect.bisect_left(tails, x)
    tails[i:i + 1] = [x]             # replace, or append when i == len(tails)
assert tails == [1, 4]               # length 2 is right, but 1 comes after 4 in the input

random.seed(300)
for _ in range(300):
    nums = [random.randint(0, 10) for _ in range(random.randint(0, 12))]
    seq = lis_sequence(nums)
    assert length_of_lis(nums) == lis_quadratic(nums) == len(seq)
    assert all(a < b for a, b in zip(seq, seq[1:])) and is_subsequence(seq, nums)


def job_scheduling(start, end, profit):      # G-1235: DP + binary search
    jobs = sorted(zip(end, start, profit))   # by end time
    ends = [e for e, _, _ in jobs]
    dp = [0] * (len(jobs) + 1)               # dp[i]: best profit from the first i jobs
    for i, (e, s, p) in enumerate(jobs):
        j = bisect.bisect_right(ends, s, 0, i)   # jobs that end at or before s
        dp[i + 1] = max(dp[i], dp[j] + p)        # skip job i, or take it
    return dp[-1]


assert job_scheduling([1, 2, 3, 3], [3, 4, 5, 6], [50, 10, 40, 70]) == 120
assert job_scheduling([1, 2, 3, 4, 6], [3, 5, 10, 6, 9], [20, 20, 100, 70, 60]) == 150
assert job_scheduling([1, 1, 1], [2, 3, 4], [5, 6, 4]) == 6
```

The same 0/1 knapsack, bitset and LIS in C++:

```cpp
#include <algorithm>
#include <bitset>
#include <cassert>
#include <vector>

bool canPartition(const std::vector<int>& nums) {
    int total = 0;
    for (int x : nums) total += x;
    if (total % 2) return false;
    int target = total / 2;
    std::vector<bool> reach(target + 1, false);
    reach[0] = true;
    for (int x : nums)
        for (int s = target; s >= x; --s)        // descending: 0/1
            if (reach[s - x]) reach[s] = true;
    return reach[target];
}

bool canPartitionBits(const std::vector<int>& nums) {
    std::bitset<20001> bits;                      // size fixed at compile time
    bits[0] = 1;
    int total = 0;
    for (int x : nums) {
        bits |= bits << x;
        total += x;
    }
    return total % 2 == 0 && bits[total / 2];
}

int lengthOfLIS(const std::vector<int>& nums) {
    std::vector<int> tails;
    for (int x : nums) {
        auto it = std::lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) tails.push_back(x);
        else *it = x;
    }
    return static_cast<int>(tails.size());
}

int main() {
    assert(canPartition({1, 5, 11, 5}));
    assert(!canPartition({1, 2, 5}));
    assert(canPartitionBits({1, 5, 11, 5}));
    assert(!canPartitionBits({1, 2, 3, 5}));
    assert(lengthOfLIS({10, 9, 2, 5, 3, 7, 101, 18}) == 4);
    assert(lengthOfLIS({7, 7, 7}) == 1);
    assert((6 & (1 == 0)) == 0);                  // C++ reads "6 & 1 == 0" this way
    return 0;
}
```

### Space compression

| `dp[i]` reads | Keep | Example |
|---|---|---|
| `dp[i-1]`, `dp[i-2]` | two variables | House Robber, Decode Ways |
| the previous row only | two rows `prev`, `cur` | LCS, Unique Paths |
| previous row + the diagonal | one row + a `diag` variable | Edit Distance |
| previous row, each item once | one row, loop **descending** | 0/1 knapsack |

- Put the **shorter** string on the row: O(min(M, N)) space.
- A compressed table cannot rebuild the answer path. Keep the full table if you must print it.

```python
import random
from functools import cache


def lcs_two_rows(a, b):
    if len(a) < len(b):
        a, b = b, a                  # LCS is symmetric: keep the row short
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)     # a new list each row
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur                   # rebinds the name; no copy needed
    return prev[-1]


def edit_distance_one_row(a, b):
    row = list(range(len(b) + 1))    # dp[0][j] = j
    for i in range(1, len(a) + 1):
        diag, row[0] = row[0], i     # diag = dp[i-1][0]; row[0] = dp[i][0]
        for j in range(1, len(b) + 1):
            above = row[j]           # dp[i-1][j], about to be overwritten
            if a[i - 1] == b[j - 1]:
                row[j] = diag
            else:
                row[j] = 1 + min(above, row[j - 1], diag)
            diag = above             # dp[i-1][j] is the next j's diagonal
    return row[-1]


@cache
def lcs_ref(a, b, i=0, j=0):         # reference answers, top-down (short strings only)
    if i == len(a) or j == len(b):
        return 0
    if a[i] == b[j]:
        return 1 + lcs_ref(a, b, i + 1, j + 1)
    return max(lcs_ref(a, b, i + 1, j), lcs_ref(a, b, i, j + 1))


@cache
def edit_ref(a, b, i=0, j=0):        # edits to turn a[i:] into b[j:]
    if i == len(a) or j == len(b):
        return (len(a) - i) + (len(b) - j)
    if a[i] == b[j]:
        return edit_ref(a, b, i + 1, j + 1)
    return 1 + min(edit_ref(a, b, i + 1, j), edit_ref(a, b, i, j + 1), edit_ref(a, b, i + 1, j + 1))


assert lcs_two_rows("abcde", "ace") == 3
assert edit_distance_one_row("horse", "ros") == 3
random.seed(72)
for _ in range(300):
    a = "".join(random.choice("abc") for _ in range(random.randint(0, 8)))
    b = "".join(random.choice("abc") for _ in range(random.randint(0, 8)))
    assert lcs_two_rows(a, b) == lcs_ref(a, b)
    assert edit_distance_one_row(a, b) == edit_ref(a, b)
```

### When `@cache` hits the recursion limit

- Memo recursion depth = the longest chain of nested calls. LCS top-down: up to `len(a) + len(b)`.
- Default limit is **1000** frames. Raising it does not reliably help `@cache` (table below).
- Fixes: **bottom-up** (best), or **warm the cache** in increasing order so each call recurses 1 level.

| Recursion style | Default limit (1000) | After `sys.setrecursionlimit(10**6)` (measured) |
|---|---|---|
| plain recursion, dict memo | RecursionError near depth 1000 | works; tested to 200,000 on 3.11, 3.12, 3.13 |
| `@cache` / `@lru_cache` | RecursionError near depth 1000 | 3.12.3, 3.13.16: RecursionError near 3,300 |
| `@cache` / `@lru_cache` | RecursionError near depth 1000 | 3.11: segfault (crashed at 20,000) |

- The `@cache` wrapper is C code. 3.12+ caps C-level recursion on its own; 3.11 can overflow the C stack.
- Recursing inside `max(gen)` or `sum(gen)` also passes through C: a similar cap of a few thousand levels.

```python
import sys
from functools import cache

assert sys.getrecursionlimit() == 1000


def stairs_memo(n):                  # ways to climb n steps, 1 or 2 at a time
    @cache
    def ways(i):
        return 1 if i <= 1 else ways(i - 1) + ways(i - 2)

    return ways(n)


def raises_recursion_error(fn, *args):
    try:
        fn(*args)
    except RecursionError:
        return True
    return False


assert stairs_memo(10) == 89
assert raises_recursion_error(stairs_memo, 5000)


def stairs_table(n):                 # fix 1: bottom-up, O(1) space
    a, b = 1, 1                      # ways(0), ways(1)
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def stairs_warm(n):                  # fix 2: fill the cache from small to large
    @cache
    def ways(i):
        return 1 if i <= 1 else ways(i - 1) + ways(i - 2)

    for i in range(n + 1):
        ways(i)                      # ways(i - 1) and ways(i - 2) are already cached
    return ways(n)


assert stairs_table(10) == 89 and stairs_table(1) == 1 and stairs_table(0) == 1
assert stairs_table(5000) == stairs_warm(5000)


def lcs_memo(a, b):
    @cache
    def go(i, j):
        if i == len(a) or j == len(b):
            return 0
        if a[i] == b[j]:
            return 1 + go(i + 1, j + 1)
        return max(go(i + 1, j), go(i, j + 1))

    return go(0, 0)


assert lcs_memo("ab" * 100, "ba" * 100) == 199
assert raises_recursion_error(lcs_memo, "a" * 600, "b" * 600)   # depth ~1200 > 1000
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `vector<int> dp(n + 1, 0)` | `dp = [0] * (n + 1)` | O(N) |
| `vector<vector<int>> dp(m + 1, vector<int>(n + 1))` | `[[0] * (n + 1) for _ in range(m + 1)]` | O(M * N) |
| `memset(memo, -1, sizeof memo)` + recursion | `@cache` on a nested function | average O(1) per lookup |
| `unordered_map` memo keyed by a packed state | `@cache` with int / tuple args | average O(1) per lookup |
| `INT_MAX` / `1e9` as "impossible" | `math.inf` (a float) | O(1) |
| `long long` / `% MOD` to avoid overflow | plain `int`; `% MOD` only when asked | big ints: cost grows with digits |
| `std::lower_bound(tails.begin(), tails.end(), x)` | `bisect.bisect_left(tails, x)` | O(log N) |
| `std::upper_bound` | `bisect.bisect_right` | O(log N) |
| `std::bitset<N> b; b \|= b << x;` | `bits \|= bits << x` on an `int` (any size) | O(S / 30) per item: 30-bit digits |
| `std::max({a, b, c})` | `max(a, b, c)` | O(1) |
| `*max_element(dp.begin(), dp.end())` | `max(dp)` (`max(dp, default=0)` if empty) | O(N) |
| `swap(prev, cur)` | `prev, cur = cur, prev` | O(1) |
| `int t = a; a = b; b = t + b;` | `a, b = b, a + b` | O(1) |

## Complexity

| Problem / technique | Time | Space |
|---|---|---|
| Top-down `@cache` | O(states x work per state) | O(states) cache + O(depth) stack |
| House Robber, Decode Ways, Max Product | O(N) | O(1) |
| Word Break (words up to length L) | O(N * L^2): N * L slices, each O(L) | O(N) |
| Coin Change (amount A, C coins) | O(A * C) | O(A) |
| 0/1 or unbounded knapsack (N items, capacity S) | O(N * S) | O(S) |
| Partition via int bitset | O(N * S / 30) word ops | O(S) bits |
| Edit Distance, LCS, Regex (M x N) | O(M * N) | O(M * N), compressed O(min(M, N)) |
| Unique Paths | O(M * N), or `math.comb` | O(M * N), compressed O(N) |
| Burst Balloons (interval DP) | O(N^3) | O(N^2) |
| LIS with `bisect` | O(N log N) | O(N) |
| LIS quadratic DP | O(N^2) | O(N) |
| Job Scheduling (sort + DP + bisect) | O(N log N) | O(N) |

## Gotchas for C++ developers

- `[[0] * n] * m` repeats **one** row: writing `dp[0][0]` changes every row.
- Bare `@lru_cache` keeps only **128** entries. Use `@cache` or `@lru_cache(maxsize=None)`.
- Cached arguments must be **hashable**: lists raise **TypeError**. Pass indices or tuples.
- Slices as cache keys copy O(N) per call. Use indices `i, j` into the original string.
- A module-level `@cache` that reads outside data returns **stale** answers on the next test case.
- `@cache` on a method adds `self` to the key and keeps every instance alive. Cache a nested function.
- Recursion limit is **1000**. `sys.setrecursionlimit` does not reliably lift the `@cache` cap: go bottom-up.
- 0/1 knapsack loops capacity **down**; unbounded loops **up**. `range(cap, w - 1, -1)` stops before `w - 1`.
- Counting: items outer = **combinations**; capacity outer = **sequences** (order counts).
- Interval DP picks the **last** action in `(l, r)`, and fills by increasing length.
- `math.inf` is a float. `int(math.inf)` raises **OverflowError**: map it to `-1` first.
- Ints never overflow, so "number of ways" can grow huge and slow. Apply `% MOD` each step when asked.
- Python `%` with a positive modulus is never negative: `-3 % 5 == 2` (C++ gives `-3`).
- `x & 1 == 0` means `(x & 1) == 0` in Python; C++ parses it as `x & (1 == 0)`.
- LIS: `bisect_left` for strictly increasing, `bisect_right` for non-decreasing. `tails` is not the LIS.
- `max([])` raises **ValueError**. Use `max(dp, default=0)`.
- `a, b = b, a + b` evaluates the whole right side first. No temp variable needed.

## In the Google bank

- **1D:** G-198 House Robber, G-91 Decode Ways, G-139 Word Break, G-152 Maximum Product Subarray.
- **Unbounded knapsack:** G-322 Coin Change.
- **0/1 knapsack:** G-416 Partition Equal Subset Sum.
- **2D over prefixes:** G-72 Edit Distance, G-1143 Longest Common Subsequence, G-10 Regular Expression Matching.
- **Grid paths:** G-62 Unique Paths.
- **Interval DP:** G-312 Burst Balloons; G-5 Longest Palindromic Substring (expand-around-centre is simpler).
- **LIS / DP + binary search:** G-300 Longest Increasing Subsequence, G-1235 Maximum Profit in Job Scheduling.
- **Memo DFS on a DAG (note 09):** G-329 Longest Increasing Path in a Matrix.
- **Tagged DP, solved otherwise:** G-410 Split Array Largest Sum (binary search on the answer),
  G-42 Trapping Rain Water (two pointers or a monotonic stack).
