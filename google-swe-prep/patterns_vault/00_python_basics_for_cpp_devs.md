# Python Basics for C++ Devs

**Use when:**
- You reach for `INT_MAX`, `/`, `%`, `>>` or `~` while writing Python.
- The problem mentions **negative numbers** with division or modulo, **32-bit** limits, or **truncate toward zero**.
- You build a **2D grid**, copy a list, or pass a list into a function.
- You build a string **one character at a time**.

## Python template(s)

### Integers and limits

- `int` has **arbitrary precision**: no overflow, no `long long`.
- `math.inf` replaces `INT_MAX` as a sentinel; it compares with any int.

```python
import math
import sys

x = 2 ** 100
assert x == 1267650600228229401496703205376      # exact
assert x.bit_length() == 101
assert (2**63 - 1) + 1 == 2**63                   # LLONG_MAX + 1 is just a bigger int

assert (0b1011).bit_count() == 3                  # __builtin_popcount, 3.10+
assert (-1).bit_count() == 1                      # counts abs(x); C++ popcount(-1) is 32
assert (-1 & 0xFFFFFFFF).bit_count() == 32        # mask first for the 32-bit view
assert (1000).bit_length() == 10                  # 64 - __builtin_clzll(1000)
assert (0).bit_length() == 0

# 32-bit limits are checked by hand (Reverse Integer)
INT_MIN, INT_MAX = -2**31, 2**31 - 1

def reverse_int(n):
    r = int(str(abs(n))[::-1]) * (-1 if n < 0 else 1)
    return r if INT_MIN <= r <= INT_MAX else 0

assert reverse_int(-123) == -321
assert reverse_int(1534236469) == 0               # 9646324351 > INT_MAX

# Sentinels
assert -math.inf < -10**100 < 10**100 < math.inf
assert min(math.inf, 7) == 7 and type(min(math.inf, 7)) is int
assert math.inf + 1 == math.inf                   # dist + w cannot overflow
assert math.isnan(math.inf - math.inf)

# sys.maxsize is the largest container size, not an int limit
assert sys.maxsize + 1 > sys.maxsize

# int <-> str conversion is capped at 4300 digits by default (3.11+)
try:
    str(10 ** 5000)
    capped = False
except ValueError:
    capped = True
assert capped
```

### Division and modulo

- `//` **floors**, `%` takes the **divisor's** sign. C++ truncates and `%` takes the dividend's sign.
- `/` always returns a `float`.

```python
import math

assert 7 / 2 == 3.5 and type(6 / 2) is float      # / always returns float
assert -7 // 2 == -4 and -7 % 2 == 1              # C++: -3 and -1
assert 7 // -2 == -4 and 7 % -2 == -1             # C++: -3 and 1
for a in (-7, 7):
    for b in (-2, 2):
        assert (a // b) * b + a % b == a          # same invariant as C++

assert divmod(-7, 2) == (-4, 1)
assert -7 % 5 == 3                                # no ((a % m) + m) % m fix-up

# Truncate toward zero like C++
assert int(-7 / 2) == -3                          # float path: exact while abs(a) < 2**53

def trunc_div(a, b):                              # integer-only, exact at any size
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q

assert trunc_div(-7, 2) == -3 and trunc_div(7, -2) == -3 and trunc_div(-8, -2) == 4
assert trunc_div(-(10**30) - 1, 10) == -(10**29)

# Evaluate Reverse Polish Notation (LeetCode 150)
def eval_rpn(tokens):
    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: int(a / b),             # NOT a // b
    }
    stack = []
    for t in tokens:
        if t in ops:
            b = stack.pop()
            a = stack.pop()
            stack.append(ops[t](a, b))
        else:
            stack.append(int(t))
    return stack[0]

assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
assert eval_rpn(["6", "-132", "/"]) == 0          # a // b would give -1

# Ceiling division without floats (b > 0)
assert -(-7 // 2) == 4 and (7 + 2 - 1) // 2 == 4

# Rounding
assert int(-3.7) == -3 and math.trunc(-3.7) == -3
assert math.floor(-3.7) == -4 and math.ceil(-3.2) == -3
assert round(2.5) == 2 and round(3.5) == 4        # half to even
assert 0.1 + 0.2 != 0.3 and math.isclose(0.1 + 0.2, 0.3)
```

C++ side by side:

```cpp
#include <cassert>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

int main() {
    // Division truncates toward zero; % takes the dividend's sign.
    assert(-7 / 2 == -3 && -7 % 2 == -1);
    // A non-negative modulo needs a fix-up that Python's % does for you.
    assert(((-7 % 5) + 5) % 5 == 3);
    // Unsigned 32-bit wraparound is defined; Python needs & 0xFFFFFFFF.
    uint32_t u = 0;
    u -= 1;
    assert(u == 4294967295u);
    // Signed overflow is undefined: test before adding instead.
    int x = INT_MAX;
    bool add_one_overflows = x > INT_MAX - 1;
    assert(add_one_overflows);
    // Nested vectors copy every row: no aliasing.
    std::vector<std::vector<int>> grid(3, std::vector<int>(2, 0));
    grid[0][0] = 7;
    assert(grid[1][0] == 0);
    // std::string is mutable and push_back is amortized O(1).
    std::string s = "google";
    s[0] = 'G';
    s.push_back('!');
    assert(s == "Google!");
    // substr takes (pos, count); a Python slice takes (start, end).
    assert(std::string("interview").substr(2, 3) == "ter");
    return 0;
}
```

### Bits and 32-bit masks

- Ints act like **infinite two's complement**: `~x == -(x + 1)`, `>>` floors.
- Emulate 32 bits with `& 0xFFFFFFFF`, then convert back to signed.

```python
MASK = 0xFFFFFFFF

assert ~5 == -6 and ~-1 == 0 and ~0 == -1         # ~x == -(x + 1)
assert -7 >> 1 == -4 == -7 // 2                   # arithmetic shift floors
assert -1 >> 100 == -1
assert 1 << 64 == 18446744073709551616            # no overflow, no UB

def to_signed32(x):
    x &= MASK
    return x - (1 << 32) if x >= 1 << 31 else x

assert -1 & MASK == 4294967295                    # (uint32_t)-1
assert ~5 & MASK == 4294967290                    # (uint32_t)~5
assert to_signed32(4294967295) == -1
assert to_signed32(2**31) == -2**31               # wraps like int32_t

# Sum of Two Integers (LeetCode 371): masks make the carry die out
def get_sum(a, b):
    a, b = a & MASK, b & MASK
    while b:
        a, b = (a ^ b) & MASK, ((a & b) << 1) & MASK
    return to_signed32(a)

assert get_sum(-3, 5) == 2 and get_sum(-8, -9) == -17 and get_sum(1, 2) == 3

def unmasked_steps(a, b, limit=100):              # same loop, no masks
    for step in range(limit):
        if not b:
            return step
        a, b = a ^ b, (a & b) << 1
    return None

assert unmasked_steps(-3, 5) is None              # still carrying: loops forever
assert unmasked_steps(-8, -9) is not None         # some negative inputs do finish

# Logical right shift of a 32-bit value: mask first
assert (-16 & MASK) >> 28 == 0xF

# Classic tricks carry over
x = 0b101100
assert x & -x == 0b100                            # lowest set bit
assert x & (x - 1) == 0b101000                    # clear lowest set bit
assert (x >> 2) & 1 == 1                          # test bit 2
assert x | (1 << 0) == 0b101101                   # set bit 0
assert x & 1 == 0                                 # (x & 1) == 0; C++ reads x & (1 == 0)
assert bin(x) == "0b101100" and int("101100", 2) == x
assert format(5, "08b") == "00000101"
```

### Strings

- `str` is **immutable**. Edit through `list(s)`; build with a list and `"".join`.
- No `char` type: `s[i]` is a 1-char `str`. Use `ord` / `chr` for arithmetic.

```python
s = "google"
try:
    s[0] = "G"
except TypeError:
    pass
else:
    raise AssertionError("str should be immutable")

t = "G" + s[1:]                    # new string, O(N)
assert t == "Google" and s == "google"

chars = list(s)                    # mutable copy, O(N)
chars[0], chars[-1] = chars[-1], chars[0]
assert "".join(chars) == "eooglg"

def compress(word):                # "aaabcc" -> "a3b1c2", O(N) with list + join
    out = []
    i = 0
    while i < len(word):
        j = i
        while j < len(word) and word[j] == word[i]:
            j += 1
        out.append(word[i])
        out.append(str(j - i))
        i = j
    return "".join(out)

assert compress("aaabcc") == "a3b1c2"

c = s[0]
assert type(c) is str and len(c) == 1
assert ord(c) - ord("a") == 6 and chr(ord("a") + 6) == "g"
assert "a" <= c <= "z" and c.isalpha() and not c.isdigit()

cnt = [0] * 26                     # int cnt[26]
for ch in s:
    cnt[ord(ch) - ord("a")] += 1
assert cnt[ord("o") - ord("a")] == 2

assert " a  b ".split() == ["a", "b"]             # whitespace split drops empties
assert "a,b,,c".split(",") == ["a", "b", "", "c"]
assert "-".join(["2026", "10", "12"]) == "2026-10-12"
assert "abcabc".find("c") == 2 and "abc".find("z") == -1
```

### Slicing

- `s[start:end:step]`: `end` is an **exclusive index**, not a length.
- Every slice is an **O(K) copy**. In hot loops pass `l`, `r` indexes.

```python
s = "interview"
assert s[2:5] == "ter"              # indexes 2, 3, 4
assert s[2:2 + 3] == "ter"          # C++ s.substr(2, 3)
assert s[:5] == "inter" and s[5:] == "view"
assert s[-4:] == "view" and s[-1] == "w"
assert s[::-1] == "weivretni" and s[::2] == "itriw"
assert s[100:] == "" and s[2:100] == "terview"    # slices clamp
try:
    s[100]
except IndexError:
    pass
else:
    raise AssertionError("indexing does not clamp")

a = [0, 1, 2, 3, 4, 5]
assert a[1:5:2] == [1, 3]
assert a[::-1] == [5, 4, 3, 2, 1, 0] and a == [0, 1, 2, 3, 4, 5]
a[1:3] = [9, 9, 9]                  # slice assignment can change the length
assert a == [0, 9, 9, 9, 3, 4, 5]
result = a.reverse()                # in place, returns None
assert result is None and a[0] == 5

def longest_palindrome(t):          # expand around centres with indexes
    best_l, best_r = 0, 0
    for center in range(2 * len(t) - 1):
        l, r = center // 2, center // 2 + center % 2
        while l >= 0 and r < len(t) and t[l] == t[r]:
            l -= 1
            r += 1
        if r - l - 1 > best_r - best_l:
            best_l, best_r = l + 1, r
    return t[best_l:best_r]         # slice once, at the end

assert longest_palindrome("babad") in ("bab", "aba")
assert longest_palindrome("cbbd") == "bb"
assert longest_palindrome("a") == "a"
```

### References, grids and default arguments

- Every name is a **reference**. `b = a` never copies.
- `[[0] * c] * r` shares one row. Build rows in a comprehension.

```python
a = [1, 2, 3]
b = a                               # another name for the same list
b.append(4)
assert a == [1, 2, 3, 4] and a is b

b = b + [5]                         # + builds a NEW list and rebinds b
assert a == [1, 2, 3, 4]
c = a
c += [5]                            # += on a list extends IN PLACE
assert a == [1, 2, 3, 4, 5]

def push(lst, x):
    lst.append(x)                   # caller sees the mutation

def rebind(lst):
    lst = [0]                       # only rebinds the local name

nums = [1]
push(nums, 2)
rebind(nums)
assert nums == [1, 2]

rows, cols = 3, 2
bad = [[0] * cols] * rows
bad[0][0] = 7
assert bad == [[7, 0], [7, 0], [7, 0]]
assert all(row is bad[0] for row in bad)

good = [[0] * cols for _ in range(rows)]
good[0][0] = 7
assert good == [[7, 0], [0, 0], [0, 0]]

counts = [0] * 3                    # safe: ints are immutable
counts[0] += 1
assert counts == [1, 0, 0]

buckets = [[]] * 2                  # same trap with any mutable item
buckets[0].append("x")
assert buckets == [["x"], ["x"]]

def collect_bad(x, seen=[]):        # default built once, at def time
    seen.append(x)
    return seen

assert collect_bad(1) == [1]
assert collect_bad(2) == [1, 2]     # leaked from the first call

def collect_good(x, seen=None):
    if seen is None:
        seen = []
    seen.append(x)
    return seen

assert collect_good(1) == [1] and collect_good(2) == [2]

def count_nodes(n):                 # rebinding an outer name needs nonlocal
    total = 0
    def dfs(i):
        nonlocal total
        if i >= n:
            return
        total += 1
        dfs(2 * i + 1)
        dfs(2 * i + 2)
    dfs(0)
    return total

assert count_nodes(6) == 6
```

### Copies and equality

- `a[:]`, `a.copy()`, `list(a)` are **shallow**. `copy.deepcopy` copies every level.
- `==` compares values; `is` compares identity. Use `is` only for `None`.

```python
import copy

grid = [[1, 2], [3, 4]]
shallow = grid[:]                   # same as grid.copy() or list(grid)
assert shallow == grid and shallow is not grid
assert shallow[0] is grid[0]        # rows are shared
shallow[0][0] = 9
assert grid[0][0] == 9

grid = [[1, 2], [3, 4]]
rows = [row[:] for row in grid]     # fast 2D copy of a numeric grid
deep = copy.deepcopy(grid)          # every level, slower
rows[0][0] = 8
deep[1][1] = 8
assert grid == [[1, 2], [3, 4]]

a, b = [1, [2, 3]], [1, [2, 3]]
assert a == b and a is not b        # == recurses into nested lists
assert (1, 2) < (1, 3) and [1, 2] < [1, 2, 0]   # lexicographic, like std::vector

node = None
assert node is None and (node is not None) is False
assert 1 == 1.0 and hash(1) == hash(1.0)        # 1 and 1.0 are the same dict key
```

### Loops and idioms

- `range`, `enumerate`, `zip`, `reversed` replace index bookkeeping.
- Empty containers are **falsy**: `while stack:`.

```python
a = [3, 1, 4]

assert list(range(1, 7, 2)) == [1, 3, 5]
assert list(range(4, -1, -2)) == [4, 2, 0]        # for (i = 4; i >= 0; i -= 2)
assert list(reversed(range(3))) == [2, 1, 0]

assert list(enumerate(a)) == [(0, 3), (1, 1), (2, 4)]
assert list(enumerate(a, start=1))[0] == (1, 3)
assert list(zip(a, "xy")) == [(3, "x"), (1, "y")] # stops at the shorter
assert list(reversed(a)) == [4, 1, 3] and a == [3, 1, 4]
assert [y - x for x, y in zip(a, a[1:])] == [-2, 3]   # neighbour pairs

i = 0
i += 1
++i                                 # parses as +(+i): no effect
assert i == 1
try:
    compile("i++", "<demo>", "exec")
except SyntaxError:
    pass
else:
    raise AssertionError("i++ should not parse")

visited = []
for i in range(3):
    visited.append(i)
    i += 100                        # reset on the next iteration
assert visited == [0, 1, 2]

stack = [5]
popped = []
while stack:                        # while (!stack.empty())
    popped.append(stack.pop())
assert popped == [5] and not stack
assert not [] and not {} and not set() and not "" and not 0 and not None
assert [0] and [[]] and "0"         # non-empty is truthy, whatever it holds

def first_negative(nums):
    for idx, x in enumerate(nums):
        if x < 0:
            break
    else:                           # runs only when the loop did not break
        return -1
    return idx

assert first_negative([2, -1, -5]) == 1 and first_negative([1, 2]) == -1

x, y = 1, 2
x, y = y, x                         # swap, no temp
assert (x, y) == (2, 1)
assert 0 <= x < 10                  # chained comparison
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `int`, `long long` | `int` (arbitrary precision) | O(1) for word-size values |
| `INT_MAX` / `INT_MIN` as sentinel | `math.inf` / `-math.inf` | O(1) |
| Overflow check before `a + b` | not needed; check `2**31 - 1` only if the problem asks | O(1) |
| `a / b` on ints (truncates) | `int(a / b)`, or integer-only `trunc_div` | O(1) |
| `a / b` with `a, b >= 0` | `a // b` | O(1) |
| `((a % m) + m) % m` | `a % m` (for `m > 0`) | O(1) |
| `std::div(a, b)` | `divmod(a, b)` (floored, not truncated) | O(1) |
| `(a + b - 1) / b` (ceil, `a >= 0`, `b > 0`) | `-(-a // b)` (any signs) | O(1) |
| `(double)a / b` | `a / b` | O(1) |
| `__builtin_popcount(x)` | `x.bit_count()` (3.10+); for `x < 0`: `(x & 0xFFFFFFFF).bit_count()` | O(1) for word-size values |
| `64 - __builtin_clzll(x)` (`x > 0`) | `x.bit_length()` | O(1) for word-size values |
| `(uint32_t)x` | `x & 0xFFFFFFFF` | O(1) |
| `(int32_t)u` | `u - (1 << 32) if u >= 1 << 31 else u` | O(1) |
| `x >> k` on `uint32_t` | `(x & 0xFFFFFFFF) >> k` | O(1) |
| `s[i] = 'x'` on `std::string` | `chars = list(s)`, edit, `"".join(chars)` | O(N) |
| `s.push_back(c)` in a loop | `parts.append(c)`, then `"".join(parts)` | O(1) amortized, O(N) join |
| `c - 'a'` | `ord(c) - ord("a")` | O(1) |
| `(char)('a' + i)` | `chr(ord("a") + i)` | O(1) |
| `s.substr(pos, len)` | `s[pos:pos + len]` | O(len) |
| `std::string_view` | pass `(s, lo, hi)` indexes | O(1) |
| `std::reverse(v.begin(), v.end())` | `v.reverse()` (list, in place) | O(N) |
| reversed copy of a string | `s[::-1]` | O(N) |
| `vector<vector<int>> g(r, vector<int>(c))` | `[[0] * c for _ in range(r)]` | O(R * C) |
| `auto copy = g;` (nested vectors) | `[row[:] for row in g]` or `copy.deepcopy(g)` | O(R * C) |
| `T* b = &a;` (not `T&`) | `b = a`; a later `b = x` rebinds `b`, never writes into `a` | O(1) |
| `&a == &b` | `a is b` | O(1) |
| `a == b` on vectors | `a == b` | O(N) |
| `p == nullptr` | `x is None` | O(1) |
| `for (int i = 0; i < n; ++i)` | `for i in range(n):` | O(1) per step |
| `for (int i = n - 1; i >= 0; --i)` | `for i in range(n - 1, -1, -1):` | O(1) per step |
| `for (auto it = v.rbegin(); ...)` | `for x in reversed(v):` | O(1) per step |
| `while (!st.empty())` | `while st:` | O(1) |
| `std::swap(x, y)` | `x, y = y, x` | O(1) |

## Complexity

| Operation | Time | Space |
|---|---|---|
| `+ - * // %` on word-size ints | O(1) | O(1) |
| `+` on D-digit ints | O(D) | O(D) |
| `*` on D-digit ints | about O(D^1.58) (Karatsuba) | O(D) |
| `x.bit_length()`, `x.bit_count()` (word-size) | O(1) | O(1) |
| `s[i]`, `len(s)`, `ord(c)`, `chr(i)` | O(1) | O(1) |
| `s + t` | O(len(s) + len(t)) | new string |
| `s += ch`, repeated N times | O(N) to O(N^2), not guaranteed | new string per step in the worst case |
| `"".join(parts)` | O(total length) | O(total length) |
| `s[i:j]`, `a[i:j]` | O(j - i) | O(j - i) copy |
| `s[::-1]` | O(N) | O(N) |
| `a.reverse()` | O(N) | O(1) |
| `reversed(a)` | O(1) to create, O(N) to walk | O(1) |
| `range`, `enumerate`, `zip` | O(1) to create (lazy) | O(1) |
| `a[:]`, `a.copy()`, `list(a)` | O(N) | O(N), shallow |
| `[row[:] for row in grid]` | O(R * C) | O(R * C) |
| `copy.deepcopy(obj)` | O(total objects), large constant | O(total objects) |
| `[[0] * c for _ in range(r)]` | O(R * C) | O(R * C) |
| `a == b` on lists | O(N) worst case | O(1) |
| `a is b`, `x is None` | O(1) | O(1) |
| `s.split()`, `s.find(t)`, `s.upper()` | O(N) typical | O(N) for new strings |

## Gotchas for C++ developers

- **Floor semantics:** `-7 // 2 == -4`, `-7 % 2 == 1`. For "truncate toward zero" use `int(a / b)`.
- `int(a / b)` goes through a float: exact while `abs(a) < 2**53`. Huge ints need integer-only truncation.
- `/` always returns a float: `a[n / 2]` raises `TypeError`. Use `a[n // 2]`.
- **No overflow** means no free 32-bit clamping: check `-2**31 <= r <= 2**31 - 1` yourself.
- `math.inf` is a `float`: convert to `-1` (or the required int) before returning.
- `sys.maxsize` is a container-size limit, not a numeric max. Prefer `math.inf` for sentinels.
- `~x` is negative for `x >= 0`. For a 32-bit flip use `~x & 0xFFFFFFFF`.
- Carry loops (add with XOR) can loop forever on negatives (`-3 + 5`) without `& 0xFFFFFFFF`.
- `x.bit_count()` counts `abs(x)`: `(-1).bit_count() == 1`, C++ `__builtin_popcount(-1) == 32`. Mask first.
- `& | ^` bind tighter than `==` in Python: `x & 1 == 0` means `(x & 1) == 0`. C++ reads `x & (1 == 0)`.
- `str` is immutable: `s[i] = c` raises `TypeError`.
- `s += c` in a loop: CPython sometimes resizes in place, but not reliably. Join a list.
- `c - 'a'` raises `TypeError`: use `ord(c) - ord("a")`.
- Slices copy: `s[i:j]` inside a loop is a hidden O(K) per step.
- Slice ends are indexes: `s[2:5]` is 3 chars, not `substr(2, 5)`.
- `[[0] * c] * r` aliases rows. `[0] * n` is fine because ints are immutable.
- `lst += [x]` mutates in place (every alias sees it); `lst = lst + [x]` builds a new list.
- `def f(acc=[])` shares one list across calls. Default to `None`.
- `a[:]` and `.copy()` are shallow: nested lists stay shared.
- `is` on numbers or strings: CPython caches small ints (-5 to 256), so it can pass by accident. Use `==`.
- `++i` is legal and does nothing; `i++` is a `SyntaxError`.
- Changing `i` inside `for i in range(n)` does not skip iterations. Use `while`.
- A nested function that does `total += 1` raises `UnboundLocalError` without `nonlocal total`.
- `round(2.5) == 2` (half to even). `int(-3.7) == -3`, `math.floor(-3.7) == -4`.
- `0.1 + 0.2 != 0.3`, same IEEE-754 doubles as C++. Compare with `math.isclose`.
- Default recursion limit is 1000 frames: deep DFS raises `RecursionError`. Use an explicit stack.
- `str(n)` / `int(s)` above 4300 digits raises `ValueError` (3.11+ default limit).

## In the Google bank

- **G-200 Number of Islands**: `visited = [[False] * cols for _ in range(rows)]`; recursion depth on a huge island.
- **G-62 Unique Paths**: DP grid built with a comprehension, never `[[0] * n] * m`.
- **G-1143 Longest Common Subsequence**: `(m + 1) x (n + 1)` table, row by comprehension.
- **G-72 Edit Distance**: same 2D table setup; base row and column filled with `range`.
- **G-98 Validate Binary Search Tree**: bounds `-math.inf` / `math.inf`; no `LLONG_MIN` tricks.
- **G-322 Coin Change**: `dp = [math.inf] * (amount + 1)`; return `-1` when `dp[amount] == math.inf`.
- **G-743 Network Delay Time**: `dist` starts at `math.inf`; `dist[u] + w` never overflows.
- **G-124 Binary Tree Maximum Path Sum**: `best = -math.inf` and `nonlocal best` in the DFS.
- **G-152 Maximum Product Subarray**: products never overflow in Python.
- **G-410 Split Array Largest Sum**: `mid = (lo + hi) // 2` is overflow-free.
- **G-5 Longest Palindromic Substring**: expand with indexes, slice once at the end.
- **G-139 Word Break**: each `s[j:i] in words` check slices O(i - j).
- **G-91 Decode Ways**: `int(s[i - 2:i])` for the two-digit case; compare chars like `s[i - 1] != "0"`.
- **G-208 Implement Trie (Prefix Tree)**: child index `ord(c) - ord("a")`, or a dict of children.
- **G-297 Serialize and Deserialize Binary Tree**: `",".join(parts)` instead of `+=`.
- **G-739 Daily Temperatures**: `while stack and temps[stack[-1]] < t:` relies on truthiness.
