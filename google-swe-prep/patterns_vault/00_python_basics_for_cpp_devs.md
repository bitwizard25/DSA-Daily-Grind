# Python basics for C++ developers

> In one line: Python looks like C++ without braces, but `//` rounds down, ints never overflow, strings never change, and `b = a` shares a list.

## When to use it
- You are writing your first Python lines and want each C++ habit translated.
- The problem divides **negative numbers**, or says "truncate toward zero".
- You need a start value like `INT_MAX` or `INT_MIN` for a minimum or maximum search.
- You build a **2D grid** (a list of rows), or you need a copy of one.
- You build a string one character at a time.

## The idea in plain words
1. **Shape.** Indentation (4 spaces at the start of a line) shows the block, like `{ }` in C++. No `;` is needed, and `#` starts a comment.
2. **Values.** You never write types. Python sees `int`, `float`, `str`, `bool` or `None` from the value itself.
3. **Numbers.** An `int` grows as big as needed, so it never overflows (gets too big for its type). Use `float('inf')` instead of `INT_MAX`.
4. **Division.** `/` always gives a float. `//` rounds down, toward minus infinity, so `-7 // 2` is -4. C++ gives -3; get that with `int(a / b)`.
5. **Strings** are immutable (cannot be changed). Build the pieces in a list, then glue them once with `''.join(parts)`.
6. **Names are references.** A name points at an object (a value in memory). So `b = a` does not copy, and `[[0] * 3] * 2` repeats one row.
7. **Copies.** `grid.copy()` is shallow: a new outer list, but the same rows. `copy.deepcopy(grid)` copies the rows too.

## Python template

### Shape, values and choices
```python
# A comment. No semicolons, no braces.
count = 3
name = 'Asha'
found = None
if count > 5:
    print('many')
elif 1 < count < 5:
    print('a few')  # a few
else:
    print('none')
print(name, count, found)  # Asha 3 None
print(count > 0 and not count > 5)  # True
```

### Loops and functions
```python
def min_max(nums, start=0):
    low, high = nums[start], nums[start]
    for i in range(start + 1, len(nums)):
        low = min(low, nums[i])
        high = max(high, nums[i])
    return low, high

print(min_max([4, 1, 9]))  # (1, 9)
print(min_max([4, 1, 9], 2))  # (9, 9)
low, high = min_max([4, 1, 9])
print(low, high)  # 1 9

# How many times can we halve 20 before we reach 1?
n = 20
steps = 0
while n > 1:
    n //= 2
    steps += 1
print(steps)  # 4
```

### Numbers: no overflow, infinity
```python
print(2 ** 100)  # 1267650600228229401496703205376
INT_MAX = 2 ** 31 - 1
print(INT_MAX + 1)  # 2147483648

def smallest(nums):
    best = float('inf')
    for x in nums:
        best = min(best, x)
    if best == float('inf'):
        return -1
    return best

print(smallest([7, 3, 9]), smallest([]))  # 3 -1
```

### Division: `/`, `//`, `%`
```python
print(7 / 2, 6 / 2)  # 3.5 3.0
print(7 // 2, 7 % 2)  # 3 1
print(-7 // 2, -7 % 2)  # -4 1

# Cut toward zero, like C++ integer division.
def divide_like_cpp(a, b):
    return int(a / b)

print(divide_like_cpp(-7, 2), divide_like_cpp(7, -2))  # -3 -3

# With a positive m, a % m is never negative: handy for wrap-around.
nums = [10, 20, 30]
i = 0
print(nums[(i - 1) % len(nums)])  # 30
# The middle index for binary search: no overflow to worry about.
lo, hi = 0, 9
print((lo + hi) // 2)  # 4
```

### Strings: index, slice, join
```python
s = 'google'
print(s[0], s[-1], len(s))  # g e 6
print(s[1:4], s[:3], s[3:])  # oog goo gle
print(s[::-1])  # elgoog
# s[0] = 'G' fails: a string cannot change. Build a new one.
print('G' + s[1:])  # Google

parts = []
for ch in s:
    if ch != 'o':
        parts.append(ch)
print(''.join(parts))  # ggle
print('-'.join(['2026', '10', '12']))  # 2026-10-12

# No char type: ord gives a letter's number, chr goes back.
print(ord('c') - ord('a'), chr(ord('a') + 2))  # 2 c
```

### Lists, references and grids
```python
a = [1, 2, 3]
b = a
b[0] = 9
print(a)  # [9, 2, 3]

rows, cols = 2, 3
bad = [[0] * cols] * rows
bad[0][0] = 1
print(bad)  # [[1, 0, 0], [1, 0, 0]]

grid = [[0] * cols for _ in range(rows)]
grid[0][0] = 1
print(grid)  # [[1, 0, 0], [0, 0, 0]]
print([0] * 3, [i * i for i in range(4)])  # [0, 0, 0] [0, 1, 4, 9]
```

### Copies and handy helpers
```python
import copy

grid = [[1, 2], [3, 4]]
shallow = grid.copy()
deep = copy.deepcopy(grid)
grid[0][0] = 7
print(shallow[0][0], deep[0][0])  # 7 1

nums = [30, 10, 20]
for i, x in enumerate(nums):
    if x == 10:
        print('found at', i)  # found at 1
names = ['Ann', 'Bo', 'Cy']
print(list(zip(names, nums)))  # [('Ann', 30), ('Bo', 10), ('Cy', 20)]
print(sorted(nums), nums)  # [10, 20, 30] [30, 10, 20]
nums.sort()
print(nums)  # [10, 20, 30]
```

### Extra: bits (rare in interviews)
```python
print(6 & 3, 6 | 3, 6 ^ 3)  # 2 7 5
print(1 << 4, 20 >> 2, -7 >> 1)  # 16 5 -4
print(~5)  # -6
MASK = 0xFFFFFFFF
print(-1 & MASK)  # 4294967295
print(~5 & MASK)  # 4294967290
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `if (a && !b) {` | `if a and not b:` | words, a colon, then indentation |
| `for (int i = 0; i < n; i++)` | `for i in range(n):` | `range` stops before `n` |
| `int best = INT_MAX;` | `best = float('inf')` | ints never overflow |
| `q = a / b;` | `q = int(a / b)` | cuts toward zero; `a // b` rounds down (`-7 // 2` is -4) |
| `r = ((a % m) + m) % m;` | `r = a % m` | never negative when `m > 0` |
| `s.substr(i, len)` | `s[i:i + len]` | a slice takes an end index, not a length |
| `s[i] = 'x';` | `s = s[:i] + 'x' + s[i + 1:]` | strings cannot change |
| `c - 'a'` | `ord(c) - ord('a')` | there is no `char` type |
| `vector<vector<int>> g(r, vector<int>(c));` | `g = [[0] * c for _ in range(r)]` | never `[[0] * c] * r` |
| `auto g2 = g;` | `g2 = copy.deepcopy(g)` | `g2 = g` only adds a second name |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| `a + b`, `a // b`, `a % b` on normal ints | O(1) | constant time: the same small cost for any input size |
| `s[i]`, `nums[i]`, `len(s)` | O(1) | one step |
| `nums.append(x)` | O(1) | adding at the end is fast (on average) |
| `s[a:b]` | O(b - a) | a slice copies every character it takes |
| `s += ch`, N times in a loop | up to O(N^2) | may copy the whole string at every step |
| `''.join(parts)` | O(N) | one pass over all the pieces (N = total length) |
| `[[0] * c for _ in range(r)]` | O(R * C) | touches every cell once |
| `copy.deepcopy(grid)` | O(R * C) | copies every cell |
| `sorted(nums)`, `nums.sort()` | O(N log N) | about N × log2(N) steps (N = 1000: about 10,000), like `std::sort` |

## Common mistakes
- Mistake: `-7 // 2` when the problem says "truncate toward zero". It gives -4. Fix: `int(a / b)` gives -3.
- Mistake: `grid = [[0] * cols] * rows`. One change shows up in every row. Fix: `[[0] * cols for _ in range(rows)]`.
- Mistake: `copy2 = grid` or `grid.copy()`, then changing a cell. The original changes too. Fix: `copy.deepcopy(grid)`.
- Mistake: `s += ch` inside a long loop. Fix: append the pieces to a list and call `''.join(parts)` once.
- Mistake: returning `float('inf')` when nothing was found. Fix: check for it, and return what the problem asks for (often -1).
- Mistake: C++ words like `&&`, `!x`, `true`, `null` or `i++`. Fix: write `and`, `not x`, `True`, `None` and `i += 1`.

## Practice in the app
These basics are used in every problem. They matter most in these:
- **G-399 Evaluate Division**: answers are decimals, so use `/`; return `-1.0` when a value cannot be found.
- **G-200 Number of Islands**: build `visited` with `[[False] * cols for _ in range(rows)]`, never `* rows`.
- **G-62 Unique Paths**: the DP grid (a table of answers) needs a new list for every row.
- **G-322 Coin Change**: start with `[float('inf')] * (amount + 1)` and turn a leftover infinity into -1.
- **G-98 Validate Binary Search Tree**: start the allowed range at `float('-inf')` and `float('inf')`.
- **G-152 Maximum Product Subarray**: products can get huge, but Python ints never overflow.
- **G-297 Serialize and Deserialize Binary Tree**: collect the pieces in a list, then `','.join(parts)`.
