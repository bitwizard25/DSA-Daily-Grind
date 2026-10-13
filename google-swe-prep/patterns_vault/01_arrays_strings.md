# Two pointers and sliding window

> In one line: do not check every pair with two nested loops; move two indexes through the list once.

## When to use it
- The list is **sorted** (smallest to largest) and you need a pair with a given sum.
- The word **palindrome** appears: a word that reads the same from both ends, like "level".
- You must change the list **in place**: inside the same list, without making a new one.
- The words **subarray** or **substring** appear: a run of items next to each other, with no gaps.
- The question asks for the **longest** or **shortest** run that follows a rule ("at most K", "no repeats").

## The idea in plain words
A **pointer** here is just an index variable, like `l` (left) or `r` (right), that you move along the list.
**O(...)** (Big-O) tells how the work grows when the list has N items.
The slow way checks every pair with two nested loops. That is O(N²): about N × N steps.
Two pointers look at each item only a few times. That is O(N): about N steps.

**Two pointers from both ends** (the list must be sorted):
1. Put `l` on the first item and `r` on the last item.
2. Add `nums[l] + nums[r]`. Too small? Move `l` one step right to get a bigger number.
3. Too big? Move `r` one step left to get a smaller number.
4. Stop when you find the target, or when `l` and `r` meet.

**Read and write pointers** (both move left to right):
1. `r` reads every item. `w` is the next place to write an item you keep.
2. When the loop ends, the kept items are at indexes 0 to `w - 1`. No second list is needed.

**Sliding window**: the **window** is the part of the list from `l` to `r`, both ends included.
1. Move `r` one step right each time. The window grows by one item.
2. If the window now breaks the rule, move `l` right until the rule holds again. The window shrinks.
3. After each step, check if this window is the best so far. Its length is `r - l + 1`.

Each item enters the window once (when `r` reaches it) and leaves once (when `l` passes it).
So the inner `while` loop does not make it O(N²): the total work is still O(N).

**Fixed-size window**: the length is always `k`. Add the new item on the right and remove the old item on the left.

## Python template

### Two pointers from both ends

```python
def is_palindrome(word):
    l, r = 0, len(word) - 1
    while l < r:
        if word[l] != word[r]:
            return False
        l += 1
        r -= 1
    return True


def pair_with_sum(nums, target):
    # nums must be sorted, smallest first
    l, r = 0, len(nums) - 1
    while l < r:
        total = nums[l] + nums[r]
        if total == target:
            return [l, r]
        if total < target:
            l += 1
        else:
            r -= 1
    return []


print(is_palindrome("level"), is_palindrome("hello"))  # True False
print(pair_with_sum([1, 3, 4, 6, 9], 10))  # [0, 4]
print(pair_with_sum([1, 2, 3], 100))  # []
```

### Read and write pointers

```python
def move_zeroes(nums):
    # keep the order of the non-zero numbers; zeros go to the end
    w = 0
    for r in range(len(nums)):
        if nums[r] != 0:
            # swap: the kept number moves to place w
            nums[w], nums[r] = nums[r], nums[w]
            w += 1


nums = [0, 1, 0, 3, 12]
move_zeroes(nums)
print(nums)  # [1, 3, 12, 0, 0]
```

### Sliding window: variable size

Longest window: grow with `r`, shrink with `l` while the rule is **broken**.

```python
def longest_no_repeat(s):
    # a set of the letters inside the window right now
    seen = set()
    l = 0
    best = 0
    for r in range(len(s)):
        # s[r] is already inside: shrink from the left until it is gone
        while s[r] in seen:
            seen.remove(s[l])
            l += 1
        seen.add(s[r])
        best = max(best, r - l + 1)
    return best


print(longest_no_repeat("abcabcbb"))  # 3
print(longest_no_repeat("pwwkew"))  # 3
print(longest_no_repeat(""))  # 0
```

Shortest window: grow with `r`, shrink with `l` while the rule still **holds**.

```python
def shortest_sum_at_least(nums, target):
    # all numbers are positive; 0 means "no such window"
    l = 0
    total = 0
    # infinity: bigger than any length, so the first real window wins
    best = float("inf")
    for r in range(len(nums)):
        total += nums[r]
        while total >= target:
            best = min(best, r - l + 1)
            total -= nums[l]
            l += 1
    if best == float("inf"):
        return 0
    return best


print(shortest_sum_at_least([2, 3, 1, 2, 4, 3], 7))  # 2
print(shortest_sum_at_least([1, 1, 1], 11))  # 0
```

### Sliding window: fixed size

```python
def best_sum_of_k(nums, k):
    # the first window: the sum of the first k items
    window = sum(nums[:k])
    best = window
    for r in range(k, len(nums)):
        # nums[r] comes in on the right, nums[r - k] leaves on the left
        window += nums[r] - nums[r - k]
        best = max(best, window)
    return best


print(best_sum_of_k([1, 4, 2, 10, 2, 3, 1, 0, 20], 4))  # 24
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `int l = 0, r = n - 1;` | `l, r = 0, len(nums) - 1` | two variables in one line |
| `while (l < r) { ... }` | `while l < r:` | no braces: the indentation is the body |
| `for (int r = 0; r < n; ++r)` | `for r in range(len(nums)):` | `range` stops before `len(nums)` |
| `swap(a[i], a[j]);` | `a[i], a[j] = a[j], a[i]` | no temporary variable needed |
| `unordered_set<char> seen;` | `seen = set()` | `seen.add(c)`, `seen.remove(c)`, `c in seen` |
| `int best = INT_MAX;` | `best = float("inf")` | Python has no `INT_MAX` |
| `s.substr(l, n)` | `s[l:l + n]` | a slice makes a copy |
| `sort(v.begin(), v.end());` | `nums.sort()` | sorts the same list; returns None |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| Check every pair with two nested loops | O(N²) | 10,000 items: about 50 million pairs to check |
| Two pointers from both ends | O(N) | each item is looked at about once |
| Read and write pointers | O(N) | one pass; no extra list |
| Sliding window (grow and shrink) | O(N) | each item enters once and leaves once |
| Sort first, then two pointers | O(N log N) | the sort is the slow part |
| `x in seen` for a set | O(1) on average | the same speed for any size |
| Slice `s[l:r + 1]` | O(length) | copies the items: keep it out of the loop |

## Common mistakes
- Mistake: two pointers from both ends on an **unsorted** list. Fix: sort first with `nums.sort()`, or use a set or dict (note 05).
- Mistake: `while l <= r` when you need two different items. Fix: `while l < r`, so `l` and `r` never point at the same item.
- Mistake: window length `r - l`. Fix: both ends are inside the window, so the length is `r - l + 1`.
- Mistake: `if` to shrink the window. One new item may need many steps of `l`. Fix: use `while`.
- Mistake: slicing `s[l:r + 1]` in every step: that copies the text each time. Fix: save `l` and the length, slice once at the end.
- Mistake: the "shrink while the sum is big enough" window with **negative** numbers. A new item can make the sum smaller. Fix: G-862 in note 03 gives the idea.
- Mistake: `nums = nums.sort()`. `sort()` returns None, so the list is lost. Fix: write `nums.sort()` alone.

## Practice in the app
- **G-42 Trapping Rain Water**: two pointers from both ends; move the side with the lower wall and keep the tallest wall seen on each side.
- **G-5 Longest Palindromic Substring**: start two pointers at the middle and move them outward while both letters match.
- **G-239 Sliding Window Maximum**: a fixed-size window; a deque finds each window's maximum fast (note 03).
- **G-862 Shortest Subarray with Sum at Least K**: looks like a sliding window, but negative numbers break it; note 03 shows the fix.
- **G-56 Merge Intervals**: sort by start, then one pointer walks the list and joins each interval to the last one you kept.
