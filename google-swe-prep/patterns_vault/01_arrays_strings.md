# Arrays and Strings: Two Pointers and Sliding Window

**Use when:**
- Input is **sorted** and you need a pair or triplet with a target sum, or the task says **in place** / **O(1) extra space**.
- The words **palindrome**, reverse, partition, remove duplicates or move zeroes appear.
- A **contiguous subarray / substring** that is **longest**, **shortest**, has **at most K** of X, or has no repeats.
- The task says **exactly K** or "number of subarrays with K ...": count `atMost(K) - atMost(K - 1)`.

## Python template(s)

### Converging pointers (sorted input)

- `l` starts left, `r` starts right. Each move throws away a candidate that **cannot** be in a better answer.
- One pass: O(N). Sorting first adds O(N log N).

```python
def two_sum_sorted(a, target):   # Two Sum II: a is sorted
    l, r = 0, len(a) - 1
    while l < r:
        s = a[l] + a[r]
        if s == target:
            return l, r
        if s < target:
            l += 1               # need a bigger sum
        else:
            r -= 1               # need a smaller sum
    return None

assert two_sum_sorted([2, 7, 11, 15], 9) == (0, 1)
assert two_sum_sorted([1, 3, 4], 100) is None


def three_sum(nums):             # 3Sum: unique triplets that sum to 0
    nums = sorted(nums)          # new list, O(N log N), stable
    n = len(nums)
    out = []
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue             # skip a duplicate anchor
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                out.append([nums[i], nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1       # skip a duplicate second value
    return out

assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
assert three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
assert three_sum([1, 2]) == []


def max_area(h):                 # Container With Most Water
    l, r = 0, len(h) - 1
    best = 0
    while l < r:
        best = max(best, (r - l) * min(h[l], h[r]))
        if h[l] < h[r]:
            l += 1               # the shorter wall can never do better
        else:
            r -= 1
    return best

assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49


def trap(h):                     # G-42 Trapping Rain Water, O(1) extra space
    l, r = 0, len(h) - 1
    left_max = right_max = water = 0
    while l < r:
        if h[l] < h[r]:          # right side has a wall >= h[l]
            left_max = max(left_max, h[l])
            water += left_max - h[l]
            l += 1
        else:
            right_max = max(right_max, h[r])
            water += right_max - h[r]
            r -= 1
    return water

assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap([4, 2, 0, 3, 2, 5]) == 9
assert trap([]) == 0


def is_palindrome(s):            # Valid Palindrome: letters and digits only
    l, r = 0, len(s) - 1
    while l < r:
        if not s[l].isalnum():
            l += 1
        elif not s[r].isalnum():
            r -= 1
        elif s[l].lower() != s[r].lower():
            return False
        else:
            l += 1
            r -= 1
    return True

assert is_palindrome("A man, a plan, a canal: Panama")
assert not is_palindrome("race a car")
assert is_palindrome(" ")
```

### Same-direction pointers (read / write)

- `r` reads every element. `w` is the **next slot to write**. Everything before `w` is the answer.
- In place, O(N) time, O(1) extra space. Also: one pointer per input when merging sorted lists.

```python
def remove_duplicates(a):        # sorted a; returns k, answer is a[:k]
    if not a:
        return 0
    w = 1
    for r in range(1, len(a)):
        if a[r] != a[w - 1]:
            a[w] = a[r]
            w += 1
    return w

a = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
k = remove_duplicates(a)
assert k == 5 and a[:k] == [0, 1, 2, 3, 4]
assert remove_duplicates([]) == 0


def move_zeroes(a):              # keeps the order of non-zero values
    w = 0
    for r in range(len(a)):
        if a[r] != 0:
            a[w], a[r] = a[r], a[w]
            w += 1

b = [0, 1, 0, 3, 12]
move_zeroes(b)
assert b == [1, 3, 12, 0, 0]


def merge(x, y):                 # merge two sorted lists
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        if x[i] <= y[j]:         # <= keeps equal items in input order (stable)
            out.append(x[i])
            i += 1
        else:
            out.append(y[j])
            j += 1
    out.extend(x[i:])            # at most one of these is non-empty
    out.extend(y[j:])
    return out

assert merge([1, 3, 5], [2, 3, 4, 9]) == [1, 2, 3, 3, 4, 5, 9]
assert merge([], [1]) == [1]


def is_subsequence(s, t):        # is s a subsequence of t?
    i = 0
    for ch in t:
        if i < len(s) and s[i] == ch:
            i += 1
    return i == len(s)

assert is_subsequence("ace", "abcde") and not is_subsequence("aec", "abcde")
```

### Fast / slow pointers

- `slow` moves 1 step, `fast` moves 2. If they meet, there is a **cycle** (Floyd).
- When `fast` hits the end, `slow` is at the **middle**.

```python
class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def build(vals):
    head = None
    for v in reversed(vals):
        head = ListNode(v, head)
    return head


def middle(head):                # second middle for an even length
    slow = fast = head
    while fast and fast.next:    # fast != nullptr && fast->next != nullptr
        slow = slow.next
        fast = fast.next.next
    return slow

assert middle(build([1, 2, 3, 4, 5])).val == 3
assert middle(build([1, 2, 3, 4, 5, 6])).val == 4


def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:         # same node object: compare with `is`
            return True
    return False

head = build([3, 2, 0, -4])
assert not has_cycle(head)
head.next.next.next.next = head.next     # tail -> second node
assert has_cycle(head)


def find_duplicate(nums):        # Find the Duplicate Number: i -> nums[i] is a linked list
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]               # phase 2: both move 1 step, meet at the cycle entry
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow

assert find_duplicate([1, 3, 4, 2, 2]) == 2
assert find_duplicate([3, 1, 3, 4, 2]) == 3
```

### Sliding window: variable size

- Grow `r` each step. **Shrink** `l` while invalid (longest) or still valid (shortest). Each index enters and leaves once: O(N).
- Needs **monotonic validity**: growing only ever breaks it, or only ever fixes it. Counts and sums of non-negatives qualify.

```python
import math
from collections import Counter


def longest_unique(s):           # Longest Substring Without Repeating Characters
    last = {}                    # char -> last index seen
    l = best = 0
    for r, ch in enumerate(s):
        if last.get(ch, -1) >= l:
            l = last[ch] + 1     # jump past the previous copy
        last[ch] = r
        best = max(best, r - l + 1)
    return best

assert longest_unique("abcabcbb") == 3
assert longest_unique("bbbbb") == 1
assert longest_unique("pwwkew") == 3
assert longest_unique("abba") == 2     # stale index of "a" is left of l
assert longest_unique("") == 0


def min_len_sum_at_least(target, nums):   # Minimum Size Subarray Sum, nums > 0
    l = total = 0
    best = math.inf
    for r, x in enumerate(nums):
        total += x
        while total >= target:   # valid: record, then try to shrink
            best = min(best, r - l + 1)
            total -= nums[l]
            l += 1
    return 0 if best == math.inf else best   # never return the float inf

assert min_len_sum_at_least(7, [2, 3, 1, 2, 4, 3]) == 2
assert min_len_sum_at_least(11, [1, 1, 1]) == 0


def min_window(s, t):            # Minimum Window Substring
    need = Counter(t)
    missing = len(t)             # chars of t not yet covered
    l = 0
    best_len, best_l = math.inf, 0
    for r, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1            # negative = surplus inside the window
        while missing == 0:
            if r - l + 1 < best_len:
                best_len, best_l = r - l + 1, l
            need[s[l]] += 1
            if need[s[l]] > 0:
                missing += 1     # dropped a char we needed
            l += 1
    return "" if best_len == math.inf else s[best_l:best_l + best_len]   # slice once

assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "a") == "a"
assert min_window("a", "aa") == ""
```

### Exactly K = atMost(K) - atMost(K - 1)

- "**Exactly K**" is not monotonic: growing **and** shrinking can both break it. "**At most K**" is.
- In an at-most window, every subarray ending at `r` and starting in `[l, r]` is valid: add `r - l + 1`.

```python
from collections import defaultdict


def at_most_k_distinct(nums, k):
    if k < 0:
        return 0
    count = defaultdict(int)
    l = total = 0
    for r, x in enumerate(nums):
        count[x] += 1
        while len(count) > k:
            count[nums[l]] -= 1
            if count[nums[l]] == 0:
                del count[nums[l]]   # or len(count) never drops
            l += 1
        total += r - l + 1           # windows that end at r
    return total


def exactly_k_distinct(nums, k):     # Subarrays with K Different Integers
    return at_most_k_distinct(nums, k) - at_most_k_distinct(nums, k - 1)

assert exactly_k_distinct([1, 2, 1, 2, 3], 2) == 7
assert exactly_k_distinct([1, 2, 1, 3, 4], 3) == 3


def num_subarrays_with_sum(bits, goal):   # Binary Subarrays With Sum (0/1 values)
    def at_most(g):
        if g < 0:
            return 0
        l = s = total = 0
        for r, b in enumerate(bits):
            s += b
            while s > g:
                s -= bits[l]
                l += 1
            total += r - l + 1
        return total
    return at_most(goal) - at_most(goal - 1)

assert num_subarrays_with_sum([1, 0, 1, 0, 1], 2) == 4
assert num_subarrays_with_sum([0, 0, 0, 0, 0], 0) == 15


def brute_exactly_k(nums, k):    # check the trick against brute force
    return sum(len(set(nums[i:j + 1])) == k
               for i in range(len(nums)) for j in range(i, len(nums)))

for arr in ([1, 2, 1, 2, 3], [1, 1, 1], [4, 3, 4, 3, 2, 1, 1]):
    for k in range(1, 4):
        assert exactly_k_distinct(arr, k) == brute_exactly_k(arr, k)
```

### Sliding window: fixed size

- Add `a[r]`. Once `r >= k`, remove `a[r - k]`. The window is `a[r - k + 1 .. r]`.
- Anagram search: keep two **26-count lists**; `have == need` is O(26).

```python
def max_sum_k(a, k):
    s = sum(a[:k])               # first window
    best = s
    for r in range(k, len(a)):
        s += a[r] - a[r - k]     # slide by one
        best = max(best, s)
    return best

assert max_sum_k([1, 12, -5, -6, 50, 3], 4) == 51    # [12, -5, -6, 50]
assert max_sum_k([5], 1) == 5


def find_anagrams(s, p):         # Find All Anagrams in a String (lowercase)
    k = len(p)
    if k > len(s):
        return []
    need = [0] * 26
    have = [0] * 26
    for ch in p:
        need[ord(ch) - ord("a")] += 1
    out = []
    for r, ch in enumerate(s):
        have[ord(ch) - ord("a")] += 1
        if r >= k:
            have[ord(s[r - k]) - ord("a")] -= 1
        if r >= k - 1 and have == need:   # list == list compares values
            out.append(r - k + 1)
    return out

assert find_anagrams("cbaebabacd", "abc") == [0, 6]
assert find_anagrams("abab", "ab") == [0, 1, 2]
assert find_anagrams("a", "ab") == []
```

C++ side by side:

```cpp
#include <algorithm>
#include <cassert>
#include <string>
#include <utility>
#include <vector>

int longestUnique(const std::string& s) {
    std::vector<int> last(256, -1);              // char -> last index
    int l = 0, best = 0;
    for (int r = 0; r < (int)s.size(); ++r) {
        unsigned char c = s[r];
        if (last[c] >= l) l = last[c] + 1;
        last[c] = r;
        best = std::max(best, r - l + 1);
    }
    return best;
}

int main() {
    assert(longestUnique("abcabcbb") == 3);
    assert(longestUnique("abba") == 2);
    assert(longestUnique("") == 0);

    // Converging pointers on a sorted vector.
    std::vector<int> a{2, 7, 11, 15};
    int l = 0, r = (int)a.size() - 1;
    while (l < r && a[l] + a[r] != 9) {
        if (a[l] + a[r] < 9) {
            ++l;
        } else {
            --r;
        }
    }
    assert(l == 0 && r == 1);

    // std::sort is not stable; std::stable_sort matches Python's sort.
    std::vector<std::pair<int, char>> v{{1, 'b'}, {0, 'x'}, {1, 'a'}};
    std::stable_sort(v.begin(), v.end(),
                     [](const auto& x, const auto& y) { return x.first < y.first; });
    assert(v[1].second == 'b' && v[2].second == 'a');
    return 0;
}
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `int l = 0, r = n - 1;` | `l, r = 0, len(a) - 1` | O(1) |
| `for (int r = 0; r < n; ++r)` reading `a[r]` | `for r, x in enumerate(a):` | O(1) per step |
| `std::sort(v.begin(), v.end())` (not stable) | `v.sort()` in place, or `sorted(v)` for a new list (both stable) | O(N log N) |
| `std::stable_sort` | `v.sort()` / `sorted(v)` | O(N log N) |
| `std::swap(a[i], a[j])` | `a[i], a[j] = a[j], a[i]` | O(1) |
| `int cnt[26] = {};` | `cnt = [0] * 26` | O(26) |
| `std::equal` on two count arrays | `have == need` | O(26) |
| `unordered_map<char, int>` window counts | `Counter()` or `defaultdict(int)` | O(1) average per update |
| `m.erase(key)` when a count hits 0 | `del count[key]` | O(1) average |
| `m.size()` as "distinct in window" | `len(count)`, only if zero counts are deleted | O(1) |
| `s.substr(l, len)` for the answer | `s[l:l + length]`, once, at the end | O(length) |
| `INT_MAX` as "no answer yet" | `math.inf`, converted before returning | O(1) |
| `std::reverse(v.begin(), v.end())` | `v.reverse()` in place, or `v[::-1]` copy | O(N) |
| `v.erase(std::unique(...), v.end())` on sorted `v` | read/write pointers, or `list(dict.fromkeys(v))` | O(N) |
| `ListNode *slow = head, *fast = head;` | `slow = fast = head` | O(1) |
| `slow == fast` (pointer compare) | `slow is fast` | O(1) |
| `fast && fast->next` | `fast and fast.next` | O(1) |

## Complexity

| Operation | Time | Space |
|---|---|---|
| Converging pointers on sorted input | O(N) | O(1) |
| Sort first (`list.sort()` / `sorted()`) | O(N log N) | O(N) worst case (Timsort merge buffer; `sorted` also copies) |
| 3Sum: sort + pointers per anchor | O(N^2) | O(1) besides sort and output |
| Read / write pointers in place | O(N) | O(1) |
| Merge two sorted lists | O(N + M) | O(N + M) output |
| Fast / slow (cycle, middle, duplicate) | O(N) | O(1) |
| Variable window with an O(1) validity check | O(N): each index enters and leaves once | O(distinct in window) |
| Exactly K via two at-most passes | O(N) | O(K) |
| Fixed window with a running sum | O(N) | O(1) |
| Anagram window with 26-count compare | O(26 * N) = O(N) | O(26) |
| `s[l:r + 1]` inside the loop | O(window) per step: O(N * W) total | O(W) per copy |
| `sum(a[:k])` | O(K) | O(K) copy |

## Gotchas for C++ developers

- **Slices copy.** `s[l:r + 1]` inside the loop turns O(N) into O(N * W). Track `best_l`, `best_len`; slice once.
- `a.sort()` returns `None`. `a = a.sort()` silently sets `a` to `None`; use `sorted(a)` for a new list.
- Python's sort is **stable**; `std::sort` is not. Ties keep their input order.
- `len(count)` still counts keys whose value fell to 0. `del` them or "distinct in window" is wrong.
- Reading `d[k]` on a `defaultdict` **inserts** `k`. `Counter` reads return 0 without inserting.
- Changing `l` inside `for l in range(n)` has no effect next iteration. Keep `l` as a plain variable.
- **Negative indexes do not raise.** `a[l - 1]` with `l == 0` silently reads the last element.
- Compare nodes with `is`. `==` calls a user `__eq__` if one exists.
- `None.next` raises `AttributeError`: check `fast and fast.next` in that order.
- **Negative numbers** break "shrink while sum >= K". Use prefix sums + hash map, or a monotonic deque (G-862).
- `math.inf` is a `float`: return `0` or `-1` when no window was found.
- `str.isalnum()` is Unicode-aware (`"é".isalnum()` is `True`); C `isalnum` in the "C" locale is ASCII only.

## In the Google bank

- **G-42 Trapping Rain Water**: converging pointers with `left_max` / `right_max`, O(1) space (stack version in 02).
- **G-5 Longest Palindromic Substring**: two pointers expanding outward from each of the 2N - 1 centres.
- **G-239 Sliding Window Maximum**: fixed window; the max needs a monotonic deque (03).
- **G-862 Shortest Subarray with Sum at Least K**: negatives break the variable window; prefix sums + monotonic deque (03).
- **G-57 Insert Interval**: one forward pointer, three phases: before, overlap, after.
- **G-56 Merge Intervals**: sort, then a read pointer and the last kept interval as the write slot.
- **G-23 Merge k Sorted Lists**: the two-pointer merge generalised to a heap of K heads.
- **G-759 Employee Free Time**: merge sorted interval lists, then scan for gaps.
