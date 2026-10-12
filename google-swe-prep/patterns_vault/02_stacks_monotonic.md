# Stacks and Monotonic Stacks

**Use when:**
- **Matching or nesting**: brackets, "valid parentheses", decode `3[a2[c]]`, undo, simplify a path.
- **Next greater / next smaller** element, "how many days until warmer", "first taller to the right".
- **Span, visible, blocked by**: stock span, people who can see over others.
- **Largest rectangle** under bars, water trapped between bars, "sum of subarray minimums".

## Python template(s)

### Stack basics with `list`

- `append`, `pop()` and `st[-1]` work at the **end** of a list: O(1) (amortized for `append`).
- `pop()` **returns** the value. On an empty list it raises `IndexError`: guard with `if st:`.

```python
st = []
st.append(1)                     # push
st.append(2)
assert st[-1] == 2               # top
assert st.pop() == 2             # pop returns the value (C++ pop() is void)
assert len(st) == 1 and st       # non-empty list is truthy
st.pop()
assert not st                    # st.empty()
try:
    st.pop()
except IndexError:
    pass
else:
    raise AssertionError("pop on an empty list must raise")


def is_valid(s):                 # Valid Parentheses
    pairs = {")": "(", "]": "[", "}": "{"}
    st = []
    for ch in s:
        if ch in pairs:
            if not st or st.pop() != pairs[ch]:
                return False
        else:
            st.append(ch)
    return not st                # leftover openers are invalid

assert is_valid("()[]{}") and is_valid("{[()]}")
assert not is_valid("(]") and not is_valid("(") and not is_valid(")")


def decode_string(s):            # Decode String: "3[a2[c]]" -> "accaccacc"
    st = []                      # (parts before "[", repeat count)
    cur, num = [], 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == "[":
            st.append((cur, num))
            cur, num = [], 0
        elif ch == "]":
            prev, k = st.pop()
            prev.append("".join(cur) * k)
            cur = prev
        else:
            cur.append(ch)
    return "".join(cur)

assert decode_string("3[a2[c]]") == "accaccacc"
assert decode_string("2[abc]3[cd]ef") == "abcabccdcdcdef"
assert decode_string("10[a]") == "a" * 10


class MinStack:                  # Min Stack: every operation O(1)
    def __init__(self):
        self.st = []             # (value, min of the stack up to here)

    def push(self, x):
        m = min(x, self.st[-1][1]) if self.st else x
        self.st.append((x, m))

    def pop(self):
        self.st.pop()

    def top(self):
        return self.st[-1][0]

    def get_min(self):
        return self.st[-1][1]

ms = MinStack()
for x in (5, 3, 7, 3):
    ms.push(x)
assert ms.get_min() == 3 and ms.top() == 3
ms.pop()
ms.pop()
assert ms.get_min() == 3 and ms.top() == 3
ms.pop()
assert ms.get_min() == 5 and ms.top() == 5
```

### Explicit stack instead of recursion

- Python's default recursion limit is **1000 frames**. A skewed tree or a long path raises `RecursionError`.
- Replace the call stack with a `list`. Same O(N) time and O(H) space.

```python
import sys


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def kth_smallest(root, k):       # G-230: iterative in-order traversal
    st, node = [], root
    while st or node:
        while node:              # go left as far as possible
            st.append(node)
            node = node.left
        node = st.pop()
        k -= 1
        if k == 0:
            return node.val
        node = node.right
    return None

#       5
#      / \
#     3   6
#    / \
#   2   4
root = TreeNode(5, TreeNode(3, TreeNode(2), TreeNode(4)), TreeNode(6))
assert [kth_smallest(root, k) for k in range(1, 6)] == [2, 3, 4, 5, 6]

# A 10000-node left chain: 10000 -> 9999 -> ... -> 1
N = 10_000
chain = None
for v in range(1, N + 1):
    chain = TreeNode(v, left=chain)


def inorder_recursive(node, out):
    if node:
        inorder_recursive(node.left, out)
        out.append(node.val)
        inorder_recursive(node.right, out)

assert sys.getrecursionlimit() < N           # CPython default is 1000
try:
    inorder_recursive(chain, [])
    raised = False
except RecursionError:
    raised = True
assert raised                                # recursion would be N frames deep
assert kth_smallest(chain, 1) == 1           # the list stack grows to N: fine
assert kth_smallest(chain, N) == N
```

### Monotonic stack: next greater element

- Keep **indexes whose answer is still unknown**. For "next greater" their values are **non-increasing** bottom to top.
- A new value `x` pops every smaller top: `x` is their answer. Each index is pushed once and popped once: O(N).

| Want for each `i` | Pop while | Stack values, bottom to top |
|---|---|---|
| next strictly greater | `a[st[-1]] < x` | non-increasing |
| next greater or equal | `a[st[-1]] <= x` | strictly decreasing |
| next strictly smaller | `a[st[-1]] > x` | non-decreasing |
| next smaller or equal | `a[st[-1]] >= x` | strictly increasing |
| previous strictly greater | `a[st[-1]] <= x`, then read `st[-1]` before pushing `i` | strictly decreasing |
| previous strictly smaller | `a[st[-1]] >= x`, then read `st[-1]` before pushing `i` | strictly increasing |

```python
def next_greater(a):             # value of the next strictly greater, -1 if none
    ans = [-1] * len(a)
    st = []                      # indexes; values non-increasing
    for i, x in enumerate(a):
        while st and a[st[-1]] < x:
            ans[st.pop()] = x    # x is the answer for every popped index
        st.append(i)
    return ans

assert next_greater([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
assert next_greater([5, 5]) == [-1, -1]          # strictly greater only


def daily_temperatures(t):       # G-739: days to wait, 0 if never
    ans = [0] * len(t)
    st = []
    for i, x in enumerate(t):
        while st and t[st[-1]] < x:
            j = st.pop()
            ans[j] = i - j       # distance needs the index, not the value
        st.append(i)
    return ans

assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
assert daily_temperatures([30, 60, 90]) == [1, 1, 0]


def next_greater_circular(a):    # G-503: walk the array twice
    n = len(a)
    ans = [-1] * n
    st = []
    for k in range(2 * n):
        i = k % n
        while st and a[st[-1]] < a[i]:
            ans[st.pop()] = a[i]
        if k < n:
            st.append(i)         # push each index only once
    return ans

assert next_greater_circular([1, 2, 1]) == [2, -1, 2]
assert next_greater_circular([1, 2, 3, 4, 3]) == [2, 3, 4, -1, 4]


def previous_smaller(a):         # index of the previous strictly smaller, -1 if none
    ans = []
    st = []
    for i, x in enumerate(a):
        while st and a[st[-1]] >= x:
            st.pop()
        ans.append(st[-1] if st else -1)
        st.append(i)
    return ans

assert previous_smaller([3, 1, 4, 1, 5]) == [-1, -1, 1, -1, 3]


def can_see_persons_count(h):    # G-1944: distinct heights, scan right to left
    ans = [0] * len(h)
    st = []                      # heights to the right; decreasing bottom to top
    for i in range(len(h) - 1, -1, -1):
        seen = 0
        while st and st[-1] < h[i]:
            st.pop()             # i sees it, and it hides from everyone left of i
            seen += 1
        ans[i] = seen + (1 if st else 0)   # plus the first taller person
        st.append(h[i])
    return ans

assert can_see_persons_count([10, 6, 8, 5, 11, 9]) == [3, 1, 2, 1, 1, 0]
assert can_see_persons_count([5, 1, 2, 3, 10]) == [4, 1, 1, 1, 0]
```

### Monotonic stack: areas between bars

- **Largest rectangle (G-84)**: increasing stack. A lower bar at `i` ends the top's rectangle at `i - 1`.
- The left edge is the new top + 1. A **sentinel** bar of height 0 at `i == n` flushes the stack.

```python
def largest_rectangle(heights):  # G-84
    n = len(heights)
    st = []                      # indexes; heights increasing
    best = 0
    for i in range(n + 1):
        h = heights[i] if i < n else 0     # sentinel, no list copy
        while st and heights[st[-1]] >= h:
            height = heights[st.pop()]
            left = st[-1] if st else -1    # first lower bar on the left
            best = max(best, height * (i - left - 1))
        st.append(i)
    return best

assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10
assert largest_rectangle([2, 4]) == 4
assert largest_rectangle([3, 3, 3]) == 9
assert largest_rectangle([]) == 0


def trap_stack(h):               # G-42 with a decreasing stack
    st = []
    water = 0
    for i, x in enumerate(h):
        while st and h[st[-1]] < x:
            bottom = st.pop()
            if not st:
                break            # no wall on the left
            left = st[-1]
            water += (min(h[left], x) - h[bottom]) * (i - left - 1)
        st.append(i)
    return water

assert trap_stack([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_stack([4, 2, 0, 3, 2, 5]) == 9


def sum_subarray_mins(a):        # Sum of Subarray Minimums: a[i] is the min of left*right subarrays
    n = len(a)
    left = [0] * n               # i - previous strictly smaller index
    right = [0] * n              # next smaller-or-equal index - i
    st = []
    for i in range(n):
        while st and a[st[-1]] >= a[i]:
            st.pop()
        left[i] = i - (st[-1] if st else -1)
        st.append(i)
    st = []
    for i in range(n - 1, -1, -1):
        while st and a[st[-1]] > a[i]:     # one side strict, one not: no double count
            st.pop()
        right[i] = (st[-1] if st else n) - i
        st.append(i)
    return sum(x * l * r for x, l, r in zip(a, left, right))

assert sum_subarray_mins([3, 1, 2, 4]) == 17
assert sum_subarray_mins([11, 81, 94, 43, 3]) == 444
assert sum_subarray_mins([1, 1]) == 3              # [1], [1], [1, 1]
brute = lambda a: sum(min(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))
for arr in ([2, 2, 2], [5, 1, 5, 1], [4, 2, 3, 2, 4]):
    assert sum_subarray_mins(arr) == brute(arr)
```

C++ side by side:

```cpp
#include <cassert>
#include <stack>
#include <vector>

std::vector<int> dailyTemperatures(const std::vector<int>& t) {
    std::vector<int> ans(t.size(), 0);
    std::stack<int> st;                          // indexes
    for (int i = 0; i < (int)t.size(); ++i) {
        while (!st.empty() && t[st.top()] < t[i]) {
            int j = st.top();
            st.pop();                            // pop() returns void
            ans[j] = i - j;
        }
        st.push(i);
    }
    return ans;
}

int main() {
    std::vector<int> want{1, 1, 4, 2, 1, 1, 0, 0};
    assert(dailyTemperatures({73, 74, 75, 71, 69, 72, 76, 73}) == want);

    // std::stack has no iteration and no indexing; a vector used as a stack does.
    std::vector<int> v;
    v.push_back(1);
    v.push_back(2);
    assert(v.back() == 2 && v[0] == 1);
    v.pop_back();
    assert(v.size() == 1);
    return 0;
}
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `std::stack<int> st;` | `st = []` | O(1) |
| `st.push(x)` | `st.append(x)` | O(1) amortized |
| `st.top()` | `st[-1]` | O(1) |
| `int v = st.top(); st.pop();` | `v = st.pop()` | O(1) |
| `st.empty()` | `not st` | O(1) |
| `st.size()` | `len(st)` | O(1) |
| `vector<int>` as a stack (`back`, `pop_back`) | the same `list` | O(1) |
| `std::stack<std::pair<int, int>>` | list of tuples: `st.append((x, m))` | O(1) amortized |
| `std::string` as a char stack | `list` of chars, then `"".join(st)` | O(1) per op, O(N) join |
| `vector<int> ans(n, -1)` | `ans = [-1] * n` | O(N) |
| recursion with a deep call stack | explicit `list` stack | same as recursion |
| `st.push(i)` only for `i < n` in a `2n` loop | `if k < n: st.append(i)` | O(1) |

## Complexity

| Operation | Time | Space |
|---|---|---|
| `st.append(x)` | O(1) amortized | O(1) |
| `st.pop()`, `st[-1]`, `len(st)` | O(1) | O(1) |
| `st.pop(0)`, `st.insert(0, x)` | O(N) | O(1) |
| `x in st` | O(N) | O(1) |
| Monotonic stack pass (next greater, spans) | O(N): each index pushed once, popped at most once | O(N) |
| Circular next greater (loop `2n`) | O(N) | O(N) |
| Largest rectangle in histogram | O(N) | O(N) |
| Sum of subarray minimums (two passes) | O(N) | O(N) |
| Brute-force next greater | O(N^2) | O(1) |
| Iterative DFS / in-order with a stack | O(N) | O(H), H = tree height |

## Gotchas for C++ developers

- `st.pop()` **returns** the value. No `top()` then `pop()` pair; `st.top()` does not exist.
- `st[-1]` on an empty list raises `IndexError`. Write `while st and a[st[-1]] < x:`; `and` short-circuits.
- Store **indexes** when you need a distance or width (G-739, G-84). Store values only when the answer is a value.
- `<` vs `<=` in the pop condition decides how **duplicates** behave. Test `[2, 2]` and `[1, 1]`.
- `heights.append(0)` as a sentinel mutates the caller's list. Use an `i == n` check, or `heights + [0]` (an O(N) copy).
- Default recursion limit is 1000 frames. `sys.setrecursionlimit` raises it, but very deep recursion can still crash.
- `queue.LifoQueue` is a **thread-safe** stack with locks: slow and pointless in an interview. Use `list`.
- `"".join(cur) * k` builds the string once; `+=` inside the loop can degrade to O(N^2).
- Python ints never overflow: sum of subarray minimums is exact; LeetCode still asks for `% (10**9 + 7)`.

## In the Google bank

- **G-739 Daily Temperatures**: decreasing stack of indexes; answer is `i - j` on pop.
- **G-503 Next Greater Element II**: loop `2n` with `i = k % n`; push only when `k < n`.
- **G-84 Largest Rectangle in Histogram**: increasing stack + sentinel height 0 at `i == n`.
- **G-1944 Number of Visible People in a Queue**: scan right to left; count pops, plus one if a taller person remains.
- **G-42 Trapping Rain Water**: decreasing stack; each pop fills a layer between the new top and the current bar.
- **G-230 Kth Smallest Element in a BST**: iterative in-order with an explicit stack; stop at the k-th pop.
- **G-98 Validate Binary Search Tree**: the same iterative in-order; values must strictly increase.
- **G-332 Reconstruct Itinerary**: iterative Hierholzer with a stack of airports; reverse the finished path.
- **G-200 Number of Islands**: DFS with an explicit stack avoids `RecursionError` on a huge island.
