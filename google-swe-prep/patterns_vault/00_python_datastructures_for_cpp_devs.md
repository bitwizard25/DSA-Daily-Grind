# Python Data Structures for C++ Developers

**Use when:**
- You reach for `vector`, `queue`, `priority_queue`, `unordered_map` or `lower_bound` and need the Python form.
- The problem says "shortest", "level", "top k", "k-th largest", "frequency", "visited", "sorted order".
- A correct Python solution times out: look for `pop(0)`, `x in some_list`, `insort` in a loop, `s += c`.
- Recursion can go 1000+ levels deep (long chain, big grid): plan for **RecursionError**.

## Python templates

### list: vector, stack, copies

```python
def daily_temperatures(temps):
    ans = [0] * len(temps)
    stack = []                                   # indices; temps decreasing
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:    # stack[-1] is top()
            j = stack.pop()                      # pop() returns the item
            ans[j] = i - j
        stack.append(i)                          # push_back, O(1) amortized
    return ans


assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]

a = [1, 2, 3]
b = a                          # alias, NOT a copy (C++ `vector<int> b = a;` copies)
b.append(4)
assert a == [1, 2, 3, 4]
c = a[:]                       # shallow copy, O(N); also a.copy() or list(a)
c.append(5)
assert a == [1, 2, 3, 4] and c == [1, 2, 3, 4, 5]

grid = [[0] * 3 for _ in range(2)]   # independent rows
grid[0][0] = 1
assert grid == [[1, 0, 0], [0, 0, 0]]
shared = [[0] * 3] * 2               # ONE row object, listed twice
shared[0][0] = 1
assert shared == [[1, 0, 0], [1, 0, 0]]

res, path = [], []


def subsets(i, nums):
    if i == len(nums):
        res.append(path[:])          # copy; `path` alone stores one shared list
        return
    subsets(i + 1, nums)
    path.append(nums[i])
    subsets(i + 1, nums)
    path.pop()


subsets(0, [1, 2])
assert res == [[], [2], [1], [1, 2]]


def add(x, acc=[]):                  # the default list is built ONCE
    acc.append(x)
    return acc


assert add(1) == [1]
assert add(2) == [1, 2]              # same list again; use acc=None instead
```

### deque: queue, BFS, sliding window

```python
from collections import deque


def shortest_path(grid, start, goal):
    """Steps from start to goal through '.' cells; -1 if unreachable."""
    rows, cols = len(grid), len(grid[0])
    q = deque([(start, 0)])          # deque([...]): a list holding ONE item
    seen = {start}
    while q:
        (r, c), d = q.popleft()      # O(1); list.pop(0) is O(N)
        if (r, c) == goal:
            return d
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "." and (nr, nc) not in seen:
                seen.add((nr, nc))   # mark when pushed, not when popped
                q.append(((nr, nc), d + 1))
    return -1


assert shortest_path(["..#", ".##", "..."], (0, 0), (2, 2)) == 4
assert shortest_path([".#", "#."], (0, 0), (1, 1)) == -1


def rotting_minutes(grid):
    """Multi-source BFS, one level per minute."""
    rows, cols = len(grid), len(grid[0])
    q = deque((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2)
    fresh = sum(row.count(1) for row in grid)
    minutes = 0
    while q and fresh:
        for _ in range(len(q)):      # len(q) is read once, like saving q.size()
            r, c = q.popleft()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    q.append((nr, nc))
        minutes += 1
    return minutes if fresh == 0 else -1


assert rotting_minutes([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
assert rotting_minutes([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1


def max_sliding_window(nums, k):
    dq, out = deque(), []            # indices; their values decrease
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()                 # pop_back
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()             # pop_front: index left the window
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]

last3 = deque(maxlen=3)              # keeps the last 3; NOT a reserve() hint
for x in range(5):
    last3.append(x)
assert list(last3) == [2, 3, 4]
```

### heapq: top k, merge k lists, Dijkstra, two heaps

```python
import heapq
import math
from collections import Counter, defaultdict


def kth_largest(nums, k):
    h = []
    for x in nums:
        heapq.heappush(h, x)
        if len(h) > k:
            heapq.heappop(h)         # drop the smallest: h keeps the k largest
    return h[0]


assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5


def top_k_frequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count, key=count.get)   # O(N log k)


assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]


class ListNode:                      # no __lt__: two ListNodes cannot be compared
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_k(lists):
    # i is unique among heap entries, so a tie on val never compares ListNodes
    heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
    heapq.heapify(heap)              # O(k)
    dummy = ListNode()
    tail = dummy
    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next


def from_list(vals):
    dummy = ListNode()
    tail = dummy
    for v in vals:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


merged = merge_k([from_list([1, 4, 5]), from_list([1, 3, 4]), from_list([2, 6])])
assert to_list(merged) == [1, 1, 2, 3, 4, 4, 5, 6]


def dijkstra(n, edges, src):
    g = defaultdict(list)
    for u, v, w in edges:
        g[u].append((v, w))
    dist = [math.inf] * n            # no INT_MAX: Python ints never overflow
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                 # stale entry: heapq has no decrease-key
        for v, w in g[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist


assert dijkstra(4, [(0, 1, 1), (0, 2, 4), (1, 2, 2), (2, 3, 1)], 0) == [0, 1, 3, 4]


class MedianFinder:
    def __init__(self):
        self.lo = []                 # lower half, max-heap via negation
        self.hi = []                 # upper half, min-heap

    def add(self, x):
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]       # negate back when reading
        return (-self.lo[0] + self.hi[0]) / 2


mf = MedianFinder()
mf.add(1)
mf.add(2)
assert mf.median() == 1.5
mf.add(3)
assert mf.median() == 2
```

The same Dijkstra in C++, for comparison. Note the `std::greater` needed to get the min-heap that `heapq` gives by default.

```cpp
#include <cassert>
#include <climits>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

std::vector<long long> dijkstra(int n, const std::vector<std::tuple<int, int, int>>& edges, int src) {
    std::vector<std::vector<std::pair<int, int>>> g(n);
    for (auto [u, v, w] : edges) g[u].push_back({v, w});
    std::vector<long long> dist(n, LLONG_MAX);
    using State = std::pair<long long, int>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;   // min-heap
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;                   // stale entry
        for (auto [v, w] : g[u])
            if (d + w < dist[v]) { dist[v] = d + w; pq.push({dist[v], v}); }
    }
    return dist;
}

int main() {
    auto dist = dijkstra(4, {{0, 1, 1}, {0, 2, 4}, {1, 2, 2}, {2, 3, 1}}, 0);
    assert((dist == std::vector<long long>{0, 1, 3, 4}));
    return 0;
}
```

### dict, defaultdict, Counter: hash maps

```python
from collections import Counter, defaultdict


def group_anagrams(words):
    groups = defaultdict(list)                   # missing key -> new []
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())                 # first-seen group first


assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [
    ["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]

d = {"b": 2, "a": 1, "c": 3}
assert list(d) == ["b", "a", "c"]                # insertion order, not sorted
assert sorted(d) == ["a", "b", "c"]              # std::map order costs a sort
assert max(d, key=d.get) == "c"                  # key with the largest value
assert d.get("z") is None and d.get("z", 0) == 0
assert "z" not in d                              # get() never inserts

adj = {}
adj.setdefault(1, []).append(2)                  # plain-dict version of defaultdict
assert adj == {1: [2]}

try:
    d["z"] += 1                                  # C++ m["z"]++ would insert 0 first
    raised = False
except KeyError:
    raised = True
assert raised

cnt = Counter("abracadabra")
assert cnt["a"] == 5 and cnt["z"] == 0 and "z" not in cnt
assert cnt.most_common(2) == [("a", 5), ("b", 2)]   # equal counts: first seen wins

window = Counter("aab")
window["b"] -= 1
assert len(window) == 2                          # "b" is still a key, with count 0
del window["b"]
assert len(window) == 1                          # now len() = distinct items

nums = {1: "a", 2: "b"}
try:
    for k in nums:
        nums[k + 10] = "x"                       # grows the dict mid-loop
    changed = True
except RuntimeError:
    changed = False
assert changed is False
for k in list(nums):                             # loop over a snapshot instead
    if k > 10:
        del nums[k]
assert all(k <= 10 for k in nums)
```

### set, frozenset, tuple keys

```python
def count_islands(grid):
    rows, cols = len(grid), len(grid[0])
    seen = set()                                 # set of (r, c) tuples
    islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in seen:
                islands += 1
                seen.add((r, c))
                stack = [(r, c)]                 # explicit stack: no recursion limit
                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1"
                                and (nr, nc) not in seen):
                            seen.add((nr, nc))
                            stack.append((nr, nc))
    return islands


assert count_islands(["11000", "11000", "00100", "00011"]) == 3

a, b = {1, 2, 3}, {2, 3, 4}
assert a | b == {1, 2, 3, 4} and a & b == {2, 3} and a - b == {1} and a ^ b == {1, 4}
assert type({}) is dict and type(set()) is set   # {} is an empty DICT

memo = {}
memo[(2, 3)] = 7                                 # tuple key: like map<pair<int,int>, int>
assert memo[2, 3] == 7                           # d[2, 3] is the same as d[(2, 3)]

try:
    memo[[2, 3]] = 7                             # list key
    ok = True
except TypeError:                                # unhashable type: 'list'
    ok = False
assert ok is False

try:
    hash((1, [2]))                               # a tuple is hashable only if its items are
    ok = True
except TypeError:
    ok = False
assert ok is False

seen_states = {frozenset({1, 2})}                # a set inside a set: use frozenset
assert frozenset([2, 1]) in seen_states
key = tuple(sorted([3, 1, 2]))                   # canonical key for "same multiset"
assert key == (1, 2, 3)
```

### Linked lists and trees: classes instead of pointers

```python
from collections import deque


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def from_list(vals):
    dummy = ListNode()               # dummy head: the first node is not special
    tail = dummy
    for v in vals:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head):
    out = []
    while head:                      # None is falsy, like nullptr
        out.append(head.val)
        head = head.next
    return out


def reverse(head):
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next   # right side evaluated first
    return prev


def middle(head):                    # second middle for even length
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow


def remove_value(head, target):
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next:
        if prev.next.val == target:
            prev.next = prev.next.next   # unlinked node is freed for you: no delete
        else:
            prev = prev.next
    return dummy.next


assert to_list(reverse(from_list([1, 2, 3, 4]))) == [4, 3, 2, 1]
assert middle(from_list([1, 2, 3, 4])).val == 3
assert to_list(remove_value(from_list([7, 1, 7, 2]), 7)) == [1, 2]
n1 = ListNode(5)
n2 = ListNode(5)
assert n1 is not n2                  # `is` = same object (pointer equality)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(level):
    """LeetCode-style level-order list with None gaps."""
    if not level or level[0] is None:
        return None
    root = TreeNode(level[0])
    q, i = deque([root]), 1
    while q and i < len(level):
        node = q.popleft()
        if level[i] is not None:
            node.left = TreeNode(level[i])
            q.append(node.left)
        i += 1
        if i < len(level) and level[i] is not None:
            node.right = TreeNode(level[i])
            q.append(node.right)
        i += 1
    return root


def level_order(root):
    out, q = [], deque([root] if root else [])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        out.append(level)
    return out


def inorder(root):                   # iterative: safe on a 10^5-deep chain
    out, stack, node = [], [], root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


root = build_tree([3, 9, 20, None, None, 15, 7])
assert level_order(root) == [[3], [9, 20], [15, 7]]
assert inorder(root) == [9, 3, 15, 20, 7]
```

### bisect: lower_bound and upper_bound

```python
from bisect import bisect_left, bisect_right, insort


def lis_length(nums):
    tails = []                       # tails[k] = smallest tail of an increasing subsequence of length k+1
    for x in nums:
        i = bisect_left(tails, x)    # strict increase; bisect_right allows equal values
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4


def count_in_range(a, lo, hi):       # how many lo <= x <= hi in sorted a
    return bisect_right(a, hi) - bisect_left(a, lo)


assert count_in_range([1, 3, 3, 5, 8], 3, 5) == 3

ends = [3, 5, 9]                     # sorted end times
assert bisect_right(ends, 5) - 1 == 1    # last index with ends[j] <= 5
j = bisect_right(ends, 2) - 1
assert j == -1                       # "none": but ends[-1] is 9, no error raised
assert ends[j] == 9                  # so always check j >= 0 first

a = [1, 4]
insort(a, 2)                         # O(log N) search + O(N) shift
assert a == [1, 2, 4]

jobs = [(1, "a"), (4, "b"), (7, "c")]    # sorted by start
k = bisect_left(jobs, 4, key=lambda job: job[0])   # 3.10+; x is a KEY, not a record
assert jobs[k] == (4, "b")
```

### Sorting: key, stability, comparators

```python
from functools import cmp_to_key

intervals = [[1, 3], [8, 10], [2, 6], [15, 18]]
intervals.sort(key=lambda iv: iv[0])     # in place; returns None
merged = []
for s, e in intervals:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
assert merged == [[1, 6], [8, 10], [15, 18]]

counts = {"i love you": 5, "island": 3, "ironman": 2, "i love leetcode": 2}
top3 = sorted(counts, key=lambda s: (-counts[s], s))[:3]   # count desc, text asc
assert top3 == ["i love you", "island", "i love leetcode"]

rows = [("amy", 90), ("cat", 95), ("bob", 90)]
rows.sort(key=lambda r: r[0], reverse=True)   # strings cannot be negated: sort twice,
rows.sort(key=lambda r: r[1])                 # secondary key first (stability keeps it)
assert rows == [("bob", 90), ("amy", 90), ("cat", 95)]   # score asc, then name desc

pairs = [("b", 2), ("a", 1), ("c", 2)]
assert sorted(pairs, key=lambda p: p[1], reverse=True) == [("b", 2), ("c", 2), ("a", 1)]


def compare(a, b):                   # 3-way: negative means a goes first
    if a + b > b + a:
        return -1
    if a + b < b + a:
        return 1
    return 0


def largest_number(nums):
    s = "".join(sorted(map(str, nums), key=cmp_to_key(compare)))
    return "0" if s[0] == "0" else s


assert largest_number([3, 30, 34, 5, 9]) == "9534330"
assert largest_number([0, 0]) == "0"
assert max([3, 7, 7], key=lambda x: -x) == 3   # min/max take key= too
```

### Memoization and deep recursion

```python
import sys
from functools import cache


def longest_increasing_path(matrix):
    rows, cols = len(matrix), len(matrix[0])

    @cache                           # defined inside: a fresh cache per call
    def dfs(r, c):
        best = 1
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                best = max(best, 1 + dfs(nr, nc))
        return best

    return max(dfs(r, c) for r in range(rows) for c in range(cols))


assert longest_increasing_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]) == 4


def word_break(s, words):
    words = set(words)

    @cache
    def ok(i):                       # cache on an int index, not on a list
        if i == len(s):
            return True
        return any(s[i:j] in words and ok(j) for j in range(i + 1, len(s) + 1))

    return ok(0)


assert word_break("leetcode", ["leet", "code"])
assert not word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])

assert sys.getrecursionlimit() == 1000


def subtree_sizes(n, edges, root=0):
    """Iterative DFS: works on a 10^5-node chain where recursion fails."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    parent = [-1] * n
    seen = [False] * n
    seen[root] = True
    order, stack = [], [root]
    while stack:
        u = stack.pop()
        order.append(u)              # pre-order
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                stack.append(v)
    size = [1] * n
    for u in reversed(order):        # reversed pre-order: children before parents
        if parent[u] != -1:
            size[parent[u]] += size[u]
    return size


n = 100_000
sizes = subtree_sizes(n, [(i, i + 1) for i in range(n - 1)])
assert sizes[0] == n and sizes[-1] == 1
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `vector<int> v; v.push_back(x)` | `v = []; v.append(x)` | O(1) amortized |
| `v.back()` | `v[-1]` | O(1) |
| `v.pop_back()` (returns void) | `v.pop()` (returns the item) | O(1) |
| `v.insert(v.begin(), x)` | `v.insert(0, x)` | O(N) |
| `v.erase(v.begin())` | `v.pop(0)` or `del v[0]` | O(N) |
| `vector<int> b = a;` (copy) | `b = a[:]` / `a.copy()` (`b = a` is an alias) | O(N) |
| `vector<vector<int>> g(r, vector<int>(c))` | `[[0] * c for _ in range(r)]` | O(R * C) |
| `std::find(v.begin(), v.end(), x)` | `x in v` / `v.index(x)` | O(N) |
| `std::reverse(v.begin(), v.end())` | `v.reverse()` / `v[::-1]` (copy) | O(N) |
| `std::stack<int>` | `list`: `append`, `pop`, `[-1]` | O(1) |
| `std::queue<int>` | `deque`: `append`, `popleft`, `[0]` | O(1) |
| `std::deque` `push_front` / `pop_front` | `appendleft` / `popleft` | O(1) |
| `priority_queue<int, vector<int>, greater<int>>` | `heapq` on a list (min-heap) | push / pop O(log N), top `h[0]` O(1) |
| `priority_queue<int>` (max-heap) | `heapq` with negated values | O(log N) |
| `std::make_heap` (max-heap) | `heapq.heapify(xs)` (min-heap) | O(N) |
| `partial_sort` / `nth_element` for top k | `heapq.nlargest(k, xs)` / `nsmallest` | O(N log k) |
| `unordered_map<K, V>` | `dict` | average O(1) |
| `m[k]++` (inserts 0 first) | `Counter` / `defaultdict(int)`: `cnt[k] += 1` | average O(1) |
| `unordered_map<int, vector<int>>` adjacency | `defaultdict(list)` | average O(1) per append |
| `m.count(k)` / `m.find(k)` | `k in d` / `d.get(k)` | average O(1) |
| `std::map` (sorted keys) | no stdlib tree: `sorted(d)` when needed | O(N log N) per sort |
| `unordered_set<int>` | `set` | average O(1) |
| `set<pair<int, int>>` | `set` of tuples (hashed, not a tree) | average O(1) |
| `pair` / `tuple`, `std::tie` | `tuple`, unpacking `a, b = t` | O(1) |
| `std::swap(a, b)` | `a, b = b, a` | O(1) |
| `std::lower_bound` | `bisect.bisect_left` (returns an index) | O(log N) |
| `std::upper_bound` | `bisect.bisect_right` | O(log N) |
| `v.insert(upper_bound(...), x)` | `bisect.insort(v, x)` | O(N) |
| `std::sort` (not stable) | `v.sort()` / `sorted(v)` (stable Timsort) | O(N log N) |
| `std::stable_sort` | `v.sort()` / `sorted(v)` | O(N log N) |
| `sort(..., greater<int>())` | `v.sort(reverse=True)` | O(N log N) |
| bool comparator `cmp(a, b)` | `key=` or `cmp_to_key(three_way_cmp)` | O(N log N) |
| `ListNode*`, `nullptr`, `p == q` | object reference, `None`, `p is q` | O(1) |
| recursion + `unordered_map` memo | `@functools.cache` | average O(1) per lookup |
| `INT_MAX`, `LLONG_MAX` as infinity | `math.inf` (ints never overflow) | O(1) |
| `string s; s += c;` in a loop | `parts.append(c)` then `"".join(parts)` | O(N) total |

## Complexity

| Operation | Time | Space / note |
|---|---|---|
| `list.append`, `list.pop()` | O(1) amortized | |
| `list.pop(0)`, `list.insert(0, x)`, `del lst[0]` | O(N) | shifts every element |
| `lst[i]`, `lst[i] = x`, `len(lst)` | O(1) | |
| `x in lst`, `lst.index(x)`, `lst.count(x)` | O(N) | linear scan |
| `lst[i:j]`, `lst[:]`, `lst.copy()` | O(j - i), O(N) | new list |
| `deque.append / appendleft / pop / popleft` | O(1) | |
| `deque[i]` | O(1) at the ends, O(N) in the middle | no slicing |
| `heapq.heappush / heappop / heappushpop` | O(log N) | |
| `h[0]` (peek) | O(1) | |
| `heapq.heapify(xs)` | O(N) | in place |
| `heapq.nlargest(k, xs)` / `nsmallest` | O(N log k) | O(k) extra |
| `dict` get / set / `in` / `del` | O(1) average, O(N) worst | |
| `Counter(xs)` | O(N) | |
| `Counter.most_common(k)` | O(N log k); no argument: O(N log N) | |
| `set` add / remove / `in` | O(1) average | |
| Set union `a \| b` / intersection `a & b` | O(len(a) + len(b)) / O(min(len(a), len(b))) | new set |
| `bisect_left`, `bisect_right` | O(log N) | |
| `insort` | O(N) | the insert shifts |
| `sorted(xs)`, `xs.sort()` | O(N log N); O(N) on already sorted input | O(N) extra |
| `s += c` on `str` in a loop | can degrade to O(N^2) | use `"".join(parts)`: O(N) |
| `@cache` call that hits | O(1) average | O(number of distinct arguments) |
| Recursion | each frame costs memory | default limit 1000 frames |

## Gotchas for C++ developers

- `list.pop(0)` and `insert(0, x)` are **O(N)**. A BFS built on them can degrade to O(N^2). Use `deque.popleft()`.
- `x in some_list` is a linear scan. Build a `set` once if you test membership in a loop.
- `b = a` makes an alias, not a copy. Copy with `a[:]`; nested data needs `copy.deepcopy`.
- Backtracking: `res.append(path)` stores the same list every time. Append `path[:]`.
- `[[0] * c] * r` repeats one row object. Build grids with a comprehension.
- A default argument `def f(acc=[])` is created once and shared by every call. Use `acc=None`.
- `heapq` is a **min-heap**; `priority_queue` is a max-heap. Negate for max, and negate back on read.
- On equal priorities `heapq` compares the next tuple field. Put a unique counter before any unorderable item.
- `d[k]` on a plain dict raises **KeyError**. Reading a missing key from a `defaultdict` inserts it.
- Adding or removing dict keys while looping over the dict raises RuntimeError. Loop over `list(d)`.
- `Counter` keeps keys whose count drops to 0. `del cnt[x]` if `len(cnt)` must count distinct items.
- `{}` is an empty dict. An empty set is `set()`.
- Lists are unhashable; tuples are hashable only when all their items are.
- `xs.sort()` returns None, so `xs = xs.sort()` loses the list. `sorted(xs)` returns a new list.
- `bisect_right(a, x) - 1` can be -1, and `a[-1]` is the last item, not an error. Check `j >= 0`.
- `deque((r, c))` holds two ints. Write `deque([(r, c)])`.
- Tuple assignment evaluates the right side first, but targets are assigned left to right: `curr.next` must come before `curr`.
- Recursion limit is 1000. `sys.setrecursionlimit` helps plain recursion only.
  Deep `@cache` recursion still fails: RecursionError a few thousand levels deep (3.12+), a hard crash (3.11).
- Strings are immutable: `s += c` in a loop can be O(N^2). Collect parts, then `"".join(parts)`.
- Ints never overflow: no `long long`, no `INT_MAX`. Use `math.inf` as infinity.
- `//` floors (`-7 // 2 == -4`) and `%` takes the divisor's sign (`-7 % 2 == 1`). C++ truncates: -3 and -1.
- `for _ in range(len(q))` reads `len(q)` once, so a BFS level loop works while you append.
- `x is None` for null checks. Never use `is` to compare numbers or strings.

## In the Google bank

- **list as stack:** G-739 Daily Temperatures, G-503 Next Greater Element II, G-84 Largest Rectangle in Histogram.
- **deque BFS:** G-200 Number of Islands, G-994 Rotting Oranges, G-127 Word Ladder, G-199 Binary Tree Right Side View.
- **deque window:** G-239 Sliding Window Maximum, G-862 Shortest Subarray with Sum at Least K.
- **heapq:** G-215 Kth Largest Element in an Array, G-347 Top K Frequent Elements, G-23 Merge k Sorted Lists.
- **heapq + tuples:** G-743 Network Delay Time, G-1834 Single-Threaded CPU.
- **two heaps:** G-295 Find Median from Data Stream.
- **dict / defaultdict / Counter:** G-133 Clone Graph, G-721 Accounts Merge, G-621 Task Scheduler.
- **set of tuples / set algebra:** G-417 Pacific Atlantic Water Flow (`pacific & atlantic`).
- **set of ids:** G-827 Making A Large Island (distinct neighbour islands per flipped 0).
- **classes and pointers:** G-23 Merge k Sorted Lists, G-236 Lowest Common Ancestor of a Binary Tree, G-297 Serialize and Deserialize Binary Tree.
- **bisect:** G-300 Longest Increasing Subsequence, G-1235 Maximum Profit in Job Scheduling.
- **sorting with key:** G-56 Merge Intervals, G-987 Vertical Order Traversal of a Binary Tree, G-642 Design Search Autocomplete System.
- **@cache memoization:** G-329 Longest Increasing Path in a Matrix, G-139 Word Break, G-312 Burst Balloons.
- **deep recursion risk:** G-200 Number of Islands (300 x 300 grid), G-1192 Critical Connections in a Network.
