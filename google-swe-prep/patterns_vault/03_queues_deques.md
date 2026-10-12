# Queues, Deques and BFS

**Use when:**
- **Shortest path**, fewest steps or minimum moves in an **unweighted** graph or grid.
- **Level by level**, "spread", "rot", "infect", "distance to the nearest X" (multi-source BFS).
- Edge weights are only **0 or 1** (0-1 BFS instead of Dijkstra).
- **Sliding window max / min**, or "shortest subarray with sum at least K" with negatives (monotonic deque).

## Python template(s)

### `collections.deque` basics

- `append`, `appendleft`, `pop`, `popleft` are all **O(1)**. `list.pop(0)` is **O(N)**: never use a list as a queue.
- `dq[0]` and `dq[-1]` are O(1). Indexing toward the **middle** is O(N), unlike `std::deque`.

```python
from collections import deque

q = deque([1, 2])                # deque takes an iterable
q.append(3)                      # push_back
q.appendleft(0)                  # push_front
assert list(q) == [0, 1, 2, 3]
assert q[0] == 0 and q[-1] == 3  # front(), back()
assert q.popleft() == 0          # pop_front, returns the value
assert q.pop() == 3              # pop_back (the RIGHT end), returns the value
assert len(q) == 2 and q         # non-empty deque is truthy

window = deque(maxlen=3)         # ring buffer: the oldest item drops out
for x in range(5):
    window.append(x)
assert list(window) == [2, 3, 4]

r = deque([1, 2, 3, 4])
r.rotate(1)                      # rotate right by 1
assert list(r) == [4, 1, 2, 3]

assert len(deque((2, 5))) == 2   # two ints, not one tuple
assert deque([(2, 5)])[0] == (2, 5)

try:
    deque().popleft()
except IndexError:
    pass
else:
    raise AssertionError("popleft on an empty deque must raise")
```

### BFS on a graph: mark visited on push

- Mark a node **when you enqueue it**. Then each node enters the queue at most once: O(V + E).
- Need the level? Process `for _ in range(len(q))`: `len(q)` is read once, like `int sz = q.size()`.

```python
from collections import deque


def bfs_dist(adj, src):
    dist = {src: 0}              # doubles as the visited set
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1    # mark on push
                q.append(v)
    return dist

adj = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
assert bfs_dist(adj, 0) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}


def shortest_path(adj, src, dst):   # rebuild the path from a parent map
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            path = []
            while u is not None:     # not `while u:` (node 0 is falsy)
                path.append(u)
                u = parent[u]
            return path[::-1]
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    return []

assert shortest_path(adj, 0, 4) == [0, 1, 3, 4]
assert shortest_path(adj, 4, 0) == []


def word_ladder(begin, end, words):  # G-127: number of words on the shortest path
    words = set(words)
    if end not in words:
        return 0
    q = deque([begin])
    seen = {begin}
    steps = 1
    while q:
        for _ in range(len(q)):      # exactly one level
            w = q.popleft()
            if w == end:
                return steps
            for i in range(len(w)):
                for c in "abcdefghijklmnopqrstuvwxyz":
                    nxt = w[:i] + c + w[i + 1:]
                    if nxt in words and nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
        steps += 1
    return 0

assert word_ladder("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
assert word_ladder("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
```

### BFS on a grid

- Neighbours from a `DIRS` tuple. Bounds check with chained comparisons: `0 <= nr < R`.
- Mark on push: write the distance, flip the cell, or add to `seen`.

```python
from collections import deque

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def shortest_path_grid(grid):    # 0 = open, 1 = wall; steps from top-left to bottom-right
    R, C = len(grid), len(grid[0])
    if grid[0][0] or grid[R - 1][C - 1]:
        return -1
    dist = [[-1] * C for _ in range(R)]  # -1 = not visited
    dist[0][0] = 0
    q = deque([(0, 0)])
    while q:
        r, c = q.popleft()
        if (r, c) == (R - 1, C - 1):
            return dist[r][c]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0 and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return -1

assert shortest_path_grid([
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
]) == 4
assert shortest_path_grid([[0, 1], [1, 0]]) == -1
assert shortest_path_grid([[0]]) == 0


def num_islands(grid):           # G-200 with BFS: no recursion limit
    R, C = len(grid), len(grid[0])
    seen = [[False] * C for _ in range(R)]
    count = 0
    for sr in range(R):
        for sc in range(C):
            if grid[sr][sc] == "1" and not seen[sr][sc]:
                count += 1
                seen[sr][sc] = True
                q = deque([(sr, sc)])
                while q:
                    r, c = q.popleft()
                    for dr, dc in DIRS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == "1" and not seen[nr][nc]:
                            seen[nr][nc] = True
                            q.append((nr, nc))
    return count

assert num_islands([
    ["1", "1", "0", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "1", "0", "0"],
    ["0", "0", "0", "1", "1"],
]) == 3
```

### Multi-source BFS

- Push **every source** at distance 0 before the loop. One pass gives each cell its distance to the **nearest** source.
- O(R * C) total, not O(sources * R * C).

```python
from collections import deque

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def oranges_rotting(grid):       # G-994: minutes until no fresh orange, -1 if impossible
    R, C = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2:
                q.append((r, c))         # every rotten orange is a source
            elif grid[r][c] == 1:
                fresh += 1
    minutes = 0
    while q and fresh:
        for _ in range(len(q)):          # one minute = one level
            r, c = q.popleft()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1:
                    grid[nr][nc] = 2     # mark on push
                    fresh -= 1
                    q.append((nr, nc))
        minutes += 1
    return -1 if fresh else minutes

assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
assert oranges_rotting([[0, 2]]) == 0


def update_matrix(mat):          # 01 Matrix: distance to the nearest 0
    R, C = len(mat), len(mat[0])
    dist = [[-1] * C for _ in range(R)]
    q = deque()
    for r in range(R):
        for c in range(C):
            if mat[r][c] == 0:
                dist[r][c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist

assert update_matrix([[0, 0, 0], [0, 1, 0], [1, 1, 1]]) == [[0, 0, 0], [0, 1, 0], [1, 2, 1]]
```

### 0-1 BFS

- Weight 0 edge: `appendleft`. Weight 1 edge: `append`. The deque stays sorted by distance.
- Relax like Dijkstra (`dist[u] + w < dist[v]`), not visited-on-push. O(V + E), no heap.

```python
import heapq
import math
from collections import deque


def zero_one_bfs(n, edges, src):     # directed edges (u, v, w) with w in {0, 1}
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [math.inf] * n
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)     # same distance: front
                else:
                    dq.append(v)         # one more: back
    return dist


def dijkstra(n, edges, src):         # reference answer, O((V + E) log V)
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [math.inf] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                     # stale entry
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist

edges = [(0, 1, 1), (0, 2, 0), (2, 1, 0), (1, 3, 1), (2, 3, 1), (3, 4, 0)]
assert zero_one_bfs(6, edges, 0) == [0, 0, 0, 1, 1, math.inf]
assert zero_one_bfs(6, edges, 0) == dijkstra(6, edges, 0)

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def min_obstacles(grid):         # Minimum Obstacle Removal: stepping onto a 1 costs 1
    R, C = len(grid), len(grid[0])
    dist = [[math.inf] * C for _ in range(R)]
    dist[0][0] = 0
    dq = deque([(0, 0)])
    while dq:
        r, c = dq.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                w = grid[nr][nc]
                if dist[r][c] + w < dist[nr][nc]:
                    dist[nr][nc] = dist[r][c] + w
                    if w == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))
    return dist[R - 1][C - 1]

assert min_obstacles([[0, 1, 1], [1, 1, 0], [1, 1, 0]]) == 2
assert min_obstacles([[0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]]) == 0
```

### Monotonic deque: sliding window maximum (G-239)

- Deque of **indexes**; values **decreasing** front to back. The front is the window max.
- New `x`: pop the back while it is `<= x` (it can never win again). Pop the front once it leaves the window.

```python
from collections import deque


def max_sliding_window(nums, k):
    dq = deque()                 # indexes; nums[dq[...]] decreasing
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()             # smaller and older: never a max again
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()         # front slid out of the window
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out

assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert max_sliding_window([1], 1) == [1]
assert max_sliding_window([9, 8, 7, 6], 2) == [9, 8, 7]
assert max_sliding_window([4, 4, 4], 2) == [4, 4]

nums = [5, 2, 8, 1, 9, 3, 3, 7, 0, 6]
for k in range(1, len(nums) + 1):    # check against brute force
    assert max_sliding_window(nums, k) == [max(nums[i:i + k]) for i in range(len(nums) - k + 1)]
```

### Monotonic deque: shortest subarray with sum >= K (G-862)

- **Negatives** break the sliding window. Use prefix sums `P`: find the smallest `j - i` with `P[j] - P[i] >= K`.
- Deque of start indexes with **increasing** `P`. Pop the front while `P[j] - P[front] >= K`; record `j - front`.
- Pop the back while `P[back] >= P[j]`: start `j` is later and no larger, so it is always at least as good.

```python
import math
import random
from collections import deque
from itertools import accumulate


def shortest_subarray(nums, k):
    P = [0, *accumulate(nums)]   # P[j] = sum(nums[:j])
    dq = deque()                 # start indexes; P increasing
    best = math.inf
    for j, pj in enumerate(P):
        while dq and pj - P[dq[0]] >= k:
            best = min(best, j - dq.popleft())   # a later j would only be longer
        while dq and P[dq[-1]] >= pj:
            dq.pop()             # j starts later with a smaller or equal sum: better
        dq.append(j)
    return -1 if best == math.inf else best

assert shortest_subarray([1], 1) == 1
assert shortest_subarray([1, 2], 4) == -1
assert shortest_subarray([2, -1, 2], 3) == 3
assert shortest_subarray([84, -37, 32, 40, 95], 167) == 3


def brute(nums, k):
    best = math.inf
    for i in range(len(nums)):
        s = 0
        for j in range(i, len(nums)):
            s += nums[j]
            if s >= k:
                best = min(best, j - i + 1)
                break
    return -1 if best == math.inf else best

rng = random.Random(7)
for _ in range(300):
    arr = [rng.randint(-5, 9) for _ in range(rng.randint(1, 9))]
    k = rng.randint(1, 15)
    assert shortest_subarray(arr, k) == brute(arr, k)
```

C++ side by side:

```cpp
#include <cassert>
#include <deque>
#include <queue>
#include <utility>
#include <vector>

std::vector<int> maxSlidingWindow(const std::vector<int>& nums, int k) {
    std::deque<int> dq;                          // indexes
    std::vector<int> out;
    for (int i = 0; i < (int)nums.size(); ++i) {
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);
        if (dq.front() <= i - k) dq.pop_front();
        if (i >= k - 1) out.push_back(nums[dq.front()]);
    }
    return out;
}

int main() {
    // std::queue: read front(), then pop(); pop() returns void.
    std::queue<std::pair<int, int>> q;
    q.push({0, 0});
    q.push({0, 1});
    auto [r, c] = q.front();
    q.pop();
    assert(r == 0 && c == 0 && q.size() == 1);

    std::vector<int> want{3, 3, 5, 5, 6, 7};
    assert(maxSlidingWindow({1, 3, -1, -3, 5, 3, 6, 7}, 3) == want);
    return 0;
}
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `std::queue<int> q;` | `q = deque()` | O(1) |
| `q.push(x)` | `q.append(x)` | O(1) |
| `int x = q.front(); q.pop();` | `x = q.popleft()` | O(1) |
| `q.empty()` / `q.size()` | `not q` / `len(q)` | O(1) |
| `std::deque<int> dq;` | `dq = deque()` | O(1) |
| `dq.push_front(x)` / `dq.push_back(x)` | `dq.appendleft(x)` / `dq.append(x)` | O(1) |
| `dq.pop_front()` / `dq.pop_back()` | `dq.popleft()` / `dq.pop()` (both return the value) | O(1) |
| `dq.front()` / `dq.back()` | `dq[0]` / `dq[-1]` | O(1) |
| `dq[i]` (random access) | `dq[i]` | C++ O(1); CPython O(1) at the ends, O(N) in the middle |
| `q.push({r, c})` with `pair<int, int>` | `q.append((r, c))` | O(1) |
| `auto [r, c] = q.front(); q.pop();` | `r, c = q.popleft()` | O(1) |
| `int sz = q.size(); while (sz--)` | `for _ in range(len(q)):` | O(1) per step |
| `vector<vector<int>> dist(R, vector<int>(C, -1))` | `[[-1] * C for _ in range(R)]` | O(R * C) |
| `int dr[] = {1, -1, 0, 0}, dc[] = {0, 0, 1, -1};` | `DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))` | O(1) |
| `unordered_set<int> visited` | `set()`, or a `dist` dict | O(1) average |
| `priority_queue` Dijkstra on 0/1 weights | 0-1 BFS with `deque` | O(V + E) instead of O((V + E) log V) |
| `vector::erase(v.begin())` as a queue pop | never; `list.pop(0)` is the same O(N) trap | O(N) |

## Complexity

| Operation | Time | Space |
|---|---|---|
| `append`, `appendleft`, `pop`, `popleft` | O(1) | O(1) |
| `dq[0]`, `dq[-1]`, `len(dq)` | O(1) | O(1) |
| `dq[i]` toward the middle | O(N) | O(1) |
| `x in dq` | O(N) | O(1) |
| `deque(maxlen=k).append(x)` | O(1), drops the oldest | O(k) |
| `dq.rotate(k)` | O(k) | O(1) |
| `list.pop(0)`, `list.insert(0, x)` | O(N) | O(1) |
| BFS on a graph (adjacency list) | O(V + E) | O(V) |
| BFS on an R x C grid | O(R * C) | O(R * C) |
| Multi-source BFS | O(R * C) total | O(R * C) |
| 0-1 BFS | O(V + E) | O(V) |
| Dijkstra with `heapq` (lazy deletion) | O((V + E) log V) | O(V + E) |
| Sliding window maximum | O(N): each index pushed and popped once | O(k) |
| Shortest subarray with sum >= K | O(N) | O(N) |

## Gotchas for C++ developers

- `list.pop(0)` is **O(N)**. A BFS that uses a list as its queue can degrade to O(V^2). Use `deque.popleft()`.
- **Mark on push.** Marking on pop still gives correct distances, but a node can be queued once per incoming edge.
- Marking on pop also breaks counters such as `fresh -= 1` in G-994: the same cell is counted twice.
- `deque.pop()` removes the **right** end. `std::queue::pop()` removes the front; the Python twin is `popleft()`.
- No `.front()`, `.back()`, `.top()` or `.empty()`. Use `dq[0]`, `dq[-1]`, `not dq`.
- `dq[0]` on an empty deque raises `IndexError`: write `while dq and nums[dq[-1]] <= x:`.
- `deque((r, c))` is a deque of **two ints**. Start a queue of cells with `deque([(r, c)])`.
- **Negative indexes do not raise**: `grid[-1][c]` reads the last row. Always check `0 <= nr < R` first.
- Changing a deque while iterating over it raises `RuntimeError`. Loop with `while q:` and `popleft()`.
- 0-1 BFS: a node can be pushed more than once. Do not add a visited-on-push set; compare distances.
- Monotonic deque stores **indexes**: a value alone cannot tell you when it left the window.
- `queue.Queue` is a **thread-safe** queue with locks: slower, and `get()` blocks on empty. Use `deque`.

## In the Google bank

- **G-200 Number of Islands**: BFS flood fill per unvisited land cell; no recursion limit.
- **G-994 Rotting Oranges**: multi-source BFS; one level per minute; count `fresh` on push.
- **G-2812 Find the Safest Path in a Grid**: multi-source BFS from every thief, then a max-min path.
- **G-417 Pacific Atlantic Water Flow**: multi-source BFS uphill from each ocean's border cells.
- **G-127 Word Ladder**: level-order BFS; words in a `set`, mark on push.
- **G-126 Word Ladder II**: BFS layer by layer building a parent map, then backtrack every path.
- **G-815 Bus Routes**: BFS where the nodes are routes; each route is visited once.
- **G-787 Cheapest Flights Within K Stops**: BFS by stops, at most K + 1 levels.
- **G-2045 Second Minimum Time to Reach Destination**: BFS keeping two distinct arrival counts per node.
- **G-863 All Nodes Distance K in Binary Tree**: parent map, then BFS K levels from the target.
- **G-199 Binary Tree Right Side View**: level order; take the last node of each level.
- **G-103 Binary Tree Zigzag Level Order Traversal**: level order; `appendleft` on odd levels.
- **G-207 Course Schedule** and **G-210 Course Schedule II**: Kahn's algorithm, a queue of in-degree 0 nodes.
- **G-310 Minimum Height Trees**: queue of leaves, peeled layer by layer.
- **G-239 Sliding Window Maximum**: decreasing deque of indexes.
- **G-862 Shortest Subarray with Sum at Least K**: prefix sums + increasing deque of indexes.
