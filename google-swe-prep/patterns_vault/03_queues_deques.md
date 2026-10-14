# Queues, deques and BFS

> In one line: BFS uses a queue to visit places in rings around the start, so you reach each place first by the fewest steps.

## When to use it
- **Fewest steps**, shortest path or minimum moves, when every step costs the same.
- A **grid** (a 2D list of cells) where you move up, down, left and right.
- "Level by level", "spread", "rot", "infect", or "distance to the nearest X".
- "Count the islands" or "how many connected areas".
- Sliding window **maximum** or minimum.

## The idea in plain words
A **queue** is a line at a shop. The first person in is the first person out: **FIFO** (first in, first out).
A **deque** ("deck", double-ended queue) can add and remove at both ends.
In Python both come from `collections.deque`. `append(x)` adds at the back, `popleft()` takes from the front.
Both are O(1): the same speed for any size.
Do not use a list as a queue: `list.pop(0)` moves every other item one place, so it is O(N).

A **graph** is a set of **nodes** (points) joined by **edges** (lines).
A grid is a graph too: each cell is a node, joined to its 4 neighbours.

**BFS** (breadth-first search) explores in rings, like ripples in water: first every place 1 step away, then 2 steps, and so on.
1. Put the start in the queue. Mark it as **visited** (seen) with distance 0.
2. Take the front item out with `popleft()`.
3. For each neighbour that is not visited yet: mark it, give it distance + 1, add it to the back.
4. Repeat until the queue is empty.

Mark a place when you **add** it to the queue, not when you take it out.
Then each place enters the queue only once, and the work is O(number of places + number of edges).

**Multi-source BFS**: many start points, like all rotten oranges at once.
Put every start point in the queue first, all at distance 0. One BFS then finds the distance to the **nearest** start.

**Level by level**: `for _ in range(len(queue)):` takes exactly the items of one ring.
`len(queue)` is read once, before the loop starts. The name `_` means "a counter we do not use".

**Monotonic deque** (for a sliding window maximum): **monotonic** means "always increasing or always decreasing".
Keep window indexes whose values always go down from front to back. Then the front is the window's maximum.

## Python template

### deque basics

```python
from collections import deque

queue = deque([1, 2])
queue.append(3)
print(queue.popleft())  # 1
print(list(queue))  # [2, 3]
queue.appendleft(0)
print(queue[0], queue[-1])  # 0 3
# pop() takes from the RIGHT end
print(queue.pop())  # 3
```

### BFS on a graph

```python
from collections import deque

def steps_from(graph, start):
    # dist is also the visited set: a node is visited once it has a distance
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

# graph[u] lists the neighbours of node u
graph = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
print(steps_from(graph, 0))  # {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
```

### BFS on a grid

```python
from collections import deque

# the 4 moves as (row change, column change): down, up, right, left
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def fewest_steps(grid):
    # 0 = open, 1 = wall; steps from top-left to bottom-right, or -1
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == 1:
        return -1
    # -1 means "not visited yet"; a new list for each row
    dist = [[-1] * cols for _ in range(rows)]
    dist[0][0] = 0
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            # 0 <= nr < rows means: 0 <= nr and nr < rows
            inside = 0 <= nr < rows and 0 <= nc < cols
            if inside and grid[nr][nc] == 0 and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return dist[rows - 1][cols - 1]

grid = [[0, 0, 0],
        [1, 1, 0],
        [0, 0, 0]]
print(fewest_steps(grid))  # 4
```

### Multi-source BFS, level by level

```python
from collections import deque

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def minutes_to_rot(grid):
    # 0 = empty, 1 = fresh orange, 2 = rotten orange
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1
    minutes = 0
    while queue and fresh > 0:
        # one level = one minute
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    queue.append((nr, nc))
        minutes += 1
    if fresh > 0:
        return -1
    return minutes

print(minutes_to_rot([[2, 1, 1], [1, 1, 0], [0, 1, 1]]))  # 4
print(minutes_to_rot([[2, 1, 1], [0, 1, 1], [1, 0, 1]]))  # -1
```

### Monotonic deque: sliding window maximum

```python
from collections import deque

def window_max(nums, k):
    # indexes; their values go down from front to back, so the front is the max
    dq = deque()
    result = []
    for i in range(len(nums)):
        # smaller values at the back can never be the max again
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        # the front index has slid out of the window
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

print(window_max([1, 3, -1, -3, 5, 3, 6, 7], 3))  # [3, 3, 5, 5, 6, 7]
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `queue<int> q; q.push(x);` | `q = deque()` then `q.append(x)` | first `from collections import deque` |
| `int x = q.front(); q.pop();` | `x = q.popleft()` | `popleft()` returns the item |
| `q.empty()` / `q.size()` | `not q` / `len(q)` | an empty deque counts as False |
| `dq.push_front(x);` / `dq.pop_back();` | `dq.appendleft(x)` / `dq.pop()` | `pop()` takes from the right end |
| `dq.front()` / `dq.back()` | `dq[0]` / `dq[-1]` | read without removing |
| `q.push({r, c});` | `q.append((r, c))` | a tuple holds the pair |
| `auto [r, c] = q.front(); q.pop();` | `r, c = q.popleft()` | unpack the tuple |
| `vector<vector<int>> d(R, vector<int>(C, -1));` | `d = [[-1] * C for _ in range(R)]` | a new row each time |
| `int sz = q.size(); while (sz--) { ... }` | `for _ in range(len(q)):` | exactly one BFS level |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| `append`, `appendleft`, `pop`, `popleft` on a deque | O(1) | the same speed for any size |
| `list.pop(0)` | O(N) | every other item moves one place left |
| BFS on a graph | O(V + E) | V nodes and E edges: each is looked at once |
| BFS on a grid with R rows and C columns | O(R × C) | each cell enters the queue at most once |
| Multi-source BFS | O(R × C) | one BFS for all start points, not one per start |
| Sliding window maximum with a deque | O(N) | each index enters and leaves the deque once |

## Common mistakes
- Mistake: `queue.pop(0)` on a list. Each call is O(N). Fix: use a `deque` and `popleft()`.
- Mistake: `dq.pop()` to take the front, like C++ `q.pop()`. Python `pop()` takes the right end. Fix: `popleft()`.
- Mistake: marking a cell as visited when you take it out. It can enter the queue many times. Fix: mark it when you add it.
- Mistake: `[[-1] * cols] * rows`. All rows are the same list, so one change shows in every row. Fix: `[[-1] * cols for _ in range(rows)]`.
- Mistake: `deque((0, 0))` makes a deque of two numbers. Fix: `deque([(0, 0)])` holds one pair.
- Mistake: reading `grid[nr][nc]` before the bounds check. Index -1 does not fail: it reads the last row. Fix: check `0 <= nr < rows` first.
- Mistake: plain BFS when steps cost different amounts. BFS counts steps, not costs. Fix: use Dijkstra with a heap (notes 04 and 09).

## Practice in the app
- **G-200 Number of Islands**: for each land cell not yet visited, count one island and BFS to mark all of it.
- **G-994 Rotting Oranges**: the multi-source template above; one level is one minute.
- **G-127 Word Ladder**: each word is a node; change one letter to get its neighbours; BFS gives the fewest changes.
- **G-417 Pacific Atlantic Water Flow**: BFS from each ocean's edge cells, moving only to cells that are as high or higher; the answer is the cells reached from both oceans.
- **G-2812 Find the Safest Path in a Grid**: a multi-source BFS from all thieves gives each cell its safety; then find the path whose lowest safety is highest.
- **G-199 Binary Tree Right Side View**: BFS on the tree level by level; keep the last node of each level.
- **G-207 Course Schedule**: a queue of courses whose prerequisites are all done (Kahn's algorithm, note 09).
- **G-239 Sliding Window Maximum**: the monotonic deque template above.
- **G-862 Shortest Subarray with Sum at Least K**: running totals (prefix sums) plus a deque of start indexes whose totals always go up.
