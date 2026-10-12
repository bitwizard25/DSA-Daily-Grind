# Graphs and Union-Find

**Use when:**
- "connected", "islands", "provinces", "grid", "network": **DFS / BFS** or **union-find**.
- "prerequisites", "build order", "dependencies", "can all be finished": **topological sort**.
- "cheapest", "minimum time / effort" with weights: **Dijkstra**; "at most K stops": **Bellman-Ford rounds**.
- "merge accounts", "a / b = 2.0": **union-find** (weighted); "critical connection": **bridges**.

## Python template(s)

### Build the graph: `defaultdict(list)` or a list of lists

- Node labels are strings or sparse ids: `defaultdict(list)`. Labels are `0..n-1`: `[[] for _ in range(n)]`.
- Reading a missing key of a `defaultdict` **inserts** it. Use `in` or `.get` to test without inserting.

```python
from collections import defaultdict

adj = defaultdict(list)              # unordered_map<string, vector<string>>
for u, v in [("a", "b"), ("b", "c"), ("a", "c")]:
    adj[u].append(v)                 # missing key: an empty list is created first
    adj[v].append(u)                 # undirected: add both directions
assert adj["a"] == ["b", "c"]
assert sorted(adj["c"]) == ["a", "b"]

n = 4
adj2 = [[] for _ in range(n)]        # vector<vector<int>> adj(n)
for u, v in [(0, 1), (1, 2)]:
    adj2[u].append(v)
    adj2[v].append(u)
assert adj2[3] == []                 # an isolated node still has an entry

bad = [[]] * 3                       # 3 references to ONE list
bad[0].append(1)
assert bad == [[1], [1], [1]]

directed = defaultdict(list)
directed["x"].append("y")            # "y" has no out-edges, so no key yet
assert "y" not in directed
assert directed.get("z", []) == [] and "z" not in directed   # .get does not insert
_ = directed["y"]                    # a plain read inserts []
assert "y" in directed and len(directed) == 2

g = defaultdict(list)
g["x"].append("y")
try:
    for u in g:
        for v in g[u]:
            _ = g[v]                 # inserts "y" while iterating g
except RuntimeError:                 # dictionary changed size during iteration
    raised = True
else:
    raised = False
assert raised
```

### DFS: iterative stack (components, flood fill)

- An explicit stack has no depth limit. Recursion stops at **1000 frames** by default.
- Mark `seen` when you **push**, so a node is pushed at most once.

```python
import sys


def count_components(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    count = 0
    for s in range(n):
        if seen[s]:
            continue
        count += 1
        seen[s] = True
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return count


assert count_components(5, [(0, 1), (1, 2), (3, 4)]) == 2
assert count_components(3, []) == 3

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def num_islands(grid):               # G-200
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != "1" or seen[r][c]:
                continue
            islands += 1
            seen[r][c] = True
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1" \
                            and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
    return islands


assert num_islands([list("11000"), list("11000"), list("00100"), list("00011")]) == 3
assert num_islands([list("1" * 3000)]) == 1          # deep island, no recursion


def fill_recursive(grid, r, c):      # the textbook version: one frame per cell
    if not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] != "1":
        return 0
    grid[r][c] = "0"
    return 1 + sum(fill_recursive(grid, r + dr, c + dc) for dr, dc in DIRS)


assert fill_recursive([list("111"), list("010")], 0, 0) == 4
assert sys.getrecursionlimit() == 1000
try:
    fill_recursive([list("1" * 3000)], 0, 0)
except RecursionError:
    too_deep = True
else:
    too_deep = False
assert too_deep
```

### BFS: unweighted shortest path

- BFS pops nodes in order of distance, so the first time you reach a node is the shortest way.
- Grid, multi-source and 0-1 BFS templates: note `03`.

```python
from collections import deque


def shortest_path(adj, src, dst):
    parent = {src: None}             # visited set + how we got here
    q = deque([src])
    while q:
        u = q.popleft()              # O(1); list.pop(0) is O(N)
        if u == dst:
            path = []
            while u is not None:     # not "while u": node 0 is falsy
                path.append(u)
                u = parent[u]
            return path[::-1]
        for v in adj[u]:
            if v not in parent:      # mark on push
                parent[v] = u
                q.append(v)
    return []


adj = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
assert shortest_path(adj, 0, 4) in ([0, 1, 3, 4], [0, 2, 3, 4])
assert shortest_path(adj, 0, 0) == [0]
assert shortest_path(adj, 4, 0) == []
```

### Topological sort: Kahn (BFS on in-degrees)

- Push every node with in-degree 0. Pop, append to the order, decrement its neighbours.
- **Cycle check:** `len(order) < n`. Nodes on a cycle never reach in-degree 0.
- **Unique order** (G-444): the queue never holds 2+ nodes. **Semesters**: count BFS layers.

```python
from collections import deque


def find_order(num_courses, prerequisites):          # G-210
    adj = [[] for _ in range(num_courses)]
    indeg = [0] * num_courses
    for course, pre in prerequisites:                 # edge pre -> course
        adj[pre].append(course)
        indeg[course] += 1
    q = deque(i for i in range(num_courses) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == num_courses else []  # leftovers sit on or behind a cycle


def is_topo_order(n, prerequisites, order):
    pos = {u: i for i, u in enumerate(order)}
    return len(order) == n and all(pos[pre] < pos[c] for c, pre in prerequisites)


pre = [[1, 0], [2, 0], [3, 1], [3, 2]]
assert is_topo_order(4, pre, find_order(4, pre))
assert find_order(2, [[1, 0], [0, 1]]) == []          # cycle
assert find_order(1, [[0, 0]]) == []                  # a self-loop is a cycle
assert find_order(3, []) == [0, 1, 2]                 # no edges: any order is valid


def min_semesters(n, prerequisites):                  # G-207 follow-up
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for course, pre in prerequisites:
        adj[pre].append(course)
        indeg[course] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    taken = semesters = 0
    while q:
        semesters += 1
        for _ in range(len(q)):                       # one layer = one semester
            u = q.popleft()
            taken += 1
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    return semesters if taken == n else -1


assert min_semesters(4, pre) == 3
assert min_semesters(3, [[1, 0], [0, 1]]) == -1
```

### Topological sort: DFS 3-colour

- **WHITE** unvisited, **GRAY** on the current path, **BLACK** finished. An edge to GRAY is a cycle.
- A plain `visited` flag is wrong for directed graphs: reaching a BLACK node from another branch is fine.
- Order = **reverse post-order**. Recursive: fine for small V; for V near 10^5 use Kahn.

```python
WHITE, GRAY, BLACK = 0, 1, 2


def topo_dfs(n, edges):
    """Topological order of edges u -> v, or [] if there is a cycle."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    color = [WHITE] * n
    post = []

    def has_cycle(u):
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:                      # back edge
                return True
            if color[v] == WHITE and has_cycle(v):
                return True
        color[u] = BLACK
        post.append(u)                                # all descendants are done
        return False

    for u in range(n):
        if color[u] == WHITE and has_cycle(u):
            return []
    return post[::-1]


def valid(order, edges):
    pos = {u: i for i, u in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in edges)


diamond = [(0, 1), (0, 2), (1, 3), (2, 3)]
assert valid(topo_dfs(4, diamond), diamond)
assert topo_dfs(3, [(0, 1), (1, 2), (2, 0)]) == []
assert topo_dfs(3, [(0, 1), (2, 1)]) != []            # 1 is BLACK when 2 reaches it: no cycle


def has_cycle_wrong(n, edges):                        # BUG: two colours only
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    seen = set()

    def dfs(u):
        seen.add(u)
        return any(v in seen or dfs(v) for v in adj[u])

    return any(u not in seen and dfs(u) for u in range(n))


assert has_cycle_wrong(3, [(0, 1), (2, 1)])           # false alarm on a DAG
```

### Dijkstra with `heapq`: skip stale entries

- Heap of `(dist, node)`. No decrease-key: push a new entry and **skip** pops with `d > dist[u]`.
- Non-negative weights only. The first pop of a node is final, so you may return at the target.

```python
import heapq
import math
from collections import defaultdict


def network_delay_time(times, n, k):                  # G-743, nodes 1..n
    adj = defaultdict(list)
    for u, v, w in times:
        adj[u].append((v, w))
    dist = [math.inf] * (n + 1)
    dist[k] = 0
    heap = [(0, k)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                                  # stale: u was improved after this push
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    ans = max(dist[1:])
    return -1 if ans == math.inf else ans


assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
assert network_delay_time([[1, 2, 1]], 2, 1) == 1
assert network_delay_time([[1, 2, 1]], 2, 2) == -1
assert network_delay_time([[1, 2, 9], [1, 3, 1], [3, 2, 1]], 3, 1) == 2   # cheaper via 3

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def minimum_effort_path(heights):                     # G-1631: path cost = largest step
    rows, cols = len(heights), len(heights[0])
    best = [[math.inf] * cols for _ in range(rows)]
    best[0][0] = 0
    heap = [(0, 0, 0)]                                # (effort, r, c)
    while heap:
        e, r, c = heapq.heappop(heap)
        if (r, c) == (rows - 1, cols - 1):
            return e                                  # first pop of the target is final
        if e > best[r][c]:
            continue
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                ne = max(e, abs(heights[nr][nc] - heights[r][c]))
                if ne < best[nr][nc]:
                    best[nr][nc] = ne
                    heapq.heappush(heap, (ne, nr, nc))
    return -1                                         # unreachable: the target is always reachable


assert minimum_effort_path([[1, 2, 2], [3, 8, 2], [5, 3, 5]]) == 2
assert minimum_effort_path([[1, 2, 3], [3, 8, 4], [5, 3, 5]]) == 1
assert minimum_effort_path([[7]]) == 0
```

The same Dijkstra in C++: `greater<>` turns `priority_queue` into a min-heap.

```cpp
#include <algorithm>
#include <cassert>
#include <climits>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

int networkDelayTime(const std::vector<std::vector<int>>& times, int n, int k) {
    std::vector<std::vector<std::pair<int, int>>> adj(n + 1);
    for (const auto& t : times) adj[t[0]].push_back({t[1], t[2]});
    std::vector<int> dist(n + 1, INT_MAX);
    using P = std::pair<int, int>;                                  // (dist, node)
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;     // min-heap
    dist[k] = 0;
    pq.push({0, k});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();                                                   // returns void
        if (d > dist[u]) continue;                                  // stale
        for (auto [v, w] : adj[u]) {
            if (d + w < dist[v]) {                                  // d is finite here
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }
    int ans = 0;
    for (int i = 1; i <= n; ++i) {
        if (dist[i] == INT_MAX) return -1;
        ans = std::max(ans, dist[i]);
    }
    return ans;
}

int main() {
    assert(networkDelayTime({{2, 1, 1}, {2, 3, 1}, {3, 4, 1}}, 4, 2) == 2);
    assert(networkDelayTime({{1, 2, 1}}, 2, 2) == -1);
    assert(networkDelayTime({{1, 2, 9}, {1, 3, 1}, {3, 2, 1}}, 3, 1) == 2);
    return 0;
}
```

### Bellman-Ford with limited rounds: Cheapest Flights Within K Stops

- **K stops = at most K + 1 edges = K + 1 rounds.** Round `i` finds the best cost using at most `i` edges.
- Relax from a **copy** of the last round. In place, one round can chain several edges.
- Plain Dijkstra by cost is wrong here: a cheap path with too many stops can block a valid one.

```python
import math


def find_cheapest_price(n, flights, src, dst, k):    # G-787
    dist = [math.inf] * n
    dist[src] = 0
    for _ in range(k + 1):
        prev = dist[:]                                # a copy; "prev = dist" is an alias
        for u, v, w in flights:
            if prev[u] + w < dist[v]:                 # inf + w is still inf
                dist[v] = prev[u] + w
    return -1 if dist[dst] == math.inf else dist[dst]


def cheapest_in_place(n, flights, src, dst, k):      # BUG: no copy
    dist = [math.inf] * n
    dist[src] = 0
    for _ in range(k + 1):
        for u, v, w in flights:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w                 # may reuse a value set this round
    return -1 if dist[dst] == math.inf else dist[dst]


flights = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
assert find_cheapest_price(3, flights, 0, 2, 1) == 200
assert find_cheapest_price(3, flights, 0, 2, 0) == 500
assert cheapest_in_place(3, flights, 0, 2, 0) == 200  # wrong: 2 edges with k = 0

flights = [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]]
assert find_cheapest_price(4, flights, 0, 3, 1) == 700
assert find_cheapest_price(2, [[1, 0, 5]], 0, 1, 3) == -1
```

### Union-find: path compression + union by size

- `find` walks to the root, then points every node on the path at the root (**path compression**).
- `union` hangs the smaller tree under the larger (**union by size**). Both together: near O(1) per op.
- `union` returning `False` means "already connected": the edge closes a cycle.

```python
from collections import defaultdict


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))  # iota(parent.begin(), parent.end(), 0)
        self.size = [1] * n
        self.count = n                # number of components

    def find(self, x):                # iterative: no recursion limit
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while x != root:              # second pass: compress the path
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.count -= 1
        return True


def find_circle_num(is_connected):    # G-547
    n = len(is_connected)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j]:
                dsu.union(i, j)
    return dsu.count


assert find_circle_num([[1, 1, 0], [1, 1, 0], [0, 0, 1]]) == 2
assert find_circle_num([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 3

dsu = DSU(4)
assert dsu.union(0, 1) and dsu.union(1, 2)
assert not dsu.union(0, 2)            # 0 and 2 already connected: cycle edge


def accounts_merge(accounts):         # G-721
    dsu = DSU(len(accounts))
    owner = {}                        # email -> first account index that listed it
    for i, (_, *emails) in enumerate(accounts):
        for e in emails:
            if e in owner:
                dsu.union(i, owner[e])
            else:
                owner[e] = i
    groups = defaultdict(list)
    for e, i in owner.items():
        groups[dsu.find(i)].append(e)
    return [[accounts[root][0]] + sorted(es) for root, es in groups.items()]


accounts = [["John", "johnsmith@mail.com", "john_newyork@mail.com"],
            ["John", "johnsmith@mail.com", "john00@mail.com"],
            ["Mary", "mary@mail.com"],
            ["John", "johnnybravo@mail.com"]]
assert sorted(accounts_merge(accounts)) == [
    ["John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"],
    ["John", "johnnybravo@mail.com"],
    ["Mary", "mary@mail.com"]]


def num_islands2(m, n, positions):    # G-305: land is added one cell at a time
    dsu = DSU(m * n)                  # cell (r, c) -> r * n + c
    land = set()
    islands, out = 0, []
    for r, c in positions:
        if (r, c) not in land:
            land.add((r, c))
            islands += 1
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (nr, nc) in land and dsu.union(r * n + c, nr * n + nc):
                    islands -= 1
        out.append(islands)
    return out


assert num_islands2(3, 3, [[0, 0], [0, 1], [1, 2], [2, 1]]) == [1, 1, 2, 3]
assert num_islands2(1, 3, [[0, 0], [0, 2], [0, 1], [0, 1]]) == [1, 2, 1, 1]
```

### Weighted union-find: Evaluate Division

- Store `ratio[x] = x / parent[x]`. After `find(x)`, `ratio[x] = x / root`.
- Same root: `a / b = ratio[a] / ratio[b]`. Unknown variable or different roots: `-1.0`.
- Merging roots: `ra / rb = v * ratio[b] / ratio[a]` for an equation `a / b = v`.

```python
import math


def calc_equation(equations, values, queries):        # G-399
    parent, ratio = {}, {}

    def find(x):                      # recursion depth <= number of variables
        p = parent[x]
        if p != x:
            root = find(p)            # afterwards ratio[p] = p / root
            ratio[x] *= ratio[p]      # x/p * p/root = x/root
            parent[x] = root
        return parent[x]

    for (a, b), v in zip(equations, values):          # a / b = v
        for x in (a, b):
            if x not in parent:
                parent[x], ratio[x] = x, 1.0
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            ratio[ra] = v * ratio[b] / ratio[a]

    out = []
    for a, b in queries:
        if a not in parent or b not in parent or find(a) != find(b):
            out.append(-1.0)
        else:
            out.append(ratio[a] / ratio[b])            # (a/root) / (b/root)
    return out


def close(xs, ys):
    return len(xs) == len(ys) and all(math.isclose(x, y) for x, y in zip(xs, ys))


assert close(calc_equation([["a", "b"], ["b", "c"]], [2.0, 3.0],
                           [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]),
             [6.0, 0.5, -1.0, 1.0, -1.0])
assert close(calc_equation([["a", "b"], ["b", "c"], ["bc", "cd"]], [1.5, 2.5, 5.0],
                           [["a", "c"], ["c", "b"], ["bc", "cd"], ["cd", "bc"]]),
             [3.75, 0.4, 5.0, 0.2])
# a/b = 2, c/d = 3, then b/c = 4 joins two existing groups: a/d = 2 * 4 * 3
assert close(calc_equation([["a", "b"], ["c", "d"], ["b", "c"]], [2.0, 3.0, 4.0],
                           [["a", "d"], ["d", "a"], ["c", "a"]]),
             [24.0, 1 / 24, 0.125])
```

### Tarjan bridges: Critical Connections

- `disc[u]`: DFS visit time. `low[u]`: smallest `disc` reachable from u's subtree with one back edge.
- Tree edge `u -> v` is a **bridge** iff `low[v] > disc[u]`: nothing under v climbs back to u or above.
- Skip the edge you came in on by **edge id**, not by parent node. Parallel edges then stay correct.
- Recursion depth = longest DFS path, up to `n` (10^5 in G-1192). Default limit is 1000.

```python
import random
import sys


def critical_connections(n, connections):            # G-1192, recursive
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(connections):
        adj[u].append((v, i))
        adj[v].append((u, i))
    disc = [-1] * n
    low = [0] * n
    timer = 0
    bridges = []

    def dfs(u, parent_edge):
        nonlocal timer                # timer += 1 rebinds an outer name
        disc[u] = low[u] = timer
        timer += 1
        for v, i in adj[u]:
            if i == parent_edge:
                continue
            if disc[v] == -1:         # tree edge
                dfs(v, i)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append([u, v])
            else:                     # back edge
                low[u] = min(low[u], disc[v])

    for s in range(n):
        if disc[s] == -1:
            dfs(s, -1)
    return bridges


def critical_connections_iter(n, connections):       # same result, explicit stack
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(connections):
        adj[u].append((v, i))
        adj[v].append((u, i))
    disc = [-1] * n
    low = [0] * n
    timer = 0
    bridges = []
    for s in range(n):
        if disc[s] != -1:
            continue
        disc[s] = low[s] = timer
        timer += 1
        stack = [(s, -1, iter(adj[s]))]               # (node, parent edge, edges left)
        while stack:
            u, pe, it = stack[-1]
            for v, i in it:
                if i == pe:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, i, iter(adj[v])))
                    break                             # descend; resume u's iterator later
                low[u] = min(low[u], disc[v])
            else:                                     # runs only if the loop did not break
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        bridges.append([p, u])
    return bridges


def norm(bridges):
    return sorted(tuple(sorted(e)) for e in bridges)


def connected_pairs(n, edges):
    dsu = list(range(n))

    def find(x):
        while dsu[x] != x:
            x = dsu[x]
        return x

    for u, v in edges:
        dsu[find(u)] = find(v)
    return [find(x) for x in range(n)]


def bridges_brute(n, edges):          # an edge is a bridge if removing it splits its ends
    out = []
    for k, (u, v) in enumerate(edges):
        comp = connected_pairs(n, edges[:k] + edges[k + 1:])
        if comp[u] != comp[v]:
            out.append([u, v])
    return out


assert critical_connections(4, [[0, 1], [1, 2], [2, 0], [1, 3]]) == [[1, 3]]
assert critical_connections(2, [[0, 1]]) == [[0, 1]]
assert critical_connections(2, [[0, 1], [0, 1]]) == []   # parallel edges: no bridge

random.seed(1192)
for _ in range(300):
    n = random.randint(1, 8)
    edges = [[random.randrange(n), random.randrange(n)] for _ in range(random.randint(0, 10))]
    edges = [e for e in edges if e[0] != e[1]]
    expected = norm(bridges_brute(n, edges))
    assert norm(critical_connections(n, edges)) == expected
    assert norm(critical_connections_iter(n, edges)) == expected

chain = [[i, i + 1] for i in range(4999)]             # a path: DFS depth 5000
assert sys.getrecursionlimit() == 1000
try:
    critical_connections(5000, chain)
except RecursionError:
    too_deep = True
else:
    too_deep = False
assert too_deep
n = 100_000
assert len(critical_connections_iter(n, [[i, i + 1] for i in range(n - 1)])) == n - 1
```

- Recursive fix: `sys.setrecursionlimit(1 << 18)` works for plain recursion on CPython 3.11+.
- Say it out loud in the interview, and offer the iterative version as the safe one.

### Eulerian path: Reconstruct Itinerary (Hierholzer)

- Use every edge exactly once: walk until stuck, then add the airport to the route on the way back.
- Greedy "always take the smallest next airport" fails: it can strand unused tickets.

```python
from collections import defaultdict


def find_itinerary(tickets):          # G-332
    out = defaultdict(list)
    for a, b in sorted(tickets, reverse=True):
        out[a].append(b)              # reverse order: pop() returns the smallest
    route, stack = [], ["JFK"]
    while stack:
        while out[stack[-1]]:
            stack.append(out[stack[-1]].pop())
        route.append(stack.pop())     # stuck: this airport is final at this position
    return route[::-1]


assert find_itinerary([["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]) == \
    ["JFK", "MUC", "LHR", "SFO", "SJC"]
assert find_itinerary([["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], ["ATL", "JFK"],
                       ["ATL", "SFO"]]) == ["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"]
assert find_itinerary([["JFK", "KUL"], ["JFK", "NRT"], ["NRT", "JFK"]]) == \
    ["JFK", "NRT", "JFK", "KUL"]      # KUL first would leave NRT unused
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `vector<vector<int>> adj(n)` | `adj = [[] for _ in range(n)]` | O(N) |
| `unordered_map<string, vector<string>> adj` | `adj = defaultdict(list)` | average O(1) per append |
| `adj.count(u)` (no insert) | `u in adj`, `adj.get(u, [])` | average O(1) |
| `vector<bool> seen(n)` / `unordered_set` | `[False] * n` / `set()` | O(1) / average O(1) |
| `vector<vector<bool>> seen(r, vector<bool>(c))` | `[[False] * c for _ in range(r)]` | O(R * C) |
| `int dr[4], dc[4]` | `DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))` | O(1) |
| `queue<int>`: `front()`, `pop()` | `deque`: `q[0]`, `q.popleft()` | O(1) |
| `stack<int>` for DFS | `list`: `append`, `pop` | O(1) |
| `function<void(int)> dfs = [&](int u) {...}` | nested `def dfs(u)`; `nonlocal` to rebind | same as C++ |
| `priority_queue<P, vector<P>, greater<P>>` | `heapq` with `(dist, node)` tuples | O(log N) |
| `vector<int> dist(n, INT_MAX)` | `dist = [math.inf] * n` | O(N) |
| `prev = dist;` (copies) | `prev = dist[:]` (`prev = dist` aliases) | O(N) |
| `iota(parent.begin(), parent.end(), 0)` | `parent = list(range(n))` | O(N) |
| `pair<int, int>` in a `set` | tuple `(r, c)` in a `set` | average O(1) |
| `multiset<string>` adjacency for itinerary | reverse-sorted `list` + `pop()` | O(E log E) total |

## Complexity

| Algorithm | Time | Space |
|---|---|---|
| Build adjacency list | O(V + E) | O(V + E) |
| DFS / BFS (graph) | O(V + E) | O(V) |
| DFS / BFS (R x C grid) | O(R * C) | O(R * C) |
| Kahn / DFS 3-colour topological sort | O(V + E) | O(V) |
| Dijkstra with `heapq` + stale skip | O((V + E) log V) | O(V + E) heap |
| Bellman-Ford limited to K + 1 rounds | O((K + 1) * E) | O(V) |
| Full Bellman-Ford | O(V * E) | O(V) |
| Union-find: compression + union by size | O(α(N)) amortized per op | O(N) |
| Union-find: path compression only | O(log N) amortized per op | O(N) |
| Weighted union-find (G-399) | O((E + Q) log V) amortized | O(V) |
| Tarjan bridges | O(V + E) | O(V + E) |
| Hierholzer (G-332) | O(E log E) for the sort | O(E) |

## Gotchas for C++ developers

- `adj[v]` on a `defaultdict` **inserts** v. While iterating the dict this raises **RuntimeError**.
- `[[]] * n` and `[[False] * c] * r` repeat **one** inner list. Use a comprehension.
- Recursion limit is **1000**. A 300 x 300 grid DFS can need 90,000 frames: use a stack or BFS.
- `sys.setrecursionlimit` helps plain recursion on 3.11+. Recursion through C (`@cache`, `sum(gen)`) still fails: note `10`.
- Mark nodes `seen` when **pushed**, not when popped, or the queue fills with duplicates.
- `while u:` stops at node `0`. Use `while u is not None`.
- `heapq` is a **min-heap**: `(dist, node)` pops the smallest distance. No `greater<>` needed.
- No decrease-key: push again and skip **stale** pops (`d > dist[u]`).
- `math.inf` is a float. `int(math.inf)` raises **OverflowError**: return `-1` before converting.
- No `INT_MAX + w` overflow in Python, but `dist[u] + w` on a C++ `INT_MAX` is undefined behaviour.
- `prev = dist` is an **alias**, not a copy. K-stop Bellman-Ford then silently chains edges.
- Directed cycle detection needs **3 colours**. A seen-set flags DAG cross edges as cycles.
- Kahn: count every node, including ones with no edges (all letters in G-269 Alien Dictionary).
- Assigning to an outer counter (`timer += 1`) in a nested `def` needs `nonlocal`.
- `for ... else`: the `else` runs only when the loop ends **without** `break`.
- Evaluate Division returns floats: compare with `math.isclose`, never `==`.

## In the Google bank

- **Flood fill / components:** G-200 Number of Islands, G-547 Number of Provinces, G-133 Clone Graph.
- **Flood fill variants:** G-417 Pacific Atlantic Water Flow, G-827 Making A Large Island.
- **BFS shortest path:** G-994 Rotting Oranges, G-127 Word Ladder, G-126 Word Ladder II, G-815 Bus Routes.
- **BFS with extra state:** G-2045 Second Minimum Time to Reach Destination, G-2812 Find the Safest Path in a Grid.
- **Topological sort:** G-207 Course Schedule, G-210 Course Schedule II, G-269 Alien Dictionary.
- **Topological variants:** G-444 Sequence Reconstruction, G-310 Minimum Height Trees (leaf peeling).
- **DAG longest path (DFS + memo):** G-329 Longest Increasing Path in a Matrix.
- **Dijkstra:** G-743 Network Delay Time, G-1631 Path With Minimum Effort, G-778 Swim in Rising Water.
- **Bellman-Ford rounds:** G-787 Cheapest Flights Within K Stops.
- **Union-find:** G-547 Number of Provinces, G-721 Accounts Merge, G-305 Number of Islands II.
- **Weighted union-find:** G-399 Evaluate Division.
- **Bridges:** G-1192 Critical Connections in a Network.
- **Eulerian path:** G-332 Reconstruct Itinerary.
