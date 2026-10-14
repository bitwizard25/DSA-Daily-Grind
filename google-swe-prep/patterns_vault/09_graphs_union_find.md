# Graphs: BFS, DFS, topological sort, Dijkstra and union-find

> In one line: first list each node's neighbours, then pick the tool: BFS for fewest steps, topological sort for an order, Dijkstra for cheapest cost, union-find for groups.

## When to use it
- Things joined by links: cities and roads, people and friends, courses and prerequisites.
- "Connected", "groups", "islands", "provinces": count the groups with DFS, BFS or union-find.
- "Order", "prerequisite", "depends on", "schedule": topological sort.
- "Cheapest", "least time", when each road has its own cost: Dijkstra.
- "Are a and b in the same group?" while links are added one by one: union-find.

## The idea in plain words
- A **graph** is a set of **nodes** (points) joined by **edges** (lines). A node's **neighbours** are the nodes one edge away.
- **Directed** edge: one way only, like a one-way street (`a → b`). **Undirected** edge: both ways.
- **Weighted** edge: it has a cost (its **weight**), like minutes of driving.
- A **cycle** is a path that comes back to where it started.
- An **adjacency list** stores, for each node, the list of its neighbours: `graph[0] = [1, 2]`.

**Step 1 is always: build the graph.** Turn the given **edge list** (like `[[0, 1], [1, 2]]`) into an adjacency list. For an undirected graph, add each edge both ways.

**DFS and BFS** visit every node you can reach from a start. A **visited** set makes sure no node is handled twice.
- **DFS** (depth-first search) goes deep along one path, then comes back and tries the next. Write it as a **recursive** function (one that calls itself), or with a stack.
- **BFS** (breadth-first search) visits in rings: 1 step away, then 2 steps, and so on. It uses a queue and finds the fewest steps (note 03).

**Topological sort** puts tasks in an order where each task comes after the tasks it depends on.
It works only when the directed graph has no cycle. **Kahn's algorithm** does it with a queue:
1. Count each node's **in-degree**: how many edges point into it (how many prerequisites are still open).
2. Put every node with in-degree 0 in a queue. Those can start now.
3. Take a node out and add it to the order. Each of its neighbours loses 1 in-degree. At 0, it joins the queue.
4. If the order is shorter than the number of nodes, some nodes wait forever: there is a cycle.

**Dijkstra** finds the cheapest cost from a start when edges have different costs. Costs must not be negative.
- `dist[x]` is the best cost found so far to reach node x. At first it is infinity, except `dist[start] = 0`.
- A min-heap (note 04) holds `(cost, node)` pairs and always gives back the cheapest one.
- Pop the cheapest pair `(cost, node)`. For each edge from that node, if `cost + weight` is less than the neighbour's `dist`, update `dist` and push the new pair.
- The heap can still hold an old pair for a node. You know it is old when its cost is bigger than `dist[node]`: skip it.

**Union-find** (also **DSU**, disjoint set union) keeps nodes in groups and answers "same group?" fast.
- `parent[x]` points to another node of x's group. Follow the pointers up to the **root**: the group's leader, which points to itself.
- `find(x)` returns the root. Two nodes are in the same group when they have the same root.
- `union(a, b)` joins two groups: the root of one now points to the root of the other.
- **Path compression**: after `find(x)`, point x straight at the root. Next time the walk is one step.
- **Union by size**: hang the smaller group under the bigger one, so the paths stay short.
- In an undirected graph, if a and b already have the same root, the edge `a - b` closes a cycle.

## Python template

### Build an adjacency list

```python
from collections import defaultdict

edges = [[0, 1], [0, 2], [1, 3]]
# defaultdict(list): a missing key starts as an empty list
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)  # undirected: add the edge both ways
print(graph[0])  # [1, 2]
print(graph[3])  # [1]

# nodes are numbered 0 to n-1: a list of lists works too
n = 4
adj = [[] for _ in range(n)]  # _ is a loop variable we do not use
for u, v in edges:
    adj[u].append(v)  # directed: one way only
print(adj)  # [[1, 2], [3], [], []]
```

### DFS and BFS: the same graph, two visit orders

```python
from collections import deque

graph = {0: [1, 2], 1: [3], 2: [4], 3: [], 4: []}

def dfs(u, visited, order):
    visited.add(u)
    order.append(u)
    for v in graph[u]:
        if v not in visited:
            dfs(v, visited, order)

order = []
dfs(0, set(), order)
# deep first: 0, then 1 and all below it, then 2
print(order)  # [0, 1, 3, 2, 4]

def bfs(start):
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return order

# rings: 0, then 1 step away (1, 2), then 2 steps away (3, 4)
print(bfs(0))  # [0, 1, 2, 3, 4]
```

### Topological sort: Course Schedule (Kahn's algorithm)

```python
from collections import deque

def course_order(n, prereqs):
    # [a, b] means: take course b before course a, so the edge is b -> a
    graph = [[] for _ in range(n)]
    indegree = [0] * n
    for a, b in prereqs:
        graph[b].append(a)
        indegree[a] += 1
    queue = deque()
    for c in range(n):
        if indegree[c] == 0:
            queue.append(c)
    order = []
    while queue:
        c = queue.popleft()
        order.append(c)
        for nxt in graph[c]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) < n:
        return []  # a cycle: some courses can never start
    return order

print(course_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))  # [0, 1, 2, 3]
print(course_order(2, [[1, 0], [0, 1]]))  # []
```

### Dijkstra with heapq: the cheapest cost from a start

```python
import heapq

def cheapest(n, edges, start):
    # [u, v, w]: a one-way road from u to v that costs w
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    dist = [float("inf")] * n  # infinity: not reached yet
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # an old, more expensive pair: skip it
        for v, w in graph[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return dist

roads = [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1]]
# 0 -> 2 -> 1 costs 1 + 2 = 3, cheaper than the direct road 0 -> 1 (4); no road reaches 4
print(cheapest(5, roads, 0))  # [0, 3, 1, 4, inf]
```

### Union-find with path compression and union by size

```python
parent = list(range(6))  # at first, each node is its own group
size = [1] * 6

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # path compression: point x at the root
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)  # the roots of a and b
    if ra == rb:
        return False  # already one group: this edge closes a cycle
    if size[ra] < size[rb]:
        ra, rb = rb, ra  # swap, so ra is the bigger group
    parent[rb] = ra
    size[ra] += size[rb]
    return True

groups = 6
for a, b in [[0, 1], [1, 2], [3, 4]]:
    if union(a, b):
        groups -= 1
# the groups are {0, 1, 2}, {3, 4} and {5}
print(groups)  # 3
print(find(2) == find(0))  # True
print(find(3) == find(0))  # False
print(union(0, 2))  # False
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `vector<vector<int>> adj(n);` | `adj = [[] for _ in range(n)]` | a new empty list for each node |
| `unordered_map<int, vector<int>> g;` | `g = defaultdict(list)` | a missing key starts as `[]` |
| `vector<bool> seen(n, false);` | `seen = [False] * n` or `seen = set()` | a set works for any node names |
| `stack<int> st; st.push(u);` | `stack = [u]` | `stack.pop()` takes the last item |
| `priority_queue<P, vector<P>, greater<P>> pq;` | `heap = []` plus `heapq` | a min-heap, like `greater<P>` |
| `vector<int> dist(n, INT_MAX);` | `dist = [float("inf")] * n` | infinity: bigger than any number, and `inf + 5` is `inf` |
| `iota(parent.begin(), parent.end(), 0);` | `parent = list(range(n))` | each node starts as its own root |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| Build an adjacency list | O(V + E) | V = number of nodes, E = number of edges; each is handled once |
| DFS or BFS | O(V + E) | each node is visited once; each edge is looked at once (twice if undirected) |
| Topological sort (Kahn) | O(V + E) | each node enters the queue at most once |
| Dijkstra with heapq | O((V + E) log V) | every heap push or pop costs about log V steps |
| Union-find with both tricks | almost O(1) per call, on average | the trees stay very flat, so `find` takes very few steps |
| Union-find with no tricks | O(V) per call | the parent pointers can form one long chain |

## Common mistakes
- Mistake: adding an undirected edge only one way. Fix: `graph[u].append(v)` and also `graph[v].append(u)`.
- Mistake: no visited set. The walk goes round a cycle forever. Fix: mark a node the first time you reach it.
- Mistake: recursive DFS on a huge graph or grid. Python stops very deep recursion with `RecursionError`. Fix: use BFS, or DFS with a list as a stack (`append` and `pop`).
- Mistake: Kahn's algorithm without the last check. Fix: if `len(order) < n`, there is a cycle and no valid order.
- Mistake: plain BFS when edges cost different amounts. BFS counts steps, not cost. Fix: Dijkstra (costs 0 or more).
- Mistake: no skip of old pairs in Dijkstra. The answer is still right, but it does useless work. Fix: `if d > dist[u]: continue`.
- Mistake: testing "same group" with `parent[a] == parent[b]`. Parents are not always roots. Fix: `find(a) == find(b)`.

## Practice in the app
- **G-200 Number of Islands**: each land cell is a node; start a DFS or BFS from each unvisited land cell.
- **G-547 Number of Provinces**: count the groups with union-find, or with DFS.
- **G-207 Course Schedule**: the Kahn template; the answer is `len(order) == n`.
- **G-210 Course Schedule II**: the Kahn template; return the order itself.
- **G-269 Alien Dictionary**: compare two neighbouring words; their first different letters give one edge. Then topological sort.
- **G-133 Clone Graph**: DFS or BFS with a dict from each old node to its copy.
- **G-743 Network Delay Time**: Dijkstra from k; the answer is the largest `dist`, or -1 if one stays infinity.
- **G-1631 Path With Minimum Effort**: Dijkstra on a grid; a path's cost is its biggest single step, not the sum.
- **G-721 Accounts Merge**: union-find; join all the emails of one account, then list each group.
- **G-305 Number of Islands II**: union-find; add land one cell at a time and join it with land neighbours.
