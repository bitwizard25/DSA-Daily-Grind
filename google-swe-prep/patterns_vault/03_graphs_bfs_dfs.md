# Graphs: BFS / DFS / Dijkstra

**Cue:** grid, network, "shortest", "connected", "spread".

```
q = deque(sources); seen = set(sources)
while q:
    u = q.popleft()
    for v in adj(u):
        if v not in seen:
            seen.add(v)          # mark on push, not on pop
            q.append(v)
```

- Unweighted shortest path: BFS. Weighted non-negative: Dijkstra. Negative edges / K-stop limits: Bellman-Ford.
- State can be more than a node: (row, col, k_remaining).
- Edge cases: disconnected parts, recursion depth on big grids.
- Problems: Number of Islands, Word Ladder, Network Delay Time, Shortest Path with Obstacles Elimination.
