# Topological Sort

**Cue:** dependencies, prerequisites, build order, "is it possible".

```
indeg = count incoming edges
q = deque(n for n in nodes if indeg[n] == 0)
order = []
while q:
    u = q.popleft(); order.append(u)
    for v in adj[u]:
        indeg[v] -= 1
        if indeg[v] == 0: q.append(v)
cycle = len(order) < len(nodes)
```

- Number of BFS layers = minimum semesters.
- Alien Dictionary: compare adjacent words; a longer word before its own prefix is invalid.
- Problems: Course Schedule I/II, Alien Dictionary.
