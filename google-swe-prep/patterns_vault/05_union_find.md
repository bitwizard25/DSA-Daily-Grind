# Union-Find (Disjoint Set Union)

**Cue:** grouping, dynamic connectivity, "redundant edge", merging records.

```
parent = list(range(n)); size = [1] * n
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]   # path halving
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb: return False
    if size[ra] < size[rb]: ra, rb = rb, ra
    parent[rb] = ra; size[ra] += size[rb]
    return True
```

- Near O(1) amortised per operation.
- Weighted union-find solves Evaluate Division.
- Problems: Accounts Merge, Redundant Connection, Number of Islands II.
