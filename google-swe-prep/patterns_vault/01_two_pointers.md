# Two Pointers

**Cue:** sorted input, pairs/triplets, palindromes, in-place partition.

```
l, r = 0, n - 1
while l < r:
    s = a[l] + a[r]
    if s == target: record; l += 1; r -= 1
    elif s < target: l += 1
    else: r -= 1
```

- Converging: works because each move discards a candidate that cannot be optimal.
- Fast/slow: cycle detection, middle of a list.
- Edge cases: duplicates (skip them), `l <= r` vs `l < r`.
- Problems: 3Sum, Container With Most Water, Trapping Rain Water.
