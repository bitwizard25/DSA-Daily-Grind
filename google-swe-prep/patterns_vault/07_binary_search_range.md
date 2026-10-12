# Binary Search (index and answer space)

**Cue:** sorted / rotated data, or "minimum X such that condition holds".

```
lo, hi = 0, n            # half-open [lo, hi)
while lo < hi:
    mid = (lo + hi) // 2
    if predicate(mid): hi = mid
    else: lo = mid + 1
return lo                # first index where predicate is true
```

- Answer-space search: Koko Eating Bananas, capacity to ship packages.
- Rotated arrays: decide which half is sorted first.
- Edge cases: empty array, target beyond both ends, overflow in (lo + hi).
