# Sliding Window

**Cue:** contiguous subarray/substring + "longest / shortest / at most K".

```
left = 0
for right in range(n):
    add(a[right])
    while window_invalid():
        remove(a[left]); left += 1
    best = max(best, right - left + 1)
```

- "Exactly K" = atMost(K) - atMost(K - 1).
- Negative numbers break the shrink condition: use prefix sums instead.
- Fixed size max/min: monotonic deque.
- Problems: Longest Substring Without Repeating Characters, Minimum Window Substring, Sliding Window Maximum.
