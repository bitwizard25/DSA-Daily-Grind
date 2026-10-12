# Heaps / Priority Queues

**Cue:** top-K, K-th largest, merge K sorted, running median, scheduling.

```
heap = []
for x in data:
    heapq.heappush(heap, x)
    if len(heap) > k: heapq.heappop(heap)
kth_largest = heap[0]
```

- Python heapq is a min-heap; negate values for a max-heap.
- Two heaps (max-heap low half, min-heap high half) give a running median.
- Push O(log N), pop O(log N), peek O(1), heapify O(N).
- Problems: Merge k Sorted Lists, Top K Frequent, Find Median from Data Stream.
