# Heaps and Priority Queues

**Use when:**
- "k largest", "k smallest", "k closest", "k most frequent", "k-th largest".
- "merge k sorted lists / arrays / streams".
- "median of a data stream", "running median".
- "minimum rooms / cars / servers at the same time", "which task runs next".

## Python templates

### heapq basics: min-heap on a plain list

```python
import heapq

h = []
for x in [5, 1, 4]:
    heapq.heappush(h, x)             # O(log N)
assert h[0] == 1                     # peek the min, O(1); like pq.top()
assert heapq.heappop(h) == 1         # pop RETURNS the item (C++ pop() is void)

maxh = []                            # max-heap: push -x, read -h[0]
for x in [3, 1, 4]:
    heapq.heappush(maxh, -x)
assert -maxh[0] == 4
assert -heapq.heappop(maxh) == 4

xs = [5, 4, 3, 2, 1]
heapq.heapify(xs)                    # in place, O(N); returns None
assert xs[0] == 1
assert xs != sorted(xs)              # only xs[0] is guaranteed; the rest is heap order

h = [2, 5]
assert heapq.heappushpop(h, 1) == 1 and h == [2, 5]   # push, then pop: 1 leaves at once
h = [2, 5]
assert heapq.heapreplace(h, 1) == 2 and h == [1, 5]   # pop, then push: returns the old min

nums = [5, 1, 9, 3, 7]
assert heapq.nlargest(2, nums) == [9, 7]              # sorted, largest first
assert heapq.nsmallest(2, nums) == [1, 3]
assert heapq.nlargest(1, ["bb", "a", "ccc"], key=len) == ["ccc"]
```

### Top-K: keep a size-K min-heap

```python
import heapq
from collections import Counter


def kth_largest(nums, k):            # G-215
    h = []                           # the k largest seen so far; h[0] is the smallest of them
    for x in nums:
        if len(h) < k:
            heapq.heappush(h, x)
        elif x > h[0]:
            heapq.heapreplace(h, x)  # drop the min, add x: one O(log k) step
    return h[0]


assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
assert kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4


def top_k_frequent(nums, k):         # G-347
    h = []
    for x, c in Counter(nums).items():
        heapq.heappush(h, (c, x))    # equal counts compare x next (ints: fine)
        if len(h) > k:
            heapq.heappop(h)         # drop the least frequent
    return [x for c, x in sorted(h, reverse=True)]


assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
assert top_k_frequent([4, 4, 5], 1) == [4]


def k_closest(points, k):
    h = []                           # max-heap by distance: store (-dist, index)
    for i, (x, y) in enumerate(points):
        d = x * x + y * y            # no sqrt: same order, exact ints
        if len(h) < k:
            heapq.heappush(h, (-d, i))
        elif -d > h[0][0]:           # closer than the farthest point kept
            heapq.heapreplace(h, (-d, i))
    return sorted(points[i] for _, i in h)


assert k_closest([[1, 3], [-2, 2]], 1) == [[-2, 2]]
assert k_closest([[3, 3], [5, -1], [-2, 4]], 2) == [[-2, 4], [3, 3]]
```

The same top-K in C++. Note the inverted comparator: `greater` gives a min-heap.

```cpp
#include <cassert>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

int main() {
    std::priority_queue<int> maxh;                    // default: MAX-heap
    for (int x : {3, 1, 4}) maxh.push(x);
    assert(maxh.top() == 4);

    std::priority_queue<int, std::vector<int>, std::greater<int>> minh;   // what heapq gives
    std::vector<int> nums = {3, 2, 1, 5, 6, 4};
    const int k = 2;
    for (int x : nums) {
        minh.push(x);
        if (static_cast<int>(minh.size()) > k) minh.pop();   // pop() returns void
    }
    assert(minh.top() == 5);

    using P = std::pair<int, int>;
    auto cmp = [](const P& a, const P& b) { return a.first > b.first; };   // ">" = min-heap
    std::priority_queue<P, std::vector<P>, decltype(cmp)> byFirst(cmp);
    byFirst.push({5, 0});
    byFirst.push({2, 1});
    assert(byFirst.top().first == 2);
    return 0;
}
```

### K-way merge: (val, idx, node) tie-breakers

```python
import heapq


class ListNode:                      # defines no "<": two nodes cannot be compared
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def from_list(vals):
    dummy = tail = ListNode()
    for v in vals:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


def merge_k_lists(lists):            # G-23
    # i is unique per list, so a tie on val is settled by i and never reaches node
    heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
    heapq.heapify(heap)              # O(k)
    dummy = tail = ListNode()
    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next


merged = merge_k_lists([from_list([1, 4, 5]), from_list([1, 3, 4]), None, from_list([2, 6])])
assert to_list(merged) == [1, 1, 2, 3, 4, 4, 5, 6]
assert merge_k_lists([]) is None

# Without the index, equal vals make heapq compare the nodes themselves.
h = []
heapq.heappush(h, (1, ListNode(1)))
try:
    heapq.heappush(h, (1, ListNode(1)))
except TypeError:                    # '<' not supported between 'ListNode' instances
    failed = True
else:
    failed = False
assert failed


def merge_k_arrays(arrays):
    heap = [(a[0], i, 0) for i, a in enumerate(arrays) if a]   # (value, array idx, elem idx)
    heapq.heapify(heap)
    out = []
    while heap:
        val, i, j = heapq.heappop(heap)
        out.append(val)
        if j + 1 < len(arrays[i]):
            heapq.heappush(heap, (arrays[i][j + 1], i, j + 1))
    return out


arrays = [[1, 4, 7], [2, 5], [], [3, 6, 9]]
assert merge_k_arrays(arrays) == [1, 2, 3, 4, 5, 6, 7, 9]
assert list(heapq.merge(*arrays)) == [1, 2, 3, 4, 5, 6, 7, 9]   # lazy library version
```

### Two heaps: running median

```python
import heapq
import random
import statistics


class MedianFinder:                  # G-295
    def __init__(self):
        self.lo = []                 # lower half, max-heap via negation
        self.hi = []                 # upper half, min-heap
        # invariants: max(lo) <= min(hi); len(lo) == len(hi) or len(hi) + 1

    def add_num(self, x):
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))   # move lo's max to hi
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]       # negate back when reading
        return (-self.lo[0] + self.hi[0]) / 2


mf = MedianFinder()
mf.add_num(1)
mf.add_num(2)
assert mf.find_median() == 1.5
mf.add_num(3)
assert mf.find_median() == 2

random.seed(295)
mf, seen = MedianFinder(), []
for _ in range(300):
    x = random.randint(-50, 50)
    mf.add_num(x)
    seen.append(x)
    assert mf.find_median() == statistics.median(seen)
```

### Sweep line: min-heap of end times

```python
import heapq
import random


def min_meeting_rooms(intervals):    # G-253, intervals are [start, end)
    ends = []                        # one end time per room in use
    for start, end in sorted(intervals):
        if ends and ends[0] <= start:            # the earliest-ending room is free
            heapq.heapreplace(ends, end)         # reuse it
        else:
            heapq.heappush(ends, end)            # open a new room
    return len(ends)


assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2
assert min_meeting_rooms([[7, 10], [2, 4]]) == 1
assert min_meeting_rooms([[1, 5], [5, 10]]) == 1     # touching meetings share a room
assert min_meeting_rooms([]) == 0


def min_meeting_rooms_events(intervals):         # same answer, no heap
    events = [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals]
    best = cur = 0
    for _, delta in sorted(events):  # (t, -1) sorts before (t, 1): free the room first
        cur += delta
        best = max(best, cur)
    return best


random.seed(253)
for _ in range(200):
    iv = []
    for _ in range(random.randint(0, 8)):
        s = random.randint(0, 20)
        iv.append([s, s + random.randint(1, 6)])
    peak = max((sum(s <= t < e for s, e in iv) for t in range(30)), default=0)
    assert min_meeting_rooms(iv) == min_meeting_rooms_events(iv) == peak


def car_pooling(trips, capacity):    # G-1094, trip = [passengers, from, to]
    riding = []                      # (drop-off point, passengers)
    load = 0
    for num, start, end in sorted(trips, key=lambda t: t[1]):
        while riding and riding[0][0] <= start:
            load -= heapq.heappop(riding)[1]     # drop off before picking up
        load += num
        if load > capacity:
            return False
        heapq.heappush(riding, (end, num))
    return True


assert not car_pooling([[2, 1, 5], [3, 3, 7]], 4)
assert car_pooling([[2, 1, 5], [3, 3, 7]], 5)
assert car_pooling([[2, 1, 5], [3, 5, 7]], 3)


def get_order(tasks):                # G-1834, task = [enqueue time, processing time]
    by_time = sorted(range(len(tasks)), key=lambda i: tasks[i][0])
    ready, out = [], []              # ready: (processing time, index); index breaks ties
    time = i = 0
    while len(out) < len(tasks):
        if not ready and time < tasks[by_time[i]][0]:
            time = tasks[by_time[i]][0]          # CPU idle: jump the clock
        while i < len(by_time) and tasks[by_time[i]][0] <= time:
            j = by_time[i]
            heapq.heappush(ready, (tasks[j][1], j))
            i += 1
        proc, j = heapq.heappop(ready)
        time += proc
        out.append(j)
    return out


assert get_order([[1, 2], [2, 4], [3, 2], [4, 1]]) == [0, 2, 3, 1]
assert get_order([[7, 10], [7, 12], [7, 5], [7, 4], [7, 2]]) == [4, 3, 2, 0, 1]
```

### Dijkstra

- **Dijkstra** is a min-heap of `(dist, node)`; on pop, skip entries where `d > dist[u]` (stale).
- Full template and grid variants: graphs note `09`.

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `priority_queue<int, vector<int>, greater<int>>` | `heapq` functions on a plain list | push / pop O(log N) |
| `priority_queue<int>` (max-heap) | `heappush(h, -x)`, read `-h[0]` | O(log N) |
| `pq.top()` | `h[0]` | O(1) |
| `pq.push(x)` | `heapq.heappush(h, x)` | O(log N) |
| `pq.pop()` (returns void) | `heapq.heappop(h)` (returns the item) | O(log N) |
| `pq.size()`, `pq.empty()` | `len(h)`, `not h` | O(1) |
| `make_heap` / range constructor | `heapq.heapify(lst)` (in place) | O(N) |
| `pop()` then `push(x)` | `heapq.heapreplace(h, x)` | O(log N) |
| `push(x)` then `pop()` | `heapq.heappushpop(h, x)` | O(log N) |
| custom comparator struct / lambda | tuple key `(priority, tie_breaker, item)` | O(log N) per op |
| `partial_sort` for top k | `heapq.nlargest(k, xs)` / `nsmallest` | O(N log k) |
| `nth_element` | size-k heap, or `sorted(xs)[k]` | O(N log k) / O(N log N) |
| `std::merge` (two ranges) | `heapq.merge(a, b)` (lazy, any count) | O(N log k) |
| `multiset` erase-any for a window | heap + lazy deletion (`Counter` of removed) | O(log N) amortized |

## Complexity

| Operation | Time | Space / note |
|---|---|---|
| `heappush`, `heappop` | O(log N) | |
| `h[0]` (peek) | O(1) | |
| `heapify(xs)` | O(N) | in place |
| `heapreplace`, `heappushpop` | O(log N) | one sift instead of two |
| `nlargest(k, xs)`, `nsmallest(k, xs)` | O(N log k) | O(k) |
| `heapq.merge` of k sorted inputs, N items total | O(N log k) | O(k) |
| `x in h`, `h.remove(x)` + `heapify(h)` | O(N) | no fast search or erase |
| Top-K with a size-K heap | O(N log K) | O(K) |
| K-way merge | O(N log k) | O(k) |
| Running median: add / median | O(log N) / O(1) | O(N) |
| Meeting rooms II, car pooling | O(N log N) | O(N) |
| Single-Threaded CPU | O(N log N) | O(N) |

## Gotchas for C++ developers

- `heapq` is a **min-heap**; `std::priority_queue` is a **max-heap**. Negate for max, negate back on read.
- Negating is safe in Python (ints never overflow). In C++ `-INT_MIN` overflows: undefined behaviour.
- Negation only works for numbers. For a string max-heap, use a wrapper class with a reversed `__lt__`.
- Python 3.14 adds `heapq.heappush_max` and friends. Older interview runtimes lack them: negate.
- The heap is a plain list in heap order, not sorted. Only `h[0]` has a fixed meaning (the min).
- `heapify` returns None: `h = heapq.heapify(xs)` loses the list.
- `heappop([])` raises **IndexError**. Check `if h` before popping or peeking.
- Tuples compare field by field. A tie falls through to the next field; an unorderable one raises **TypeError**.
- Fix: put a unique index before the object, as in `(val, i, node)`.
- No decrease-key and no erase. Push a new entry and skip stale ones on pop (**lazy deletion**).
- Changing `h[i]` in place breaks the heap invariant. Call `heapify(h)` again (O(N)).
- `heapreplace` returns the old min even when the new item is smaller. `heappushpop` returns the smaller of the two.
- Meeting rooms: decide if `[s, e)` touching intervals overlap. `ends[0] <= start` means they do not.

## In the Google bank

- **Top-K:** G-215 Kth Largest Element in an Array, G-347 Top K Frequent Elements.
- **Max-heap by count:** G-621 Task Scheduler.
- **K-way merge:** G-23 Merge k Sorted Lists, G-759 Employee Free Time.
- **Two heaps:** G-295 Find Median from Data Stream.
- **Sweep line with end times:** G-253 Meeting Rooms II, G-1094 Car Pooling, G-2402 Meeting Rooms III.
- **Scheduling simulation:** G-1834 Single-Threaded CPU.
- **Dijkstra (note 09):** G-743 Network Delay Time, G-1631 Path With Minimum Effort, G-778 Swim in Rising Water.
