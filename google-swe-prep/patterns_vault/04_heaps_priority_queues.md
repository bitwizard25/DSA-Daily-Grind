# Heaps and priority queues: top K and meeting rooms

> In one line: a heap always gives you the smallest item fast. Keep K items in it for "top K", or one end time per room for meeting rooms.

## When to use it
- "k largest", "k smallest", "k most frequent", "k closest" or "the k-th largest".
- "Merge k sorted lists".
- "Which task runs next?": you always need the most urgent item.
- "How many rooms, cars or servers are busy at the same time?" (time **intervals**).
- "The middle value (median) of a stream of numbers".

## The idea in plain words
A **priority queue** gives back the most important item first, not the oldest one.
A **heap** is the usual way to build it. Python's `heapq` keeps a plain list in **heap order**.
Heap order means only one thing for sure: the smallest item is at index 0. The rest is not sorted.
- `heapq.heappush(h, x)` adds an item. `heapq.heappop(h)` removes and returns the smallest.
- Both are O(log N). log N is how many times you can halve N: about 20 for a million items.
- `h[0]` reads the smallest without removing it: O(1).

`heapq` is a **min-heap**: smallest first. C++ `priority_queue` is a **max-heap**: biggest first.
For biggest first in Python, push `-x`, and flip the sign back when you read it.

A **tuple** like `(priority, item)` lets you attach a priority to any item.
Python compares tuples field by field: the first field first, and the second only on a tie.

**Top K largest**: keep a min-heap that holds the K largest numbers seen so far.
1. Push each new number.
2. If the heap now has K + 1 items, pop the smallest. It cannot be in the top K.
3. At the end, `h[0]` is the K-th largest, and the heap holds the whole top K.
The heap never grows past K + 1, so each step costs about log K, not log N.

**Sweep line** (meeting rooms): an **interval** is a pair `[start, end]`, like a meeting from 9 to 10.
A **sweep line** means: sort by time and walk forward, like the hand of a clock.
1. Sort the meetings by start time.
2. Keep a min-heap of end times: one end time for each room in use.
3. For each meeting: if the room that frees up first is free by now (`ends[0] <= start`), pop it to reuse it.
4. Push this meeting's end time.
5. At the end, the heap size is the number of rooms you need.
Each meeting pushes one end time and pops at most one, so the heap never gets smaller.
So its size at the end is the most rooms in use at one time.

## Python template

### heapq basics

```python
import heapq

h = []
for x in [5, 1, 4]:
    heapq.heappush(h, x)
print(h[0])  # 1
print(heapq.heappop(h))  # 1
print(h[0])  # 4

# biggest first: store -x, flip the sign back when you read
big = []
for x in [5, 1, 4]:
    heapq.heappush(big, -x)
print(-heapq.heappop(big))  # 5

# (priority, item): the smaller priority comes out first
tasks = []
heapq.heappush(tasks, (2, "write tests"))
heapq.heappush(tasks, (1, "fix bug"))
print(heapq.heappop(tasks))  # (1, 'fix bug')

# turn a whole list into a heap, in place
nums = [5, 2, 8, 1]
heapq.heapify(nums)
print(nums[0])  # 1
```

### Top K: a heap of size K

```python
import heapq
from collections import Counter


def kth_largest(nums, k):
    # h keeps the k largest numbers seen so far; h[0] is the smallest of them
    h = []
    for x in nums:
        heapq.heappush(h, x)
        if len(h) > k:
            heapq.heappop(h)
    return h[0]


def top_k_frequent(nums, k):
    # Counter counts how many times each value appears (note 05)
    h = []
    for value, count in Counter(nums).items():
        heapq.heappush(h, (count, value))
        if len(h) > k:
            heapq.heappop(h)
    result = []
    for count, value in sorted(h, reverse=True):
        result.append(value)
    return result


print(kth_largest([3, 2, 1, 5, 6, 4], 2))  # 5
print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))  # [1, 2]
```

### Sweep line: meeting rooms with a heap of end times

```python
import heapq


def min_meeting_rooms(meetings):
    # one end time per room in use; the earliest end is ends[0]
    ends = []
    for start, end in sorted(meetings):
        if ends and ends[0] <= start:
            # the room that frees up first is free now: reuse it
            heapq.heappop(ends)
        heapq.heappush(ends, end)
    return len(ends)


print(min_meeting_rooms([[0, 30], [5, 10], [15, 20]]))  # 2
print(min_meeting_rooms([[1, 5], [5, 10]]))  # 1
```

### Sweep line: count events (no heap)

Each meeting is two **events**: +1 room at its start, -1 room at its end. Sort the events and keep a running total.

```python
def rooms_by_events(meetings):
    events = []
    for start, end in meetings:
        events.append((start, 1))
        events.append((end, -1))
    # at the same time, (t, -1) sorts before (t, 1): a room is freed before it is reused
    events.sort()
    in_use = 0
    most = 0
    for time, change in events:
        in_use += change
        most = max(most, in_use)
    return most


print(rooms_by_events([[0, 30], [5, 10], [15, 20]]))  # 2
print(rooms_by_events([[1, 5], [5, 10]]))  # 1
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `priority_queue<int, vector<int>, greater<int>> pq;` | `h = []` plus `heapq` functions | heapq is a min-heap, like `greater<int>` |
| `priority_queue<int> pq;` (max-heap) | `heapq.heappush(h, -x)` | flip the sign back when you read |
| `pq.push(x);` | `heapq.heappush(h, x)` | O(log N) |
| `pq.top()` | `h[0]` | the smallest item |
| `int x = pq.top(); pq.pop();` | `x = heapq.heappop(h)` | `heappop` returns the item |
| `pq.empty()` / `pq.size()` | `not h` / `len(h)` | an empty list counts as False |
| `pq.push({d, node});` | `heapq.heappush(h, (d, node))` | smallest `d` comes out first |
| `sort(v.begin(), v.end());` on pairs | `v.sort()` | sorts by the first field, then the second |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| `heappush`, `heappop` | O(log N) | about 20 steps for a million items |
| `h[0]` | O(1) | the smallest item is always at index 0 |
| `heapify(nums)` | O(N) | faster than N separate pushes |
| Top K with a heap of size K | O(N log K) | N items, about log K steps each |
| Sort everything, then take K | O(N log N) | also fine; a bit slower when K is small |
| Meeting rooms (sort + heap, or events) | O(N log N) | the sort is the slow part |
| `x in h` | O(N) | a heap cannot search fast |

## Common mistakes
- Mistake: expecting the biggest item first, as in C++. Fix: push `-x`, and read it back as `-heapq.heappop(h)`.
- Mistake: `h.pop()` on a heap. It removes the last list item, not the smallest. Fix: always use `heapq.heappop(h)`.
- Mistake: thinking the heap list is sorted. Only `h[0]` is certain. Fix: pop the items one by one, or use `sorted(h)`.
- Mistake: `h = heapq.heapify(nums)`. `heapify` returns None. Fix: call `heapq.heapify(nums)` alone, then use `nums`.
- Mistake: pushing `(priority, node)` with list nodes. On equal priorities Python compares the nodes and stops with a TypeError. Fix: push `(priority, i, node)` with a unique number `i`.
- Mistake: forgetting to sort the meetings by start time. Fix: `for start, end in sorted(meetings):`.
- Mistake: `ends[0] < start` in meeting rooms. A meeting that ends at 5 frees its room for one that starts at 5. Fix: `ends[0] <= start`.

## Practice in the app
- **G-215 Kth Largest Element in an Array**: a min-heap of size k; the answer is `h[0]`.
- **G-347 Top K Frequent Elements**: count with `Counter`, then keep a heap of `(count, value)` of size k.
- **G-23 Merge k Sorted Lists**: a heap of `(value, i, node)` for the first node of each list; pop the smallest, push its next.
- **G-295 Find Median from Data Stream**: two heaps: biggest-first for the lower half, smallest-first for the upper half.
- **G-253 Meeting Rooms II**: the meeting rooms template above.
- **G-1094 Car Pooling**: events: `+people` at pickup and `-people` at drop-off; the running total must stay within capacity.
- **G-2402 Meeting Rooms III**: two heaps: free room numbers, and `(end time, room)` for busy rooms.
- **G-1834 Single-Threaded CPU**: sort tasks by start time; among the tasks that have arrived, a heap of `(processing time, index)` picks the next one.
- **G-621 Task Scheduler**: always run the ready task with the most copies left (a biggest-first heap of counts).
- **G-759 Employee Free Time**: put all work intervals in time order (sort or heap), then look for the gaps.
- **G-743 Network Delay Time**: Dijkstra: a min-heap of `(distance, node)`; the full template is in note 09.
