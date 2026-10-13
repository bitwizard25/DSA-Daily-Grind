# Python data structures for C++ developers

> In one line: choose the tool by the question you ask: "last item?" list, "first item?" deque, "smallest item?" heapq, "value for this key?" dict, "seen it before?" set.

## When to use it
- The problem says "k largest", "smallest", "closest" or "next task to run": think **heap** (`heapq`).
- It says "count", "how many times" or "group by": think `Counter` or `defaultdict`.
- It says "visited", "unique" or "duplicate": think `set`.
- It gives you a `ListNode` or `TreeNode`: write a small class.
- A recursive function gets the same input again and again: add `@cache`.

## The idea in plain words
Python has a small toolbox, and almost every interview answer uses only these tools.
**Big-O** tells you how the time grows with N, the number of items.
**O(1)** means the same speed for 10 or 10 million items. **O(N)** means the work grows with N.
**O(log N)** grows very slowly: about 20 steps for a million items.

1. **list**: a growing array. Used as a **stack** (last in, first out): `append` adds on top, `pop()` takes the top.
2. **deque**: a **queue** (first in, first out). `append` at the back, `popleft()` at the front. Both O(1).
3. **heapq**: a **min-heap** (always gives the smallest item first). For the biggest first, store `-x`.
4. **dict**: a lookup table (finds the value for a key in O(1) on average), like `unordered_map`. `defaultdict` and `Counter` are dicts that give a start value for a new key.
5. **set**: each value once, O(1) "is it there?". Items must be **hashable** (immutable: numbers, strings, tuples).
6. **class**: your own object, like a C++ `struct`. `None` plays the role of `nullptr`.
7. **sorted / sort / bisect**: sort with `key=`, then search the sorted list with **binary search** (halve the range each step).
8. **@cache** (or `@lru_cache(None)`): **memoization** (remembering answers), so each input of a recursive function is worked out once.

## Python template

### list as a stack, deque as a queue

```python
from collections import deque

stack = [1, 2]
stack.append(3)
print(stack[-1])  # 3
print(stack.pop())  # 3
print(stack)  # [1, 2]

queue = deque([1, 2])
queue.append(3)
print(queue.popleft())  # 1
print(list(queue))  # [2, 3]
```

### heapq: smallest first, biggest first, (priority, item)

```python
import heapq

heap = []
for x in [5, 1, 4]:
    heapq.heappush(heap, x)
print(heap[0])  # 1
print(heapq.heappop(heap))  # 1

big = []
for x in [5, 1, 4]:
    heapq.heappush(big, -x)
print(-heapq.heappop(big))  # 5

tasks = []
heapq.heappush(tasks, (2, "email"))
heapq.heappush(tasks, (1, "fix bug"))
print(heapq.heappop(tasks))  # (1, 'fix bug')
```

### dict, defaultdict, Counter

```python
from collections import Counter, defaultdict

ages = {"amy": 25}
ages["bob"] = 30
print(ages.get("dan", 0))  # 0
print("amy" in ages)  # True

graph = defaultdict(list)
for u, v in [(1, 2), (1, 3)]:
    graph[u].append(v)
print(graph[1])  # [2, 3]

count = Counter("banana")
print(count["a"], count["z"])  # 3 0
print(count.most_common(1))  # [('a', 3)]
```

### set: seen before?

```python
seen = set()
seen.add((0, 1))
print((0, 1) in seen)  # True
print(sorted(set([4, 1, 4, 1])))  # [1, 4]
```

### Your own nodes

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


head = ListNode(1, ListNode(2))
print(head.next.val, head.next.next)  # 2 None
root = TreeNode(5, TreeNode(3), TreeNode(8))
print(root.left.val, root.right.val)  # 3 8

a, b = 1, 2
a, b = b, a
print(a, b)  # 2 1
```

### Sorting and bisect

```python
from bisect import bisect_left, bisect_right

pairs = [(2, "b"), (1, "z"), (2, "a")]
print(sorted(pairs))  # [(1, 'z'), (2, 'a'), (2, 'b')]
print(sorted(pairs, key=lambda p: p[1]))  # [(2, 'a'), (2, 'b'), (1, 'z')]

nums = [5, 1, 3, 3]
nums.sort()
print(nums)  # [1, 3, 3, 5]
print(bisect_left(nums, 3), bisect_right(nums, 3))  # 1 3
```

### Recursion with memoization

```python
from functools import cache


@cache
def climb(n):
    # ways to climb n stairs taking 1 or 2 steps at a time
    if n <= 1:
        return 1
    return climb(n - 1) + climb(n - 2)


print(climb(5))  # 8
print(climb(60))  # 2504730781961
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `vector<int> v; v.push_back(x);` | `v = []` then `v.append(x)` | grows at the end in O(1) |
| `stack<int> st; st.top(); st.pop();` | `st[-1]`, `st.pop()` | a list is the stack; `pop()` returns the item |
| `queue<int> q; q.front(); q.pop();` | `q = deque()`, `q.popleft()` | never `list.pop(0)`: it is O(N) |
| `deque<int> dq; dq.push_front(x);` | `dq.appendleft(x)` | also `append`, `pop`, `popleft` |
| `priority_queue<int> pq;` | `heapq.heappush(h, -x)` | heapq is a min-heap: store `-x` for biggest first |
| `unordered_map<string, int> m;` | `m = {}` or `defaultdict(int)` | plain dict: a missing `m[k]` raises KeyError |
| `unordered_set<int> s; s.count(x)` | `s = set()`, `x in s` | `{}` is an empty dict, not a set |
| `map<int, int> m;` (sorted keys) | `m = {}` then `sorted(m)` | a dict keeps the order you added keys; `sorted(m)` gives sorted keys |
| `lower_bound(...)` / `upper_bound(...)` | `bisect_left(v, x)` / `bisect_right(v, x)` | they return an index, not an iterator |
| `struct Node { int val; Node* next; };` | `class Node:` with `__init__` | `None` is `nullptr`; no `*` and no `->` |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| `v.append(x)`, `v.pop()`, `v[-1]`, `v[i]` | O(1) | same speed for any list size |
| `v.pop(0)`, `v.insert(0, x)` | O(N) | every other item moves one place |
| `x in v` for a list | O(N) | looks at the items one by one |
| `q.append(x)`, `q.popleft()` on a deque | O(1) | no items move |
| `heappush`, `heappop` | O(log N) | about 20 steps for a million items |
| `h[0]` | O(1) | the smallest item is always at index 0 |
| dict `d[k]`, `d.get(k)`, `k in d` | O(1) on average | jumps straight to the key |
| set `s.add(x)`, `x in s` | O(1) on average | the same trick as dict |
| `Counter(items)` | O(N) | looks at each item once |
| `sorted(v)`, `v.sort()` | O(N log N) | a little slower than O(N) |
| `bisect_left`, `bisect_right` | O(log N) | halves the range each step; list must be sorted |
| `@cache` call seen before | O(1) on average | the answer is read back, not worked out again |

## Common mistakes
- Mistake: `queue.pop(0)` to take the front of a list. Every other item moves: O(N) per call. Fix: use a `deque` and `popleft()`.
- Mistake: expecting the biggest item from `heapq`, as with C++ `priority_queue`. Fix: push `-x`, and flip the sign back after `heappop`.
- Mistake: `count[w] += 1` for a new key on a plain dict raises KeyError. Fix: use `Counter`, `defaultdict(int)` or `count.get(w, 0) + 1`.
- Mistake: `seen = {}` for an empty set. That is an empty dict. Fix: `seen = set()`.
- Mistake: `seen.add([r, c])`. A list is not hashable, so you get TypeError. Fix: add the tuple `(r, c)`.
- Mistake: `nums = nums.sort()`. `sort()` returns None, so the list is lost. Fix: `nums.sort()` alone, or `nums = sorted(nums)`.
- Mistake: pushing `(priority, node)` where two priorities are equal. Python then compares the nodes: TypeError. Fix: push `(priority, i, node)` with a unique `i`.

## Practice in the app
- **G-739 Daily Temperatures**: a list used as a stack of indexes; pop while today is warmer than the day on top.
- **G-994 Rotting Oranges**: a deque of rotten cells; `popleft()` one minute at a time.
- **G-200 Number of Islands**: a set of `(r, c)` tuples remembers which land cells you already visited.
- **G-215 Kth Largest Element in an Array**: keep a min-heap of size k; its top `h[0]` is the answer.
- **G-347 Top K Frequent Elements**: `Counter` to count, then a heap or `most_common(k)` to pick the top k.
- **G-23 Merge k Sorted Lists**: `ListNode` plus a heap of `(val, i, node)` tuples.
- **G-56 Merge Intervals**: sort by start with `key=lambda p: p[0]`, then merge neighbours.
- **G-300 Longest Increasing Subsequence**: `bisect_left` finds where each number goes in a sorted helper list.
- **G-198 House Robber**: a small recursive function with `@cache`.
