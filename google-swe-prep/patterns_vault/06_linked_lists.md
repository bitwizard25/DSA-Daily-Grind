# Linked Lists

**Use when:**
- The input is "the head of a linked list"; "in place", "O(1) extra space".
- "reverse", "reverse between", "reorder", "palindrome list".
- "middle", "has a cycle", "where does the cycle begin", "n-th node from the end".
- "merge two / k sorted lists", "sort a linked list".

## Python templates

### ListNode, helpers for asserts, identity

```python
class ListNode:
    def __init__(self, val=0, next=None):    # LeetCode signature; `next` shadows the builtin here only
        self.val = val
        self.next = next


def from_list(vals):
    dummy = tail = ListNode()        # ONE node with two names
    for v in vals:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head):
    out = []
    while head:                      # None is falsy, like nullptr
        out.append(head.val)
        head = head.next
    return out


assert to_list(from_list([1, 2, 3])) == [1, 2, 3]
assert from_list([]) is None

a, b = ListNode(1), ListNode(1)
assert a is not b                    # `is` compares identity, like comparing pointers
assert a != b                        # default == is identity too
assert len({a, b}) == 2              # nodes hash by identity: a set of visited nodes works

zero = ListNode(0)
assert zero                          # a node is truthy even when val == 0
assert not zero.val                  # so test `node`, never `node.val`


class ValueNode(ListNode):
    def __eq__(self, other):         # defining __eq__ sets __hash__ to None
        return self.val == other.val


try:
    hash(ValueNode(1))
except TypeError:                    # unhashable: cannot go in a set or be a dict key
    unhashable = True
else:
    unhashable = False
assert unhashable
```

### Dummy (sentinel) head

```python
class ListNode:
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


def remove_elements(head, target):
    dummy = ListNode(0, head)        # the real head is now removable like any node
    prev = dummy
    while prev.next:
        if prev.next.val == target:
            prev.next = prev.next.next   # no delete: the unlinked node is freed for you
        else:
            prev = prev.next
    return dummy.next


assert to_list(remove_elements(from_list([7, 7, 1, 7, 2]), 7)) == [1, 2]
assert remove_elements(from_list([7, 7]), 7) is None


def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    slow = fast = dummy
    for _ in range(n + 1):           # fast runs n + 1 nodes ahead
        fast = fast.next
    while fast:
        slow, fast = slow.next, fast.next
    slow.next = slow.next.next       # slow stops just before the target
    return dummy.next


assert to_list(remove_nth_from_end(from_list([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
assert to_list(remove_nth_from_end(from_list([1, 2]), 2)) == [2]
assert remove_nth_from_end(from_list([1]), 1) is None
```

### Reverse: iterative with multi-assignment

```python
class ListNode:
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


def reverse(head):
    prev, curr = None, head
    while curr:
        # Right side first: (prev, curr, curr.next) with OLD values.
        # Then targets left to right: curr.next is set while curr is still the old node.
        curr.next, prev, curr = prev, curr, curr.next
    return prev


assert to_list(reverse(from_list([1, 2, 3, 4]))) == [4, 3, 2, 1]
assert reverse(None) is None


def reverse_explicit(head):          # the same steps, C++ style
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev


assert to_list(reverse_explicit(from_list([1, 2, 3]))) == [3, 2, 1]


def reverse_wrong(head):
    prev, curr = None, head
    while curr:
        prev, curr, curr.next = curr, curr.next, prev   # curr is rebound BEFORE curr.next
    return prev


try:
    reverse_wrong(from_list([1, 2]))
except AttributeError:               # 'NoneType' object has no attribute 'next'
    crashed = True
else:
    crashed = False
assert crashed


def reverse_between(head, left, right):   # 1-indexed, inclusive
    dummy = ListNode(0, head)
    before = dummy
    for _ in range(left - 1):
        before = before.next
    prev, curr = None, before.next
    for _ in range(right - left + 1):
        curr.next, prev, curr = prev, curr, curr.next
    before.next.next = curr          # old segment head is now its tail
    before.next = prev               # new segment head
    return dummy.next


assert to_list(reverse_between(from_list([1, 2, 3, 4, 5]), 2, 4)) == [1, 4, 3, 2, 5]
assert to_list(reverse_between(from_list([1, 2]), 1, 2)) == [2, 1]
assert to_list(reverse_between(from_list([5]), 1, 1)) == [5]


def reverse_recursive(head):         # O(N) stack frames
    if head is None or head.next is None:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head


assert to_list(reverse_recursive(from_list([1, 2, 3]))) == [3, 2, 1]

long_list = from_list(range(5000))
try:
    reverse_recursive(long_list)     # default recursion limit is 1000
except RecursionError:
    too_deep = True
else:
    too_deep = False
assert too_deep
assert to_list(reverse(long_list))[:3] == [4999, 4998, 4997]   # iterative is fine
```

### Fast and slow pointers: middle, cycle, cycle start

```python
class ListNode:
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


def middle(head):                    # second middle on even length
    slow = fast = head
    while fast and fast.next:        # `and` short-circuits, so fast.next is safe
        slow, fast = slow.next, fast.next.next
    return slow


def first_middle(head):              # end of the first half (head must not be None)
    slow = fast = head
    while fast.next and fast.next.next:
        slow, fast = slow.next, fast.next.next
    return slow


assert middle(from_list([1, 2, 3, 4, 5])).val == 3
assert middle(from_list([1, 2, 3, 4])).val == 3
assert first_middle(from_list([1, 2, 3, 4])).val == 2
assert first_middle(from_list([1, 2, 3])).val == 2
assert middle(None) is None


def make_cycle(vals, pos):
    """Tail links to the node at index pos; pos == -1 means no cycle."""
    nodes = [ListNode(v) for v in vals]
    for x, y in zip(nodes, nodes[1:]):
        x.next = y
    if pos == -1:                    # nodes[-1] would silently be the LAST node
        return nodes[0], None
    nodes[-1].next = nodes[pos]
    return nodes[0], nodes[pos]


def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:             # identity, not ==
            return True
    return False


def cycle_start(head):               # Floyd: O(N) time, O(1) space
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            # head -> start == meeting point -> start (mod cycle length)
            slow = head
            while slow is not fast:
                slow, fast = slow.next, fast.next
            return slow
    return None


def cycle_start_set(head):           # O(N) extra space, simpler to get right
    seen = set()
    while head:
        if head in seen:
            return head
        seen.add(head)
        head = head.next
    return None


for vals, pos in [([3, 2, 0, -4], 1), ([1, 2], 0), ([1], 0), ([1], -1), ([1, 2, 3, 4, 5, 6], 3)]:
    head, start = make_cycle(vals, pos)
    assert has_cycle(head) == (start is not None)
    assert cycle_start(head) is start
    assert cycle_start_set(head) is start
```

### Merge two sorted lists, and what it builds

```python
class ListNode:
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


def merge_two(l1, l2):
    dummy = tail = ListNode()
    while l1 and l2:
        if l1.val <= l2.val:         # <= keeps it stable: l1 wins ties
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2             # attach the rest; `or` returns the non-None one
    return dummy.next


assert to_list(merge_two(from_list([1, 2, 4]), from_list([1, 3, 4]))) == [1, 1, 2, 3, 4, 4]
assert merge_two(None, None) is None
assert to_list(merge_two(None, from_list([0]))) == [0]
x, y = ListNode(1), ListNode(1)
assert merge_two(x, y) is x          # stable


def first_middle(head):
    slow = fast = head
    while fast.next and fast.next.next:
        slow, fast = slow.next, fast.next.next
    return slow


def sort_list(head):                 # merge sort: O(N log N) time, O(log N) stack
    if head is None or head.next is None:
        return head
    mid = first_middle(head)
    right, mid.next = mid.next, None # cut into two halves
    return merge_two(sort_list(head), sort_list(right))


assert to_list(sort_list(from_list([4, 2, 1, 3]))) == [1, 2, 3, 4]
assert to_list(sort_list(from_list([-1, 5, 3, 4, 0]))) == [-1, 0, 3, 4, 5]


def reverse(head):
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    return prev


def is_palindrome(head):             # middle + reverse second half, then restore it
    if head is None:
        return True
    mid = first_middle(head)
    second = reverse(mid.next)
    p, q, ok = head, second, True
    while q:
        if p.val != q.val:
            ok = False
            break
        p, q = p.next, q.next
    mid.next = reverse(second)       # leave the input as it was
    return ok


lst = from_list([1, 2, 3, 2, 1])
assert is_palindrome(lst) and to_list(lst) == [1, 2, 3, 2, 1]
assert is_palindrome(from_list([1, 2, 2, 1]))
assert not is_palindrome(from_list([1, 2]))
```

The C++ version for comparison. Python has no `delete`: the garbage collector frees unlinked nodes.

```cpp
#include <cassert>
#include <vector>

struct ListNode {
    int val;
    ListNode* next;
    explicit ListNode(int v, ListNode* n = nullptr) : val(v), next(n) {}
};

ListNode* fromVector(const std::vector<int>& vals) {
    ListNode dummy(0);                       // on the stack: nothing to free
    ListNode* tail = &dummy;
    for (int v : vals) {
        tail->next = new ListNode(v);
        tail = tail->next;
    }
    return dummy.next;
}

std::vector<int> toVector(const ListNode* head) {
    std::vector<int> out;
    for (; head; head = head->next) out.push_back(head->val);
    return out;
}

ListNode* reverse(ListNode* head) {
    ListNode* prev = nullptr;
    while (head) {
        ListNode* nxt = head->next;          // Python: head.next, prev, head = prev, head, head.next
        head->next = prev;
        prev = head;
        head = nxt;
    }
    return prev;
}

void freeList(ListNode* head) {
    while (head) {
        ListNode* nxt = head->next;
        delete head;
        head = nxt;
    }
}

int main() {
    ListNode* head = reverse(fromVector({1, 2, 3}));
    assert((toVector(head) == std::vector<int>{3, 2, 1}));
    freeList(head);
    return 0;
}
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `struct ListNode { int val; ListNode* next; }` | `class ListNode` with `self.val`, `self.next` | O(1) per node |
| `nullptr`, `if (!p)` | `None`, `if not p` / `if p is None` | O(1) |
| `p->next` | `p.next` | O(1) |
| `p == q` (same node) | `p is q` | O(1) |
| `ListNode dummy(0); ListNode* tail = &dummy;` | `dummy = tail = ListNode()` | O(1) |
| `new ListNode(v)` / `delete p` | `ListNode(v)` / drop the last reference | O(1) |
| temp `nxt` in reverse | `curr.next, prev, curr = prev, curr, curr.next` | O(1) per step |
| `unordered_set<ListNode*>` | `set()` of nodes (identity hash) | average O(1) |
| `unordered_map<Node*, Node*>` old -> copy | `dict` keyed by node | average O(1) |
| `std::list` push/pop at both ends | `collections.deque` | O(1) |
| `std::list::splice` | relink `.next` fields by hand | O(1) |
| `std::list::sort` (merge sort) | `sort_list` above, or copy values out and `sorted` | O(N log N) |
| walk to count nodes | no `len()` on a chain: walk it | O(N) |

## Complexity

| Operation | Time | Space |
|---|---|---|
| Access the k-th node | O(k) | O(1) |
| Insert / delete after a known node | O(1) | O(1) |
| Delete by value from the head | O(N) | O(1) |
| Remove n-th from end (gap pointers) | O(N), one pass | O(1) |
| Reverse, iterative | O(N) | O(1) |
| Reverse, recursive | O(N) | O(N) stack: fails past ~1000 nodes |
| Middle, cycle check, cycle start (fast/slow) | O(N) | O(1) |
| Cycle start with a `set` | O(N) | O(N) |
| Merge two sorted lists | O(N + M) | O(1) |
| Merge sort a list | O(N log N) | O(log N) stack |
| Palindrome check (reverse half) | O(N) | O(1) |

## Gotchas for C++ developers

- Multi-assignment evaluates the whole right side first, then assigns targets **left to right**.
- So `curr.next` must come before `curr` in the target list, or you write to the new node.
- `a = b = ListNode()` makes **one** node with two names. Two calls make two nodes.
- The Python "segfault" is `AttributeError: 'NoneType' object has no attribute 'next'`.
- Guard with `fast and fast.next`, in that order: `and` short-circuits like `&&`.
- `p is q` compares nodes. `==` falls back to identity only until someone defines `__eq__`.
- Defining `__eq__` without `__hash__` makes nodes unhashable: no `set` of visited nodes.
- Test `if node`, not `if node.val`: a value of 0 is falsy.
- `nodes[-1]` is the last item, not an error. A `pos = -1` "no cycle" flag needs an explicit check.
- Recursive reverse / merge hit the 1000-frame recursion limit on long lists. Use loops.
- No `delete`: unlinking is enough. A node stays alive while any variable or dict still refers to it.
- Nodes define no `<`, so `heapq` on `(val, node)` crashes on ties. Use `(val, i, node)` (note 04).

## In the Google bank

- **Merge with a heap:** G-23 Merge k Sorted Lists (merge two lists, generalised to k).
- **Old node -> new node map:** G-133 Clone Graph (same idea as copying a list with random pointers).
- The bank has few pure list problems. These templates are building blocks for G-23 and for design questions such as an LRU cache.
