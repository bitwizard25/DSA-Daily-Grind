# Linked lists: dummy node, fast and slow pointers, reversing

> In one line: a linked list is a chain of boxes and arrows. Draw it, then change the `.next` arrows in a safe order.

## When to use it
- The input is "the head of a linked list" (the first node of the chain).
- You must remove, insert or merge nodes, and the first node may change: add a **dummy node**.
- It asks for the "middle" of the list, or whether the list "has a cycle": use **fast and slow pointers**.
- It says "reverse", "reverse part of the list" or "is the list a palindrome" (reads the same both ways).
- It says "in place" or "O(1) extra memory": change the arrows, do not copy the values into a list.

## The idea in plain words
A **node** is a small object with a value `val` and a link `next` to the following node. The last node's `next` is `None` (C++ `nullptr`).
The **head** is the first node. A variable that holds a node is a **reference** (like a C++ pointer, but with no `*` and no `->`).
So `p.next` in Python is `p->next` in C++. There is no index: to reach node k, you follow k arrows.

Three tools solve most list problems:

1. **Dummy node** (also called a **sentinel**): a fake node placed before the head.
   Now the real head is a normal node with something before it, so the head needs no special case. At the end, return `dummy.next`.
2. **Fast and slow pointers**: two references walk the same list. `slow` moves 1 step, `fast` moves 2 steps.
   When `fast` reaches the end, `slow` is in the middle. If the list has a **cycle** (the last arrow points back into the list), `fast` never reaches an end.
   It goes round and round, catches up with `slow`, and they land on the same node.
3. **Reversing**: walk the list once and turn every arrow around. Keep three names:
   `prev` (the part already reversed), `curr` (the node you fix now) and `nxt` (the rest, saved before you break the arrow).

Hard problems mix the tools. A palindrome check finds the middle (tool 2), reverses the second half (tool 3), then compares the halves.

## Python template

### The node, and a walk through the list

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 1 -> 2 -> 3 -> None
# ListNode(2, ListNode(3)) means: node 2, and its next is node 3
head = ListNode(1, ListNode(2, ListNode(3)))
print(head.val, head.next.val)  # 1 2

values = []
node = head
while node:  # None counts as False, so the loop stops after the last node
    values.append(node.val)
    node = node.next
print(values)  # [1, 2, 3]
```

### Dummy node: remove nodes, merge two lists

First, remove every node with value `target`. Here the head itself (7) must go, and the dummy makes that easy.
Then merge two sorted lists into one sorted list. The answer grows behind the dummy.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def to_list(head):  # the walk from above, so we can print a list
    values = []
    while head:
        values.append(head.val)
        head = head.next
    return values

def remove_value(head, target):
    dummy = ListNode(0, head)  # a fake node before the real head
    prev = dummy
    while prev.next:
        if prev.next.val == target:
            prev.next = prev.next.next  # jump over the node: it is removed
        else:
            prev = prev.next
    return dummy.next  # the real head, which may be a new node

head = ListNode(7, ListNode(1, ListNode(7, ListNode(2))))
print(to_list(remove_value(head, 7)))  # [1, 2]

def merge_two(a, b):
    dummy = ListNode()
    tail = dummy  # the last node of the answer so far
    while a and b:
        if a.val <= b.val:
            tail.next = a
            a = a.next
        else:
            tail.next = b
            b = b.next
        tail = tail.next
    if a:
        tail.next = a  # one list is empty: attach the rest of the other
    else:
        tail.next = b
    return dummy.next

a = ListNode(1, ListNode(4))
b = ListNode(2, ListNode(3, ListNode(5)))
print(to_list(merge_two(a, b)))  # [1, 2, 3, 4, 5]
```

### Fast and slow pointers: middle and cycle

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def middle(head):
    slow = fast = head  # both start at the head
    while fast and fast.next:  # check fast first: fast may be None
        slow = slow.next
        fast = fast.next.next
    return slow

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:  # the same node object
            return True
    return False

head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
print(middle(head).val)  # 3
print(has_cycle(head))  # False

# Make a cycle: the last node (5) points back to node 2
last = head.next.next.next.next
last.next = head.next
print(has_cycle(head))  # True
```

With an even length, like `1 -> 2 -> 3 -> 4`, `middle` returns the second middle node (3).

### Reversing a list

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def to_list(head):
    values = []
    while head:
        values.append(head.val)
        head = head.next
    return values

def reverse(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next  # 1. save the rest of the list
        curr.next = prev  # 2. turn this arrow around
        prev = curr  # 3. move prev one step
        curr = nxt  # 4. move curr one step
    return prev  # the old last node is the new head

head = ListNode(1, ListNode(2, ListNode(3)))
print(to_list(reverse(head)))  # [3, 2, 1]
print(reverse(None))  # None
```

You may also see the loop body in one line: `curr.next, prev, curr = prev, curr, curr.next`.
Python works out the whole right side first, then assigns from left to right.
So `curr.next` must come before `curr`: if `curr` moved first, the arrow of the wrong node would change.

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `struct ListNode { int val; ListNode* next; };` | `class ListNode:` with `__init__` | every node variable is already a reference |
| `nullptr`, `if (!p)` | `None`, `if p is None` | `while node:` stops at `None` |
| `p->next` | `p.next` | no `*` and no `->` |
| `new ListNode(5)` | `ListNode(5)` | no `new` |
| `delete p;` | nothing | Python frees a node when nothing points to it |
| `ListNode dummy(0); ListNode* tail = &dummy;` | `dummy = ListNode()` then `tail = dummy` | |
| `p == q` (same address) | `p is q` | same object, not just the same value |
| `std::swap(a, b);` | `a, b = b, a` | the right side is built first, then assigned |
| `unordered_set<ListNode*> seen;` | `seen = set()` and `seen.add(node)` | each node object is its own key |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| Reach the k-th node | O(k) | no index: follow k arrows |
| Insert or remove right after a node you hold | O(1) | change one arrow |
| Find the middle, check for a cycle | O(N) time, O(1) memory | two names, no extra list |
| Cycle check with a set of seen nodes | O(N) time, O(N) memory | simpler, but stores every node |
| Reverse with a loop | O(N) time, O(1) memory | one walk; only the arrows change |
| Merge two sorted lists | O(N + M) | N and M are the two lengths |

## Common mistakes
- Mistake: `curr.next = prev` before saving `curr.next`. The rest of the list is lost. Fix: `nxt = curr.next` first.
- Mistake: `while fast.next and fast:` crashes when `fast` is None (AttributeError). Fix: `while fast and fast.next:`, because `and` stops at the first False, like `&&`.
- Mistake: returning `head` after removing nodes. The old head may be removed. Fix: return `dummy.next`.
- Mistake: comparing values (`slow.val == fast.val`) to find the meeting point. Two different nodes can hold the same value. Fix: `slow is fast`.
- Mistake: the one-line swap in the wrong order: `prev, curr, curr.next = curr, curr.next, prev`. It moves `curr` first, then changes the wrong node. Fix: put `curr.next` first, or use the 4-line loop.
- Mistake: `if node.val:` to test for the end of the list. A value of 0 counts as False. Fix: `if node:` or `if node is not None:`.
- Mistake: a recursive walk over a very long list stops with RecursionError (too many calls). Fix: use a `while` loop for lists.

## Practice in the app
- **G-23 Merge k Sorted Lists**: the `merge_two` dummy-node loop, plus a heap of list heads to pick the smallest (note 04).

The app's Google bank has only this one pure list problem. The same tools also come back in design questions.
One example is an LRU cache: it drops the least recently used item, using a linked list and a dict.
