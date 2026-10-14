# Trees and binary search trees (BST)

> In one line: ask "what must each child tell its parent?" and call the function on each child (DFS). Or visit level by level with a queue (BFS).

## When to use it
- It says "depth", "height", "path" or "sum from root to leaf": use **DFS** with recursion.
- It says "level by level", "right side view" or "zigzag": use **BFS** with a queue.
- It asks for the "lowest common ancestor" of two nodes: use the **LCA** template.
- It says "BST", "valid BST", "sorted order" or "k-th smallest": use the **BST rule**.

## The idea in plain words
A **binary tree** is made of nodes. Each node has a value and up to two children, `left` and `right` (`None` when missing).
The top node is the **root**. A node with no children is a **leaf**. A **subtree** is one node with everything below it.

1. **DFS** (depth-first search): go all the way down one branch, then come back. With recursion (a function that calls itself) it is short.
   Solve the left subtree, solve the right subtree, combine the answers. The **base case** `if node is None:` (the empty tree) stops the calls.
2. **DFS orders** say when you use the node itself: **pre-order** = node, left, right; **in-order** = left, node, right; **post-order** = left, right, node.
3. **BFS** (breadth-first search): visit the tree level by level, top to bottom, with a **queue** (first in, first out: a `deque`).
4. **Path sums**: pass the sum still needed **down** to the children, or return the best sum **up** to the parent.
5. **LCA** (lowest common ancestor; an ancestor is a node on the path above): the lowest node with both `p` and `q` in its subtree.
   A node counts as part of its own subtree, so if `p` is above `q`, the LCA is `p`.
6. **BST** (binary search tree): left subtree values are smaller than the node, right subtree values are bigger.
   So in-order gives sorted values, and a search goes left or right, like binary search. **H** (height) is the number of levels.

## Python template
### DFS (recursion) and BFS (a queue) on one small tree
```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

#     3
#    / \
#   9   20
#      /  \
#     15   7
root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))

def max_depth(node):
    if node is None:  # base case: an empty tree has depth 0
        return 0
    return 1 + max(max_depth(node.left), max_depth(node.right))

def inorder(node, out):
    if node is None:
        return
    inorder(node.left, out)
    out.append(node.val)  # in-order: the node goes between left and right
    inorder(node.right, out)

def level_order(root):
    if root is None:
        return []
    levels = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):  # only the nodes of this level
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        levels.append(level)
    return levels

print(max_depth(root))  # 3
values = []
inorder(root, values)
print(values)  # [9, 3, 15, 20, 7]
levels = level_order(root)
print(levels)  # [[3], [9, 20], [15, 7]]
# "right side view": the last value of each level (level[-1] is the last item)
print([level[-1] for level in levels])  # [3, 20, 7]
```

Move `out.append` above both calls for pre-order: `[3, 9, 20, 15, 7]`. Move it below both for post-order: `[9, 15, 7, 20, 3]`.

### Path sums: pass down, or return up
`has_path_sum` passes the sum still needed **down**. `max_path_sum` returns the best sum **up** to the parent. `nonlocal best` lets the inner function change `best`, a name that belongs to the outer function.
```python
class TreeNode:  # the same class: three assignments in one line
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def has_path_sum(node, target):  # a root-to-leaf path with this sum?
    if node is None:
        return False
    target -= node.val  # what is still needed below this node
    if node.left is None and node.right is None:  # a leaf
        return target == 0
    return has_path_sum(node.left, target) or has_path_sum(node.right, target)

def max_path_sum(root):  # the best path anywhere; it may bend at one node
    best = float("-inf")  # smaller than every number
    def gain(node):  # the best sum of a path going down from node
        nonlocal best
        if node is None:
            return 0
        left = max(gain(node.left), 0)  # a negative branch: leave it out
        right = max(gain(node.right), 0)
        best = max(best, node.val + left + right)  # the path bends here
        return node.val + max(left, right)  # the parent can use one side only
    gain(root)
    return best

# root-to-leaf paths: 5 -> 4 -> 11 (sum 20) and 5 -> 8 (sum 13)
# 9 is False: 5 + 4 is 9, but node 4 is not a leaf
root = TreeNode(5, TreeNode(4, TreeNode(11)), TreeNode(8))
print(has_path_sum(root, 20), has_path_sum(root, 9))  # True False
# best path: 15 -> 20 -> 7, it bends at 20
root = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
print(max_path_sum(root))  # 42
```

### Lowest common ancestor (LCA)
Each call reports what it found in its subtree: `p`, `q`, the answer, or `None`. The children report first (post-order).
```python
class TreeNode:  # the same class, written in fewer lines
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def lca(node, p, q):
    if node is None or node is p or node is q:
        return node
    left = lca(node.left, p, q)
    right = lca(node.right, p, q)
    if left and right:  # p on one side, q on the other
        return node
    if left:
        return left
    return right

#       3
#      / \
#     5   1
#    / \
#   6   2
six, two, one = TreeNode(6), TreeNode(2), TreeNode(1)
five = TreeNode(5, six, two)
root = TreeNode(3, five, one)
print(lca(root, six, two).val, lca(root, six, one).val)  # 5 3
print(lca(root, five, two).val)  # 5
```

### BST rules: limits from every ancestor, and search
```python
class TreeNode:  # the same class, written in fewer lines
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def is_valid_bst(node, low=float("-inf"), high=float("inf")):
    if node is None:
        return True
    if not low < node.val < high:  # low < val and val < high
        return False
    left_ok = is_valid_bst(node.left, low, node.val)
    right_ok = is_valid_bst(node.right, node.val, high)
    return left_ok and right_ok

def bst_search(node, target):  # one step down per level: O(H)
    while node is not None and node.val != target:
        if target < node.val:
            node = node.left
        else:
            node = node.right
    return node is not None

#   4          5
#  / \        / \
# 2   6      4   6
#               / \
#              3   7   <- 3 is right of 5, but smaller than 5
good = TreeNode(4, TreeNode(2), TreeNode(6))
bad = TreeNode(5, TreeNode(4), TreeNode(6, TreeNode(3), TreeNode(7)))
print(is_valid_bst(good), is_valid_bst(bad))  # True False
print(bst_search(good, 6), bst_search(good, 5))  # True False
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `struct TreeNode { int val; TreeNode *left, *right; };` | `class TreeNode:` with `__init__` | a missing child is `None` |
| `queue<TreeNode*> q; q.front(); q.pop();` | `queue = deque()`, `queue.popleft()` | `popleft` also returns the node |
| `int n = q.size(); for (int i = 0; i < n; i++)` | `for _ in range(len(queue)):` | `len(queue)` is read once; `_` = a name we never use |
| `INT_MIN`, `INT_MAX` as limits | `float("-inf")`, `float("inf")` | smaller / bigger than every int: no edge cases |
| an `int& best` parameter or a global | `nonlocal best` in a nested function | lets the inner function change `best` |
| `low < x && x < high` | `low < x < high` | in C++ `low < x < high` is a bug; in Python it is right |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| DFS in any order | O(N) time, O(H) memory | visits each node once; the unfinished calls hold one path |
| BFS level order | O(N) time, O(W) memory | W = the number of nodes on the widest level |
| Path sum, max path sum, LCA, validate BST | O(N) | one DFS over the whole tree |
| BST search or insert | O(H) | one step down per level |
| H for a balanced tree (both sides about equal), or a chain (one child per node) | O(log N), or O(N) | about 20 levels for a million nodes, or N levels |

## Common mistakes
- Mistake: no base case, so `node.left` is read on `None` (AttributeError). Fix: start with `if node is None:`.
- Mistake: a BST check that only compares a node with its two children. The 3 under 6 passes that test. Fix: pass `low` and `high` down.
- Mistake: `best = max(best, ...)` in an inner function without `nonlocal best` raises UnboundLocalError. Fix: add `nonlocal best` first.
- Mistake: BFS without `for _ in range(len(queue))`, so the levels get mixed. Fix: read the level size before the inner loop.
- Mistake: max path sum starting with `best = 0`. When every value is negative, 0 is wrong. Fix: `float("-inf")`.
- Mistake: comparing nodes by value (`node.val == p.val`). Values can repeat. Fix: `node is p`.

## Practice in the app
- **G-199 Binary Tree Right Side View**: BFS by level; keep the last value of each level.
- **G-103 Binary Tree Zigzag Level Order Traversal**: BFS by level; reverse every second level.
- **G-543 Diameter of Binary Tree**: like `gain`: return the height, and remember the best `left + right`.
- **G-124 Binary Tree Maximum Path Sum**: the `max_path_sum` template above.
- **G-236 Lowest Common Ancestor of a Binary Tree**: the `lca` template above.
- **G-98 Validate Binary Search Tree**: pass `low` and `high` down, as in `is_valid_bst`.
- **G-230 Kth Smallest Element in a BST**: in-order gives sorted values; stop at the k-th one.
- **G-863 All Nodes Distance K in Binary Tree**: store each node's parent in a dict, then BFS k steps from the target.
