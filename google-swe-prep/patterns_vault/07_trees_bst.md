# Trees and BSTs

**Use when:**
- "binary tree", "root", "subtree", "depth", "height", "path between nodes", "ancestor".
- "level by level", "right side view", "zigzag" -> BFS by level.
- "BST", "valid BST", "k-th smallest", "sorted order" -> in-order traversal.
- "k-th ancestor" or LCA with many queries on a fixed tree -> binary lifting.

## Python templates

### TreeNode, DFS (recursive and iterative), BFS level order

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(level):
    """LeetCode level-order list with None gaps -> root."""
    if not level or level[0] is None:
        return None
    root = TreeNode(level[0])
    q, i = deque([root]), 1
    while q and i < len(level):
        node = q.popleft()
        if level[i] is not None:
            node.left = TreeNode(level[i])
            q.append(node.left)
        i += 1
        if i < len(level) and level[i] is not None:
            node.right = TreeNode(level[i])
            q.append(node.right)
        i += 1
    return root


def dfs_orders(root):                # recursive: depth <= ~1000 only
    pre, ino, post = [], [], []

    def go(node):
        if node is None:
            return
        pre.append(node.val)         # append mutates the outer list: no nonlocal needed
        go(node.left)
        ino.append(node.val)
        go(node.right)
        post.append(node.val)

    go(root)
    return pre, ino, post


def preorder_iter(root):
    out = []
    stack = [root] if root else []
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.right:               # push right first so left pops first
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return out


def inorder_iter(root):
    out, stack, node = [], [], root
    while stack or node:
        while node:                  # go left as far as possible
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


def postorder_iter(root):
    """Visit node, right, left; reversed that is left, right, node."""
    out = []
    stack = [root] if root else []
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return out[::-1]


def level_order(root):
    out = []
    q = deque([root] if root else [])
    while q:
        level = []
        for _ in range(len(q)):      # len(q) is read once: exactly this level
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        out.append(level)
    return out


def right_side_view(root):           # G-199
    return [level[-1] for level in level_order(root)]


def zigzag(root):                    # G-103
    return [lvl if d % 2 == 0 else lvl[::-1] for d, lvl in enumerate(level_order(root))]


#     3
#    / \
#   9  20
#      / \
#     15  7
root = build_tree([3, 9, 20, None, None, 15, 7])
pre, ino, post = dfs_orders(root)
assert pre == preorder_iter(root) == [3, 9, 20, 15, 7]
assert ino == inorder_iter(root) == [9, 3, 15, 20, 7]
assert post == postorder_iter(root) == [9, 15, 7, 20, 3]
assert level_order(root) == [[3], [9, 20], [15, 7]]
assert zigzag(root) == [[3], [20, 9], [15, 7]]
assert right_side_view(build_tree([1, 2, 3, None, 5, None, 4])) == [1, 3, 4]
assert level_order(None) == [] and inorder_iter(None) == []
```

### Post-order: height, diameter, max path sum, LCA

```python
import math
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(level):
    if not level or level[0] is None:
        return None
    root = TreeNode(level[0])
    q, i = deque([root]), 1
    while q and i < len(level):
        node = q.popleft()
        if level[i] is not None:
            node.left = TreeNode(level[i])
            q.append(node.left)
        i += 1
        if i < len(level) and level[i] is not None:
            node.right = TreeNode(level[i])
            q.append(node.right)
        i += 1
    return root


def max_depth(root):                 # nodes on the longest root-to-leaf path
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def diameter(root):                  # G-543: EDGES on the longest path
    best = 0

    def height(node):
        nonlocal best                # needed: we ASSIGN to best below
        if node is None:
            return 0
        lh, rh = height(node.left), height(node.right)
        best = max(best, lh + rh)    # path that bends at node
        return 1 + max(lh, rh)

    height(root)
    return best


def max_path_sum(root):              # G-124
    best = -math.inf                 # not 0: every value may be negative

    def gain(node):                  # best downward path sum that starts at node
        nonlocal best
        if node is None:
            return 0
        left = max(gain(node.left), 0)     # a negative branch is dropped
        right = max(gain(node.right), 0)
        best = max(best, node.val + left + right)   # bend here: both sides
        return node.val + max(left, right)          # going up: one side only

    gain(root)
    return best


assert max_depth(build_tree([3, 9, 20, None, None, 15, 7])) == 3
assert diameter(build_tree([1, 2, 3, 4, 5])) == 3
assert diameter(build_tree([1, 2])) == 1
assert max_path_sum(build_tree([1, 2, 3])) == 6
assert max_path_sum(build_tree([-10, 9, 20, None, None, 15, 7])) == 42
assert max_path_sum(build_tree([-3])) == -3
assert max_path_sum(build_tree([-2, -1])) == -1


def broken_count(root):
    count = 0

    def go(node):
        if node:
            count += 1               # assignment makes `count` local to go()
            go(node.left)
            go(node.right)

    go(root)
    return count


try:
    broken_count(TreeNode(1))
except UnboundLocalError:            # fix: `nonlocal count` inside go()
    broke = True
else:
    broke = False
assert broke


def lca(root, p, q):                 # G-236: p and q are nodes in the tree
    if root is None or root is p or root is q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root                  # one target on each side
    return left or right             # both on one side (or neither)


def find(root, val):
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            if node.val == val:
                return node
            stack += [node.left, node.right]
    return None


t = build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
assert lca(t, find(t, 5), find(t, 1)).val == 3
assert lca(t, find(t, 5), find(t, 4)).val == 5      # a node is its own ancestor
assert lca(t, find(t, 7), find(t, 8)).val == 3
```

### Binary lifting: k-th ancestor and LCA in O(log N)

```python
import random
from collections import deque


class TreeAncestor:                  # G-1483
    def __init__(self, n, parent):
        self.n = n
        self.log = max(1, n.bit_length())   # 2**log > n - 1 >= any depth
        self.up = [parent[:]]                # up[j][v] = 2**j-th ancestor of v, -1 if none
        for j in range(1, self.log):
            prev = self.up[-1]
            # guard -1: prev[-1] would silently read the LAST entry
            self.up.append([-1 if prev[v] == -1 else prev[prev[v]] for v in range(n)])

    def kth_ancestor(self, node, k):
        if k >= self.n:              # deeper than any tree with n nodes
            return -1
        j = 0
        while k and node != -1:
            if k & 1:                # bit j of k set: jump 2**j levels
                node = self.up[j][node]
            k >>= 1
            j += 1
        return node


def depths(parent):
    children = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p != -1:
            children[p].append(v)
    depth = [0] * len(parent)
    q = deque([parent.index(-1)])    # BFS from the root
    while q:
        u = q.popleft()
        for v in children[u]:
            depth[v] = depth[u] + 1
            q.append(v)
    return depth


def lca_lifting(ta, depth, a, b):
    if depth[a] < depth[b]:
        a, b = b, a
    a = ta.kth_ancestor(a, depth[a] - depth[b])     # same depth now
    if a == b:
        return a
    for j in reversed(range(ta.log)):
        if ta.up[j][a] != ta.up[j][b]:              # still below the LCA: jump both
            a, b = ta.up[j][a], ta.up[j][b]
    return ta.up[0][a]


ta = TreeAncestor(7, [-1, 0, 0, 1, 1, 2, 2])
assert ta.kth_ancestor(3, 1) == 1
assert ta.kth_ancestor(5, 2) == 0
assert ta.kth_ancestor(6, 3) == -1

# Brute-force check on a random tree whose labels are shuffled (parent[v] < v is NOT assumed).
random.seed(1483)
n = 60
base = [-1] + [random.randrange(v) for v in range(1, n)]
label = list(range(n))
random.shuffle(label)
parent = [0] * n
for v in range(n):
    parent[label[v]] = -1 if base[v] == -1 else label[base[v]]


def naive_kth(v, k):
    while k and v != -1:
        v, k = parent[v], k - 1
    return v


def naive_lca(a, b):
    anc = set()
    while a != -1:
        anc.add(a)
        a = parent[a]
    while b not in anc:
        b = parent[b]
    return b


ta, depth = TreeAncestor(n, parent), depths(parent)
for v in range(n):
    for k in range(n + 2):
        assert ta.kth_ancestor(v, k) == naive_kth(v, k)
    for w in range(n):
        assert lca_lifting(ta, depth, v, w) == naive_lca(v, w)
```

### BST: validate with bounds, in-order is sorted

```python
import math
import random
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(level):
    if not level or level[0] is None:
        return None
    root = TreeNode(level[0])
    q, i = deque([root]), 1
    while q and i < len(level):
        node = q.popleft()
        if level[i] is not None:
            node.left = TreeNode(level[i])
            q.append(node.left)
        i += 1
        if i < len(level) and level[i] is not None:
            node.right = TreeNode(level[i])
            q.append(node.right)
        i += 1
    return root


def is_valid_bst(node, low=-math.inf, high=math.inf):   # G-98
    if node is None:
        return True
    if not low < node.val < high:    # chained comparison; strict: duplicates are invalid
        return False
    return is_valid_bst(node.left, low, node.val) and is_valid_bst(node.right, node.val, high)


def children_only_check(node):       # WRONG: ignores ancestors
    if node is None:
        return True
    if node.left and node.left.val >= node.val:
        return False
    if node.right and node.right.val <= node.val:
        return False
    return children_only_check(node.left) and children_only_check(node.right)


bad = build_tree([5, 4, 6, None, None, 3, 7])   # 3 sits in the RIGHT subtree of 5
assert children_only_check(bad)                  # the trap passes it
assert not is_valid_bst(bad)
assert is_valid_bst(build_tree([2, 1, 3]))
assert not is_valid_bst(build_tree([1, 1]))
assert is_valid_bst(build_tree([2**31 - 1]))     # no INT_MAX sentinel problem


def inorder_iter(root):
    out, stack, node = [], [], root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


def is_valid_bst_inorder(root):      # in-order must be strictly increasing
    vals = inorder_iter(root)
    return all(a < b for a, b in zip(vals, vals[1:]))


assert not is_valid_bst_inorder(bad) and is_valid_bst_inorder(build_tree([2, 1, 3]))


def kth_smallest(root, k):           # G-230: stop early, O(H + k)
    stack, node = [], root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        k -= 1
        if k == 0:
            return node.val
        node = node.right
    raise ValueError("k is larger than the tree")


assert kth_smallest(build_tree([3, 1, 4, None, 2]), 1) == 1
assert kth_smallest(build_tree([5, 3, 6, 2, 4, None, None, 1]), 3) == 3


def bst_insert(root, val):           # iterative, ignores duplicates
    new = TreeNode(val)
    if root is None:
        return new
    node = root
    while True:
        if val == node.val:
            return root
        side = "left" if val < node.val else "right"
        child = getattr(node, side)
        if child is None:
            setattr(node, side, new)
            return root
        node = child


def bst_search(root, val):           # O(H)
    node = root
    while node and node.val != val:
        node = node.left if val < node.val else node.right
    return node


random.seed(98)
vals = [random.randint(0, 100) for _ in range(200)]
root = None
for v in vals:
    root = bst_insert(root, v)
assert inorder_iter(root) == sorted(set(vals))   # in-order of a BST = sorted
assert is_valid_bst(root)
assert bst_search(root, vals[0]).val == vals[0]
assert bst_search(root, 101) is None
```

### Recursion limit: skewed trees

```python
import sys
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root):
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def max_depth_bfs(root):             # no recursion: safe at any depth
    depth = 0
    q = deque([root] if root else [])
    while q:
        depth += 1
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return depth


assert sys.getrecursionlimit() == 1000          # CPython default

chain = None
for v in range(5000):                # a "tree" that is a 5000-node left chain
    chain = TreeNode(v, chain)

try:
    max_depth(chain)
except RecursionError:
    too_deep = True
else:
    too_deep = False
assert too_deep
assert max_depth_bfs(chain) == 5000
```

C++ for comparison: `int` bounds reject a node equal to `INT_MAX`. Python's `math.inf` has no such edge.

```cpp
#include <cassert>
#include <climits>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int v, TreeNode* l = nullptr, TreeNode* r = nullptr) : val(v), left(l), right(r) {}
};

bool validIntBounds(const TreeNode* n, int lo = INT_MIN, int hi = INT_MAX) {
    if (!n) return true;
    if (n->val <= lo || n->val >= hi) return false;          // INT_MAX itself fails here
    return validIntBounds(n->left, lo, n->val) && validIntBounds(n->right, n->val, hi);
}

bool valid(const TreeNode* n, long long lo = LLONG_MIN, long long hi = LLONG_MAX) {
    if (!n) return true;
    if (n->val <= lo || n->val >= hi) return false;
    return valid(n->left, lo, n->val) && valid(n->right, n->val, hi);
}

int main() {
    TreeNode single(INT_MAX);
    assert(!validIntBounds(&single));        // wrong answer: the tree is a valid BST
    assert(valid(&single));

    TreeNode three(3), seven(7), four(4);
    TreeNode six(6, &three, &seven);
    TreeNode root(5, &four, &six);           // 3 is in the right subtree of 5
    assert(!valid(&root));
    return 0;
}
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `struct TreeNode { int val; TreeNode *left, *right; }` | `class TreeNode` with `val`, `left`, `right` | O(1) per node |
| `if (!root) return 0;` | `if root is None: return 0` | O(1) |
| answer in a global or `int&` parameter | `nonlocal best` in a nested function, or `self.best` | O(1) |
| `std::stack<TreeNode*>` | `list`: `append`, `pop`, `[-1]` | O(1) |
| `std::queue<TreeNode*>`, `q.size()` per level | `deque`, `for _ in range(len(q))` | O(1) per op |
| `INT_MIN` / `LLONG_MAX` as bounds | `-math.inf`, `math.inf` | O(1) |
| `lo < x && x < hi` | `lo < x < hi` (chained) | O(1) |
| `std::max({a, b, c})` | `max(a, b, c)` | O(1) |
| `unordered_map<TreeNode*, TreeNode*>` parent map | `dict` keyed by node | average O(1) |
| `vector<vector<int>> up(LOG, vector<int>(n))` | list of lists, one row per power of two | O(N log N) build |
| `32 - __builtin_clz(n)` (undefined for 0) | `n.bit_length()` (0 for 0) | O(1) |
| `k & 1`, `k >>= 1` | same operators | O(1) |
| `std::set` / `std::map` (balanced BST) | no stdlib tree: sorted list + `bisect` | O(log N) search, O(N) insert |

## Complexity

| Operation | Time | Space |
|---|---|---|
| DFS (any order), recursive or iterative | O(N) | O(H) stack |
| BFS level order | O(N) | O(W), W = widest level |
| Height H | balanced: O(log N) | skewed: O(N) |
| BST search / insert / delete | O(H) | O(1) iterative |
| Validate BST (bounds or in-order) | O(N) | O(H) |
| K-th smallest in a BST | O(H + k) | O(H) |
| Diameter, max path sum, height | O(N) | O(H) |
| LCA by post-order | O(N) per query | O(H) |
| Binary lifting build | O(N log N) | O(N log N) |
| Binary lifting k-th ancestor / LCA query | O(log N) | O(1) |

## Gotchas for C++ developers

- Assigning to an outer variable inside a nested function raises **UnboundLocalError**. Add `nonlocal`.
- Mutating an outer list (`out.append`) needs no `nonlocal`; rebinding a name (`best = ...`) does.
- Recursion limit is **1000**: a skewed tree of 10^4 nodes raises RecursionError. Use a stack or BFS.
- `sys.setrecursionlimit` raises the cap, but recursion through C code such as `@cache` can still fail.
- BST check: compare against bounds from **all ancestors**, not only the children.
- Use `-math.inf` / `math.inf` bounds. No `INT_MIN` / `INT_MAX` edge cases, no `long long` widening.
- Max path sum starts at `-math.inf`, not 0, and clamps each branch with `max(gain, 0)`.
- Diameter counts **edges**: `lh + rh` where heights count nodes.
- In binary lifting, `up[j][-1]` silently reads the last entry. Guard the `-1` "no parent" value.
- `node is p` compares nodes. Values can repeat; identities cannot.
- `if node` tests the node. `if node.val` is False for a value of 0.
- `TreeNode` keys in a dict hash by identity. Do not add `__eq__`, or nodes become unhashable.
- `for _ in range(len(q))` reads `len(q)` once, so appending children does not extend the level.

## In the Google bank

- **BFS by level:** G-199 Binary Tree Right Side View, G-103 Binary Tree Zigzag Level Order Traversal.
- **Post-order with a global:** G-543 Diameter of Binary Tree, G-124 Binary Tree Maximum Path Sum.
- **LCA:** G-236 Lowest Common Ancestor of a Binary Tree.
- **Binary lifting:** G-1483 Kth Ancestor of a Tree Node.
- **BST order:** G-98 Validate Binary Search Tree, G-230 Kth Smallest Element in a BST.
- **Pre-order with null markers:** G-297 Serialize and Deserialize Binary Tree.
- **Build from traversals:** G-105 Construct Binary Tree from Preorder and Inorder Traversal.
- **Tree to graph + BFS:** G-863 All Nodes Distance K in Binary Tree.
- **DFS with coordinates, then sort:** G-987 Vertical Order Traversal of a Binary Tree.
