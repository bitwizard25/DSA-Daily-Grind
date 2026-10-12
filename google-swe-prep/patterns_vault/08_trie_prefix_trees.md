# Tries (Prefix Trees)

**Use when:**
- "prefix", "starts with", "autocomplete", "type-ahead", "dictionary of words".
- Many words must be found at once in a grid or text (one DFS for all words).
- "longest common prefix", "replace each word with its shortest root".
- Repeated prefix queries where sorting + `bisect` would be rebuilt too often.

## Python templates

### Dict-of-children trie: insert, search, startsWith

```python
from collections import defaultdict


class TrieNode:
    __slots__ = ("children", "end")  # optional: no per-instance __dict__

    def __init__(self):
        self.children = {}           # char -> TrieNode
        self.end = False             # a whole word ends here


class Trie:                          # G-208
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):          # O(L)
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.end = True

    def _walk(self, s):              # node for prefix s, or None
        node = self.root
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word):          # O(L): must end on a word
        node = self._walk(word)
        return node is not None and node.end

    def starts_with(self, prefix):   # O(L): any node is enough
        return self._walk(prefix) is not None


t = Trie()
t.insert("apple")
assert t.search("apple")
assert not t.search("app")           # a prefix, not a word
assert t.starts_with("app")
t.insert("app")
assert t.search("app")
assert not t.starts_with("b")
assert t.starts_with("")             # empty prefix matches the root


def make_node():                     # a "trie" of nested defaultdicts
    return defaultdict(make_node)


auto = make_node()
node = auto
for ch in "cat":
    node = node[ch]
node = auto
for ch in "dog":                     # meant as a lookup...
    node = node[ch]
assert "d" in auto                   # ...but every miss inserted a node


END = "$"                            # dict-of-dicts variant; END must not be a letter
root = {}
for w in ["car", "cart"]:
    node = root
    for ch in w:
        node = node.setdefault(ch, {})
    node[END] = True


def dd_search(word):
    node = root
    for ch in word:
        if ch not in node:
            return False
        node = node[ch]
    return END in node


assert dd_search("car") and dd_search("cart") and not dd_search("ca")
```

### Word Search II: one trie, one DFS, pruning

```python
class TrieNode:
    __slots__ = ("children", "word")

    def __init__(self):
        self.children = {}
        self.word = None             # the full word that ends here


def find_words(board, words):        # G-212
    root = TrieNode()
    for w in words:
        node = root
        for ch in w:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.word = w

    rows, cols = len(board), len(board[0])
    found = []

    def dfs(r, c, parent):
        ch = board[r][c]
        node = parent.children[ch]
        if node.word is not None:
            found.append(node.word)
            node.word = None         # report each word once
        board[r][c] = "#"            # mark visited (no extra set)
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] in node.children:
                dfs(nr, nc, node)
        board[r][c] = ch             # restore on the way back
        if not node.children and node.word is None:
            del parent.children[ch]  # prune: nothing left to find below this node

    for r in range(rows):
        for c in range(cols):
            if board[r][c] in root.children:
                dfs(r, c, root)
    return found


board = [list("oaan"), list("etae"), list("ihkr"), list("iflv")]
before = [row[:] for row in board]
assert sorted(find_words(board, ["oath", "pea", "eat", "rain"])) == ["eat", "oath"]
assert board == before               # the board is restored
assert find_words([["a", "b"], ["c", "d"]], ["abcb"]) == []   # a cell is used once per word
assert sorted(find_words([["a", "a"]], ["a", "aa", "aaa"])) == ["a", "aa"]
assert sorted(find_words([["a", "b"]], ["ab", "ba", "ab"])) == ["ab", "ba"]   # no duplicates
```

### Autocomplete: top-k by count (cached per node)

```python
import random


class Node:
    __slots__ = ("children", "hot")

    def __init__(self):
        self.children = {}
        self.hot = []                # top K sentences under this prefix, best first


class AutocompleteSystem:            # G-642
    K = 3

    def __init__(self, sentences, times):
        self.root = Node()
        self.count = {}              # sentence -> times typed
        for s, t in zip(sentences, times):
            self._add(s, t)
        self.typed = []
        self.node = self.root        # trie node of the typed prefix; None = no match

    def _rank(self, s):
        return (-self.count[s], s)   # higher count first, then ASCII order

    def _add(self, s, times):        # O(L * K log K)
        self.count[s] = self.count.get(s, 0) + times
        node = self.root
        for ch in s:
            if ch not in node.children:
                node.children[ch] = Node()
            node = node.children[ch]
            if s not in node.hot:
                node.hot.append(s)
            node.hot.sort(key=self._rank)
            del node.hot[self.K:]    # counts only grow, so a dropped sentence never returns
                                     # unless its own count grows, which re-adds it here

    def input(self, c):              # LeetCode's name; methods still see the builtin input()
        if c == "#":
            self._add("".join(self.typed), 1)
            self.typed = []
            self.node = self.root
            return []
        self.typed.append(c)
        if self.node is not None:
            self.node = self.node.children.get(c)   # one step per keystroke
        return list(self.node.hot) if self.node else []


ac = AutocompleteSystem(["i love you", "island", "iroman", "i love leetcode"], [5, 3, 2, 2])
assert ac.input("i") == ["i love you", "island", "i love leetcode"]
assert ac.input(" ") == ["i love you", "i love leetcode"]
assert ac.input("a") == []
assert ac.input("#") == []
assert ac.input("i") == ["i love you", "island", "i love leetcode"]
assert ac.input(" ") == ["i love you", "i love leetcode", "i a"]
assert ac.input("a") == ["i a"]
assert ac.input("#") == []


def brute(count, prefix, k=3):
    hits = [s for s in count if s.startswith(prefix)]
    return sorted(hits, key=lambda s: (-count[s], s))[:k]


random.seed(642)
start = ["".join(random.choice("ab ") for _ in range(random.randint(1, 4))) for _ in range(8)]
start = list(dict.fromkeys(start))   # unique sentences, order kept
ac = AutocompleteSystem(start, [random.randint(1, 3) for _ in start])
for _ in range(300):
    typed = ""
    for ch in "".join(random.choice("ab ") for _ in range(random.randint(1, 4))):
        typed += ch
        assert ac.input(ch) == brute(ac.count, typed)
    assert ac.input("#") == []
```

The array-pool trie common in C++ contest code, for comparison with the dict version.

```cpp
#include <array>
#include <cassert>
#include <string>
#include <vector>

struct Trie {
    std::vector<std::array<int, 26>> next{std::array<int, 26>{}};   // node 0 = root; 0 = no child
    std::vector<bool> end{false};

    void insert(const std::string& w) {
        int u = 0;
        for (char ch : w) {
            int c = ch - 'a';
            if (!next[u][c]) {
                int id = static_cast<int>(next.size());
                next.push_back({});      // may reallocate: take no reference into next before this
                end.push_back(false);
                next[u][c] = id;
            }
            u = next[u][c];
        }
        end[u] = true;
    }

    int walk(const std::string& s) const {
        int u = 0;
        for (char ch : s) {
            u = next[u][ch - 'a'];
            if (!u) return -1;
        }
        return u;
    }

    bool search(const std::string& w) const { int u = walk(w); return u != -1 && end[u]; }
    bool startsWith(const std::string& p) const { return walk(p) != -1; }
};

int main() {
    Trie t;
    t.insert("apple");
    assert(t.search("apple"));
    assert(!t.search("app"));
    assert(t.startsWith("app"));
    t.insert("app");
    assert(t.search("app"));
    return 0;
}
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `TrieNode* children[26]` / `array<int, 26>` | `children = {}` (dict of char -> node) | average O(1) per step |
| `children[c - 'a']` | `children.get(ch)` / `ch in children` | average O(1) |
| `if (!child) child = new TrieNode();` | `if ch not in children: children[ch] = TrieNode()` | average O(1) |
| `bool isEnd` | `end = True`, or store the word itself at the node | O(1) |
| `delete` a pruned subtree | `del parent.children[ch]` (GC frees it) | average O(1) |
| `unordered_map<string, int>` counts | `dict` / `Counter` | average O(1) per op, O(L) to hash |
| `partial_sort` / `priority_queue` for top k | `sorted(xs, key=...)[:k]` / `heapq.nsmallest(k, xs, key=...)` | O(m log m) / O(m log k) |
| comparator "count desc, text asc" | key `(-count, text)` | O(1) per compare + string compare |
| `s.compare(0, p.size(), p) == 0` | `s.startswith(p)` | O(len(p)) |
| `s.substr(0, i)` | `s[:i]` (a new string) | O(i) |
| `std::string` push_back while typing | `list` of chars, `"".join(...)` at the end | O(1) per char, O(L) join |

## Complexity

| Operation | Time | Space |
|---|---|---|
| insert / search / starts_with | O(L) | O(L) new nodes per insert, worst case |
| Build from words | O(total chars) | O(total chars) nodes |
| Word Search II | O(R * C * 4 * 3^(L-1)) worst case | O(total chars) + O(L) stack |
| Autocomplete keystroke (cached `hot`) | O(1) step + O(K) copy | |
| Autocomplete `#` (record a sentence) | O(L * K log K) | O(L) new nodes; each node keeps at most K references |
| Autocomplete without a cache | O(subtree size + m log K) per keystroke | m = matches |
| Sorted list + `bisect` instead of a trie | O(L log N + matches * L) per query, O(N) per insert | O(total chars) |

## Gotchas for C++ developers

- `search` must check the end flag; `starts_with` must not. `"app"` is a prefix of `"apple"`, not a word.
- `node.children.setdefault(ch, TrieNode())` builds a new TrieNode on **every** call, even on a hit.
- Arguments are evaluated before the call. Use `if ch not in children` when node creation is costly.
- A nested `defaultdict` trie inserts nodes on every lookup miss. Use `in` / `.get` when searching.
- A dict-based node with one child costs roughly 250 bytes in CPython. The children dict dominates.
- `__slots__` saves only a little here. For 10^6 nodes, plan for hundreds of MB.
- Word Search II: build one trie for all words. One DFS per word repeats work.
- Set `node.word = None` after a hit, or the same word is reported again from another path.
- Restore the cell after the DFS (`board[r][c] = ch`), even when the word was found.
- Prune leaves that hold no word. Without pruning, large boards with long shared prefixes time out.
- Top-k order "count desc, text asc": negate the count, keep the text as is. A string cannot be negated.
- Strings compare by code point: `" " < "A" < "a"`. Same as `std::string` with ASCII chars.
- Index arrays (`ord(ch) - ord("a")`) work in Python too, but the dict version handles any alphabet.
- Common follow-ups: **deletion** (unset the end flag, drop nodes left empty) and **radix tree** compression.

## In the Google bank

- **Basic trie:** G-208 Implement Trie (Prefix Tree).
- **Trie + grid DFS with pruning:** G-212 Word Search II.
- **Trie with counts, top 3:** G-642 Design Search Autocomplete System.
- **Prefix checks over a word set:** G-139 Word Break (a trie walk from each index is an alternative to slicing).
