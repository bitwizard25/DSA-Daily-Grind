# Tries (prefix trees): words that start the same way

> In one line: a trie stores words letter by letter, so all words with the same start share one path from the top.

## When to use it
- "prefix", "starts with", "autocomplete" or "type-ahead" (suggest words while the user types).
- Many questions of the same kind: "does any word start with `ap`?"
- Many words to find at once, for example in a grid of letters (Word Search II).
- One big word list, and many lookups by the first letters of a word.

## The idea in plain words
A **prefix** is the start of a word: `"a"`, `"ap"` and `"app"` are prefixes of `"apple"`.

A **trie** (say "try"), also called a **prefix tree**, is a tree of letters.
- A **node** is one box in the tree. The top node is the **root**: it stands for the empty prefix `""`.
- Each node has **children**: one child for each letter that can come next.
  In Python, `children` is a **dict** (a key → value map, like C++ `unordered_map`), for example `{"p": node}`.
- A path from the root spells a prefix. Words with the same start share the same first nodes.
- Each node has an **end flag** (`end = True`). It means "a whole word ends here", not only a prefix.

The words `"app"`, `"apple"` and `"bat"` give this trie. A star `*` marks a node whose end flag is True:

```
root
├── a ── p ── p* ── l ── e*
└── b ── a ── t*
```

All three basic operations walk down one letter at a time:
1. `insert(word)`: for each letter, go to that child. If it is missing, create it first. At the last letter, set `end = True`.
2. `search(word)`: for each letter, go to that child. If it is missing, the answer is False. At the end, return the end flag.
3. `starts_with(prefix)`: the same walk as `search`. If the walk finishes, the answer is True. The end flag does not matter.

Each operation costs **O(L)** (big-O: how the work grows). L is the word length. The number of stored words does not matter.

**Autocomplete**: walk down to the node of the prefix. Then visit every node below it and collect the words.
To visit every node, use **DFS** (depth-first search): go deep down one branch, come back, then try the next branch.
A function that calls itself, like this DFS, is called **recursive**.

**Word Search II** (the idea): find many words in a grid of letters.
1. Put all the words in one trie.
2. Start a path from each cell. Step to a neighbour cell only if the trie has a child for its letter.
3. When you reach a node that holds a word, record the word.
A path stops as soon as no word starts with it. Cutting off useless paths early is called **pruning**.

## Python template

### Trie: insert, search, starts_with

A **class** is like a C++ `struct` with functions inside. `__init__` is the constructor. `self` is like `this`.
`None` means "no node here", like `nullptr`. Test it with `is None` or `is not None`.

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # letter -> TrieNode
        self.end = False    # True when a whole word ends here

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.end = True

    def walk(self, text):
        # the node for this prefix, or None if some letter is missing
        node = self.root
        for ch in text:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word):
        node = self.walk(word)
        return node is not None and node.end

    def starts_with(self, prefix):
        return self.walk(prefix) is not None

trie = Trie()
trie.insert("apple")
print(trie.search("apple"))  # True
# "app" is only a prefix so far, not a word
print(trie.search("app"))  # False
print(trie.starts_with("app"))  # True
trie.insert("app")
print(trie.search("app"))  # True
print(trie.starts_with("b"))  # False
```

### Autocomplete: every word that starts with a prefix

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

def build(words):
    root = TrieNode()
    for word in words:
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.end = True
    return root

def collect(node, path, result):
    # DFS: path is the text from the root down to this node
    if node.end:
        result.append(path)
    # sorted(a dict) gives its keys A to Z, so the words come out sorted
    for ch in sorted(node.children):
        collect(node.children[ch], path + ch, result)

def suggest(root, prefix):
    node = root
    for ch in prefix:
        if ch not in node.children:
            return []
        node = node.children[ch]
    result = []
    collect(node, prefix, result)
    return result

root = build(["car", "card", "care", "cat", "dog"])
print(suggest(root, "car"))  # ['car', 'card', 'care']
print(suggest(root, "ca"))  # ['car', 'card', 'care', 'cat']
print(suggest(root, "x"))  # []
```

### Word Search II: one trie for all the words

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None  # the full word that ends here

def find_words(board, words):
    root = TrieNode()
    for word in words:
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.word = word
    rows, cols = len(board), len(board[0])
    found = []
    # dfs sits inside find_words, so it can use board, rows, cols and found
    def dfs(r, c, node):
        ch = board[r][c]
        if ch not in node.children:
            return  # pruning: no word continues with this letter
        node = node.children[ch]
        if node.word is not None:
            found.append(node.word)
            node.word = None  # record each word only once
        board[r][c] = "#"  # mark as used: "#" is in no word, so paths stop here
        # try the 4 neighbour cells: down, up, right, left
        for nr, nc in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs(nr, nc, node)
        board[r][c] = ch  # give the cell back for other paths

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return found

# list("oaan") gives ['o', 'a', 'a', 'n']: one row of letters we can change
board = [list("oaan"), list("etae"), list("ihkr"), list("iflv")]
print(sorted(find_words(board, ["oath", "pea", "eat", "rain"])))  # ['eat', 'oath']
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `struct Node { Node* kids[26]; bool end; };` | `class TrieNode:` with `self.children = {}` | a dict, not a fixed array of 26 |
| `Node* root = new Node();` | `root = TrieNode()` | no `new`, no `delete` |
| `this->end = true;` | `self.end = True` | `self` is like `this` |
| `if (!node->kids[c - 'a'])` | `if ch not in node.children:` | the letter is the key, no `- 'a'` |
| `node = node->kids[c - 'a'];` | `node = node.children[ch]` | |
| `if (node == nullptr)` | `if node is None:` | `None` is like `nullptr` |
| `for (char c : word)` | `for ch in word:` | a loop over a string gives its letters |
| `s.substr(0, p.size()) == p` | `s.startswith(p)` | checks one word only |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| `insert`, `search`, `starts_with` | O(L) | L = letters in the word; the number of stored words does not matter |
| Build a trie from all words | O(total letters) | each letter of each word is handled once |
| Memory | O(total letters) | at most one node per letter; shared starts save nodes |
| Autocomplete | O(P + K × L) | P = prefix letters to walk down; then K nodes below it, each `path + ch` copies up to L letters |
| Prefix check without a trie | O(N × P) | test `startswith` on all N words, P letters each |
| Word Search II | O(R × C × 3^L) worst case | R × C start cells; L = longest word; each step has at most 3 new cells; pruning cuts most paths |

## Common mistakes
- Mistake: `search` returns True for `"app"` when only `"apple"` was added. Fix: return the end flag, `node.end`.
- Mistake: `starts_with` checks the end flag. Fix: any node at the end of the walk is enough.
- Mistake: `node.children[ch] = TrieNode()` without a check. It erases the words below that letter. Fix: create it only `if ch not in node.children`.
- Mistake: one DFS on the grid for each word in Word Search II. It is far too slow. Fix: one trie and one DFS for all words.
- Mistake: forgetting to give the cell back (`board[r][c] = ch`). Later paths cannot use it. Fix: restore it after the loop.
- Mistake: the same word appears twice in the answer. Fix: set `node.word = None` after you record it.

## Practice in the app
- **G-208 Implement Trie (Prefix Tree)**: the first template above.
- **G-212 Word Search II**: the third template. For big boards, also delete a child node once it has no children and no word.
- **G-642 Design Search Autocomplete System**: a trie of sentences, each with a count. Sort by `(-count, sentence)`: biggest count first, ties A to Z. Show the first 3.
- **G-139 Word Break**: mainly dynamic programming (note 10). A trie helps find the words that start at each index.
