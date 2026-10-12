# Tries

**Cue:** prefixes, autocomplete, many words searched together.

```
class Node:
    def __init__(self): self.children = {}; self.end = False

def insert(root, word):
    node = root
    for ch in word:
        node = node.children.setdefault(ch, Node())
    node.end = True
```

- Insert / search / startsWith: O(L).
- Word Search II: build a trie of words, DFS the board, prune finished branches.
- Follow-up: radix tree compression, deletion.
