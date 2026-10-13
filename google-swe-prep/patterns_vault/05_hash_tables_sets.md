# Hash maps and sets: count things, find pairs

> In one line: for "how many times?" or "have I seen it before?", keep a dict or a set. Each check is then instant, with no loop.

## When to use it
- The problem says "count", "how many times", "most common" or "anagram" (same letters in a different order).
- It asks for **two numbers that add up to a target**, and the list is not sorted.
- It says "duplicate", "unique", "seen before" or "visited".
- It says "group" the items that share something, like the same letters.
- Your first idea is a loop inside a loop that asks "is x somewhere in the list?". A set replaces the inner loop.

## The idea in plain words
A **hash map** stores **key → value** pairs. In Python it is the `dict`, like `unordered_map` in C++.
It turns each key into a number called a **hash**, and that number says where the pair is stored.
So finding a key is **O(1) on average** (constant time): the same speed for 10 keys or 10 million keys.
A search through a list is **O(N)**: the time grows with N, the number of items.
A **set** is the same idea with keys only. It answers one question fast: "is it there?".

Four patterns cover most interview questions:

1. **Frequency map** (frequency = how many times an item appears). Key = the item, value = its count. `Counter` (from `collections`) builds one in a single line.
2. **Pair sum** (the "two-sum" problem). For each number `x`, the partner you need is `target - x`, called the **complement**.
   Walk the list once. Ask the dict "did I already see the complement?" first, and only then store `x`.
3. **Seen set.** Add each item to a set. If an item is already in the set, it is a duplicate.
4. **Group by a key.** Make a key that is equal for every item in one group, for example the sorted letters of a word.

Keys must be **hashable** (Python can make a hash from them). Values that cannot change work: `int`, `str`, `tuple`.
A `list` can change, so it cannot be a key.

## Python template

### Frequency map with Counter

`Counter` is a dict that counts. A missing item counts as 0, and no error is raised.

```python
from collections import Counter

words = ["cat", "dog", "cat", "cat"]
count = Counter(words)
print(count["cat"])  # 3
print(count["fish"])  # 0
# most_common(k): the k most common items, as (item, count) pairs
print(count.most_common(1))  # [('cat', 3)]

# The same count with a plain dict: get(key, 0) gives 0 for a new key
by_hand = {}
for w in words:
    by_hand[w] = by_hand.get(w, 0) + 1
print(by_hand)  # {'cat': 3, 'dog': 1}

# Anagram check: the same letters with the same counts
print(Counter("listen") == Counter("silent"))  # True
```

A worked example: the index of the first letter that appears only once.

```python
from collections import Counter

def first_unique(s):
    count = Counter(s)
    # enumerate gives the index i and the letter ch together
    for i, ch in enumerate(s):
        if count[ch] == 1:
            return i
    return -1

print(first_unique("leetcode"))  # 0
print(first_unique("loveleetcode"))  # 2
print(first_unique("aabb"))  # -1
```

### Pair sum (two-sum): look up the complement

Return the two indexes whose values add up to `target`. The dict maps value → index.

```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i  # store AFTER the check
    return []

print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
print(two_sum([3, 2, 4], 6))  # [1, 2]
print(two_sum([1, 2], 7))  # []
```

Why check first? In `[3, 2, 4]` with target 6, the first 3 needs another 3.
If we stored 3 before the check, it would pair with itself and return `[0, 0]`.

To **count** pairs instead, the dict holds counts: value → how many times seen so far.

```python
from collections import Counter

def count_pairs(nums, target):
    seen = Counter()
    pairs = 0
    for x in nums:
        pairs += seen[target - x]  # each earlier partner makes one pair
        seen[x] += 1
    return pairs

# the 3 pairs: (1, 5), (1, 5) and (7, -1)
print(count_pairs([1, 5, 7, -1, 5], 6))  # 3
```

### Seen set: duplicates and fast "is it there?"

```python
def has_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False

print(has_duplicate([1, 2, 3, 1]))  # True
print(has_duplicate([1, 2, 3]))  # False

# Turn a list into a set ONCE, then every "in" check is O(1)
word_set = set(["hot", "dot", "dog"])
print("dog" in word_set)  # True
```

### Group by a key, and tuples as keys

`defaultdict(list)` gives a new empty list the first time a key is used.

```python
from collections import defaultdict

groups = defaultdict(list)
for w in ["eat", "tea", "tan", "ate", "nat"]:
    # sorted("tea") is ['a', 'e', 't']; "".join glues it into "aet"
    key = "".join(sorted(w))
    groups[key].append(w)
print(list(groups.values()))  # [['eat', 'tea', 'ate'], ['tan', 'nat']]

# A grid cell (row, col) is a tuple: it can go in a set
visited = {(0, 1)}
print((0, 1) in visited)  # True
try:
    visited.add([2, 3])  # a list can change, so it is not hashable
except TypeError:
    print("TypeError")  # TypeError
```

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `unordered_map<string, int> m;` | `m = {}` | a dict keeps the order you added keys |
| `m[k]++;` (adds 0 first) | `count[k] += 1` on a `Counter` | on a plain dict a new `k` raises KeyError (key not found) |
| `m.count(k)` or `m.find(k) != m.end()` | `k in m` | gives `True` or `False` |
| `m.erase(k);` | `del m[k]` | KeyError if `k` is missing |
| `for (auto& [k, v] : m)` | `for k, v in m.items():` | |
| `unordered_set<int> s; s.insert(x);` | `s = set()` then `s.add(x)` | `{}` is an empty dict, not a set |
| `unordered_map<string, vector<string>>` | `defaultdict(list)` | the empty list is made for you |
| `unordered_set<pair<int,int>, MyHash>` | a set of tuples `(r, c)` | no hash function to write |
| `map<int, int>` (keys kept sorted) | a dict, then `sorted(m)` | Python has no built-in sorted map |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| `d[k]`, `d[k] = v`, `k in d` | O(1) on average | jumps straight to the key; rare bad cases are slower |
| `s.add(x)`, `x in s` | O(1) on average | the same trick as dict |
| `x in nums` when `nums` is a list | O(N) | looks at the items one by one |
| `Counter(nums)`, `set(nums)` | O(N) | looks at each item once |
| Two-sum with a dict | O(N) time, O(N) memory | one pass; the dict may hold every number |
| Two-sum with two loops | O(N²) | every pair: about 500,000 checks for 1,000 numbers |
| Key `"".join(sorted(w))` | O(L log L) | L is the length of the word: sorting its letters |

## Common mistakes
- Mistake: `count[x] += 1` on a plain dict when `x` is new raises KeyError. Fix: use `Counter()`, `defaultdict(int)` or `count.get(x, 0) + 1`.
- Mistake: in two-sum, storing `x` before the check, so a number pairs with itself. Fix: check `target - x` first, then store `x`.
- Mistake: `if seen.get(need):` to test for a key. Index 0 counts as False in an `if`, so a real match is missed. Fix: `if need in seen:`.
- Mistake: `seen = {}` meant as an empty set. It is an empty dict. Fix: `seen = set()`.
- Mistake: a list as a dict key or set item gives TypeError. Fix: convert it to a tuple, like `tuple(cell)` or `(r, c)`.
- Mistake: `if x in nums` inside a loop, where `nums` is a list. That is O(N²) in total. Fix: build `num_set = set(nums)` once, before the loop.
- Mistake: expecting a set to keep order. Fix: return `sorted(s)` when the order matters.

## Practice in the app
- **G-347 Top K Frequent Elements**: `Counter(nums)` counts; then `most_common(k)` or a heap picks the top k.
- **G-621 Task Scheduler**: `Counter(tasks)`; the most frequent task decides how long the schedule is.
- **G-139 Word Break**: put the words in a set, so "is this piece a word?" is O(1).
- **G-127 Word Ladder**: the word list as a set, plus a `visited` set for the search.
- **G-815 Bus Routes**: `defaultdict(list)` maps each stop to the buses that stop there.
- **G-133 Clone Graph**: a dict maps each old node to its copy, so no node is copied twice.
- **G-105 Construct Binary Tree from Preorder and Inorder Traversal**: a dict maps each value to its index in the in-order list.
- **G-863 All Nodes Distance K in Binary Tree**: a dict maps each node to its parent, then search outward.
