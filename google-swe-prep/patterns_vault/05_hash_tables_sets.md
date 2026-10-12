# Hash Tables and Sets

**Use when:**
- **Count**, frequency, most common, anagram, duplicate, first unique character.
- **Two values sum to a target** on unsorted input, or "have I seen this before?".
- **Subarray sum equals K**, sum divisible by K, equal number of 0s and 1s (prefix sum + hash map).
- **Group by** a shared shape (anagrams, shifted strings), or O(1) membership instead of a list scan.

## Python template(s)

### Frequency maps: `Counter` and `defaultdict`

- `Counter` reads a missing key as **0 without inserting** it. `defaultdict(int)` **inserts** on read.
- A plain `dict` raises `KeyError` on a missing key: use `d.get(k, 0)`.

```python
from collections import Counter, defaultdict

c = Counter("mississippi")
assert c["s"] == 4 and c["i"] == 4 and c["p"] == 2 and c["m"] == 1
assert c["z"] == 0 and "z" not in c          # read does not insert
assert c.most_common(2) == [("i", 4), ("s", 4)]   # ties: first-seen order
assert Counter("listen") == Counter("silent")     # anagram check, O(N)

a, b = Counter("aab"), Counter("abc")
assert a - b == Counter({"a": 1})            # drops counts <= 0
assert a + b == Counter({"a": 3, "b": 2, "c": 1})
assert a & b == Counter({"a": 1, "b": 1})    # min of each count
assert a | b == Counter({"a": 2, "b": 1, "c": 1})   # max of each count

d = defaultdict(int)
for w in "the cat the hat".split():
    d[w] += 1
assert d == {"the": 2, "cat": 1, "hat": 1}
_ = d["dog"]                                 # a READ inserts dog -> 0
assert "dog" in d and len(d) == 4

counts = {}
for ch in "aab":
    counts[ch] = counts.get(ch, 0) + 1       # plain dict
assert counts == {"a": 2, "b": 1}
try:
    counts["z"]
except KeyError:
    pass
else:
    raise AssertionError("plain dict must raise KeyError")


def first_uniq_char(s):                      # First Unique Character in a String
    c = Counter(s)
    for i, ch in enumerate(s):
        if c[ch] == 1:
            return i
    return -1

assert first_uniq_char("leetcode") == 0
assert first_uniq_char("loveleetcode") == 2
assert first_uniq_char("aabb") == -1


def top_k_frequent(nums, k):                 # G-347: bucket sort, O(N)
    freq = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]   # index = frequency
    for x, f in freq.items():
        buckets[f].append(x)
    out = []
    for f in range(len(nums), 0, -1):
        for x in buckets[f]:
            out.append(x)
            if len(out) == k:
                return out
    return out

assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
assert [x for x, _ in Counter([1, 1, 1, 2, 2, 3]).most_common(2)] == [1, 2]   # O(D log k)
```

### Pair sum: complement lookup

- One pass. Look up `target - x` **before** inserting `x`, so an element never pairs with itself.
- Map value -> index for indexes; value -> count to count pairs.

```python
from collections import Counter


def two_sum(nums, target):                   # Two Sum: indexes of the pair
    seen = {}                                # value -> index
    for i, x in enumerate(nums):
        j = seen.get(target - x)
        if j is not None:                    # not `if j:` (index 0 is falsy)
            return [j, i]
        seen[x] = i                          # insert AFTER the lookup
    return []

assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 3], 6) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]       # 3 must not pair with itself
assert two_sum([1, 2], 7) == []


def count_pairs(nums, target):               # pairs i < j with nums[i] + nums[j] == target
    seen = Counter()
    total = 0
    for x in nums:
        total += seen[target - x]
        seen[x] += 1
    return total

assert count_pairs([1, 5, 7, -1, 5], 6) == 3     # (1, 5), (1, 5), (7, -1)
assert count_pairs([2, 2, 2], 4) == 3
```

### Prefix sum + hash map

- `P[j] - P[i] == k` means `nums[i:j]` sums to `k`. Count earlier prefixes equal to `run - k`.
- Seed with prefix `0` (the empty prefix). Works with **negatives and zeros**; a sliding window does not.

```python
from collections import Counter


def subarray_sum(nums, k):                   # Subarray Sum Equals K
    seen = Counter({0: 1})                   # empty prefix: subarrays starting at 0
    run = total = 0
    for x in nums:
        run += x
        total += seen[run - k]               # Counter read does not insert
        seen[run] += 1                       # insert AFTER the lookup
    return total

assert subarray_sum([1, 1, 1], 2) == 2
assert subarray_sum([1, 2, 3], 3) == 2
assert subarray_sum([1, -1, 0], 0) == 3      # [1, -1], [0], [1, -1, 0]
assert subarray_sum([3], 0) == 0             # insert-before-lookup would give 1


def max_len_sum_k(nums, k):                  # longest subarray with sum k
    first = {0: -1}                          # prefix sum -> earliest index
    run = best = 0
    for i, x in enumerate(nums):
        run += x
        if run - k in first:
            best = max(best, i - first[run - k])
        first.setdefault(run, i)             # keep only the earliest index
    return best

assert max_len_sum_k([1, -1, 5, -2, 3], 3) == 4
assert max_len_sum_k([-2, -1, 2, 1], 1) == 2


def find_max_length(bits):                   # Contiguous Array: equal 0s and 1s
    return max_len_sum_k([1 if b else -1 for b in bits], 0)

assert find_max_length([0, 1, 0]) == 2
assert find_max_length([0, 1, 1, 1, 0, 0]) == 6


def subarrays_div_by_k(nums, k):             # Subarray Sums Divisible by K
    seen = Counter({0: 1})
    run = total = 0
    for x in nums:
        run = (run + x) % k                  # in [0, k) for k > 0: no C++ fix-up
        total += seen[run]
        seen[run] += 1
    return total

assert subarrays_div_by_k([4, 5, 0, -2, -3, 1], 5) == 7
assert subarrays_div_by_k([5], 9) == 0
```

### Grouping by canonical key

- Map each item to a **hashable key** that is equal for every member of its group.
- Common keys: sorted string, 26-count `tuple`, normalised difference pattern.

```python
from collections import defaultdict


def group_anagrams(words):                   # Group Anagrams
    groups = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)     # O(L log L) key
    return list(groups.values())

out = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
assert sorted(map(sorted, out)) == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]


def count_key(w):                            # O(L) key for lowercase words
    cnt = [0] * 26
    for ch in w:
        cnt[ord(ch) - ord("a")] += 1
    return tuple(cnt)                        # list is unhashable; tuple is fine

assert count_key("eat") == count_key("tea") != count_key("tan")


def group_shifted(strings):                  # Group Shifted Strings: "abc" ~ "bcd" ~ "xyz"
    groups = defaultdict(list)
    for s in strings:
        key = tuple((ord(c) - ord(s[0])) % 26 for c in s)
        groups[key].append(s)
    return sorted(sorted(g) for g in groups.values())

assert group_shifted(["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]) == [
    ["a", "z"], ["abc", "bcd", "xyz"], ["acef"], ["az", "ba"]]
```

### Set membership

- `x in s` is **O(1) average** for `set` and `dict`, **O(N)** for `list`. Convert once: `words = set(word_list)`.
- Set iteration order is **arbitrary**. Dicts keep insertion order (guaranteed since 3.7).

```python
def longest_consecutive(nums):               # Longest Consecutive Sequence, O(N) average
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 not in s:                   # start only at the first value of a run
            y = x
            while y + 1 in s:
                y += 1
            best = max(best, y - x + 1)
    return best

assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
assert longest_consecutive([]) == 0


def contains_duplicate(nums):
    return len(set(nums)) != len(nums)

assert contains_duplicate([1, 2, 3, 1]) and not contains_duplicate([1, 2, 3])

a, b = {1, 2, 3}, {2, 3, 4}
assert a & b == {2, 3} and a | b == {1, 2, 3, 4}
assert a - b == {1} and a ^ b == {1, 4}
assert {1, 2} <= a                           # subset test

seen = set()
assert seen.add(5) is None and 5 in seen     # add returns None
seen.discard(9)                              # no error when missing
try:
    seen.remove(9)
except KeyError:
    pass
else:
    raise AssertionError("remove must raise KeyError when missing")

assert type({}) is dict and type(set()) is set   # {} is an empty DICT

d = {"a": 1, "b": 2, "c": 3}
for key in list(d):                          # copy the keys before deleting
    if d[key] % 2:
        del d[key]
assert d == {"b": 2}
try:
    for key in d:
        d["new"] = 0                         # size change during iteration
except RuntimeError:
    pass
else:
    raise AssertionError("dict size change during iteration must raise")
```

### Hashing tuples and custom keys

- Keys must be **hashable**: `int`, `str`, `tuple` of hashables, `frozenset`. `list`, `set` and `dict` are not.
- Grid cells and multi-part states are plain tuples: `(r, c)`, `(node, mask)`. No `pair_hash` needed.

```python
from dataclasses import dataclass
from functools import cache

cells = {(0, 0), (1, 2)}
assert (1, 2) in cells

try:
    {[1, 2]: "x"}
except TypeError:
    pass
else:
    raise AssertionError("list keys must fail")

try:
    hash((1, [2]))                           # a tuple holding a list is unhashable
except TypeError:
    pass
else:
    raise AssertionError("tuple containing a list must fail")

frozen = frozenset({1, 2})                   # hashable set: usable as a key
assert {frozen: "ok"}[frozenset({2, 1})] == "ok"

memo = {}
def paths(r, c):                             # memo keyed by a tuple
    if r == 0 or c == 0:
        return 1
    if (r, c) not in memo:
        memo[(r, c)] = paths(r - 1, c) + paths(r, c - 1)
    return memo[(r, c)]

assert paths(2, 2) == 6                      # G-62 Unique Paths on a 3 x 3 grid


@cache                                       # same memo, keyed by the argument tuple
def paths2(r, c):
    return 1 if r == 0 or c == 0 else paths2(r - 1, c) + paths2(r, c - 1)

assert paths2(2, 2) == 6


@dataclass(frozen=True)                      # generates __eq__ and __hash__
class Point:
    x: int
    y: int

assert len({Point(1, 2), Point(1, 2), Point(2, 1)}) == 2


class Node:                                  # no __eq__: hashed by identity
    def __init__(self, val):
        self.val = val

n1, n2 = Node(1), Node(1)
assert len({n1: "a", n2: "b"}) == 2          # G-133: old node -> copy works


class EqOnly:
    def __init__(self, v):
        self.v = v
    def __eq__(self, other):                 # __eq__ without __hash__ ...
        return self.v == other.v

try:
    {EqOnly(1)}
except TypeError:                            # ... makes instances unhashable
    pass
else:
    raise AssertionError("defining __eq__ alone must remove __hash__")

assert hash(1) == hash(1.0) == hash(True)    # 1, 1.0 and True are the same key
assert len({1: "a", 1.0: "b", True: "c"}) == 1
```

### Average vs worst case

- CPython `dict` / `set` use open addressing: lookup, insert, delete are **O(1) average, O(N) worst case**.
- `str` hashes are **randomised per process**. `int` hashes are not: `hash(n) == n` for small non-negative `n`.

```python
import sys

assert hash(42) == 42 and hash(-1) == -2     # -1 is reserved as an error code in C
P = sys.hash_info.modulus                    # 2**61 - 1 on 64-bit builds
assert hash(P) == 0 and hash(2 * P) == 0     # crafted ints all hash to 0


class BadKey:                                # every key collides
    def __init__(self, v):
        self.v = v
    def __hash__(self):
        return 0
    def __eq__(self, other):
        return isinstance(other, BadKey) and self.v == other.v

d = {BadKey(i): i for i in range(300)}       # each insert re-walks the same probe sequence: O(N^2) total
assert len(d) == 300 and d[BadKey(299)] == 299
```

C++ side by side:

```cpp
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct PairHash {                                // std::pair has no std::hash
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::uint64_t key = (std::uint64_t(std::uint32_t(p.first)) << 32) | std::uint32_t(p.second);
        return std::hash<std::uint64_t>()(key);
    }
};

int main() {
    // operator[] inserts a default value on read, like defaultdict.
    std::unordered_map<std::string, int> cnt;
    std::vector<std::string> words{"the", "cat", "the"};
    for (const auto& w : words) ++cnt[w];
    assert(cnt["the"] == 2 && cnt.size() == 2);
    assert(cnt.count("dog") == 0 && cnt.size() == 2);   // count() does not insert
    int dog = cnt["dog"];                                // inserts "dog" -> 0
    assert(dog == 0 && cnt.size() == 3);

    // Two Sum with a complement lookup.
    std::vector<int> nums{2, 7, 11, 15};
    std::unordered_map<int, int> seen;
    int a = -1, b = -1;
    for (int i = 0; i < (int)nums.size(); ++i) {
        auto it = seen.find(9 - nums[i]);
        if (it != seen.end()) {
            a = it->second;
            b = i;
            break;
        }
        seen[nums[i]] = i;
    }
    assert(a == 0 && b == 1);

    // Pairs need a custom hasher in unordered containers; Python tuples do not.
    std::unordered_set<std::pair<int, int>, PairHash> cells{{0, 0}, {1, 2}};
    assert(cells.count({1, 2}) == 1);
    std::set<std::pair<int, int>> ordered{{1, 2}, {0, 0}};   // tree: O(log N)
    assert(ordered.begin()->first == 0);
    return 0;
}
```

## C++ -> Python mapping

| C++ STL / idiom | Python equivalent | Time complexity |
|---|---|---|
| `unordered_map<K, V> m;` | `m = {}` | O(1) |
| `++m[k]` (inserts 0 first) | `cnt[k] += 1` on a `Counter` / `defaultdict(int)` | O(1) average |
| `m[k] = m[k] + 1` on a plain map | `m[k] = m.get(k, 0) + 1` | O(1) average |
| `m.count(k)` / `m.find(k) != m.end()` | `k in m` | O(1) average |
| `m.at(k)` (throws `out_of_range`) | `m[k]` (raises `KeyError`) | O(1) average |
| read `m[k]` without inserting | `m.get(k, default)` | O(1) average |
| `m.erase(k)` | `del m[k]` (raises if missing) or `m.pop(k, None)` | O(1) average |
| `for (auto& [k, v] : m)` | `for k, v in m.items():` | O(N) |
| `unordered_set<int> s; s.insert(x);` | `s = set()`, `s.add(x)` | O(1) average |
| `s.insert(x).second` (was it new?) | `if x not in s: s.add(x)` | O(1) average |
| `unordered_set<pair<int, int>, PairHash>` | `set` of tuples `(r, c)` | O(1) average, hashing is O(len) |
| `std::map<K, V>` (ordered tree) | no built-in; `dict` + `sorted(m)` when order is needed | `sorted` is O(N log N) |
| `std::multiset<T>` used for counts | `Counter` | O(1) average |
| `std::set_intersection` on sorted ranges | `a & b` on sets | O(min(len(a), len(b))) average |
| `std::hash<std::string>` (same every run in libstdc++) | `hash(s)`, randomised per process | O(L) |
| `struct` key with `operator==` and a hasher | `@dataclass(frozen=True)` or a `tuple` | O(fields) |

## Complexity

| Operation | Time | Space |
|---|---|---|
| `d[k]`, `d[k] = v`, `k in d`, `del d[k]` | O(1) average, O(N) worst | O(N) for the table |
| `x in s`, `s.add(x)`, `s.discard(x)` | O(1) average, O(N) worst | O(N) for the table |
| `x in lst` | O(N) | O(1) |
| `Counter(iterable)`, `set(iterable)` | O(N) | O(distinct) |
| `c.most_common(k)` with D distinct keys | O(D log k); without `k`: O(D log D) | O(D) |
| `Counter == Counter`, `dict == dict` | O(D) | O(1) |
| `a & b` on sets | O(min(len(a), len(b))) average | O(result) |
| `a - b` on sets | O(len(a)) average | O(result) |
| Hash a `str` / `tuple` key of length L | O(L) | O(1) |
| Key `"".join(sorted(w))` | O(L log L) | O(L) |
| Key: 26-count `tuple` | O(L + 26) | O(26) |
| Prefix sum + hash map | O(N) average | O(N) |
| Iterate a dict or set | O(N) | O(1) |

## Gotchas for C++ developers

- Reading `d[k]` on a **`defaultdict` inserts** `k`. `Counter` reads do not insert. Plain `dict` raises `KeyError`.
- `len(counter)` counts keys whose value dropped to 0. `del` them, or `+counter` drops counts <= 0.
- `if seen.get(x):` is wrong when the value can be **0** (index 0). Use `is not None` or `x in seen`.
- `list`, `set` and `dict` are **unhashable**. Use `tuple(...)` or `frozenset(...)` as keys.
- Defining `__eq__` without `__hash__` makes a class **unhashable**. `@dataclass(frozen=True)` writes both.
- `1`, `1.0` and `True` are the **same key**: `{1: "a", True: "b"}` has one entry.
- Changing a dict's size during iteration raises `RuntimeError`. Iterate over `list(d)` to delete.
- `set` order is arbitrary. Sort before returning when the order matters.
- `{}` is an empty **dict**; the empty set is `set()`.
- `s.add(x)` returns `None`, not `pair<iterator, bool>`. Test `x in s` first.
- **Worst case is still O(N)** per operation. Crafted colliding ints (multiples of `2**61 - 1`) degrade a dict.
- Prefix sum: insert the current prefix **after** the lookup, or `k == 0` counts empty subarrays.
- `%` is already non-negative for `k > 0`: no `((run % k) + k) % k` fix-up.
- No ordered map in the standard library. `sortedcontainers` is third-party: ask before relying on it.

## In the Google bank

- **G-347 Top K Frequent Elements**: `Counter` + bucket sort (O(N)), or `most_common(k)`.
- **G-621 Task Scheduler**: `Counter(tasks)`; answer is `max(len(tasks), (max_f - 1) * (n + 1) + ties_at_max_f)`.
- **G-133 Clone Graph**: `old node -> copy` dict; nodes hash by identity.
- **G-863 All Nodes Distance K in Binary Tree**: `child -> parent` dict keyed by node, then BFS.
- **G-105 Construct Binary Tree from Preorder and Inorder Traversal**: `value -> inorder index` dict for O(1) splits.
- **G-721 Accounts Merge**: `email -> account id` dict feeding union-find.
- **G-127 Word Ladder**: word `set`; wildcard buckets `"h*t" -> [words]` group by canonical key.
- **G-815 Bus Routes**: `stop -> routes` index built with `defaultdict(list)`.
- **G-139 Word Break**: `set(wordDict)` for membership; each `s[j:i]` slice and hash is O(i - j).
- **G-987 Vertical Order Traversal of a Binary Tree**: `defaultdict(list)` keyed by column, then sort.
- **G-208 Implement Trie (Prefix Tree)**: children stored in a dict per node.
- **G-642 Design Search Autocomplete System**: `sentence -> count` dict; rank by `(-count, sentence)`.
