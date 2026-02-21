# 🧠 30-Day DSA Interview Prep Roadmap — C++ Edition

> **Goal:** Pattern recognition over memorization. Build the mental muscle to crack top-tier MNC and FAANG interviews.

---

## 📋 Global Rules

| Rule | Details |
|------|---------|
| 🗣️ Talk out loud | Simulate a real interview. Say "I'm thinking..." if you pause. |
| ⏱️ Complexity first | Define Time & Space Complexity **before** writing a single line of code. |
| 🔄 Stuck > 45 min | Study the optimal approach, close it, then code from memory. |
| 📝 Pattern notes | After every problem: write a 2-sentence "pattern note" in your own words. |
| 🛠️ Use STL | Prefer `vector`, `unordered_map`, `priority_queue` — know their internals. |
| ⚠️ Edge cases | Always handle: empty input, single element, `INT_MIN`/`INT_MAX` overflow. |
| 📄 Dry-run | Trace through a small example on paper before submitting. |

---

# WEEK 1: Primitives, Pointers & Windows
> *Focus: Iterate efficiently without nested loops. Master O(N) single-pass techniques.*

---

## Day 1 — Array Basics & Prefix Sums

> 💡 **Core Insight:** Pre-compute running totals so any range-sum query becomes O(1) instead of O(N).

### Concepts to Master
- **Memory layout:** Arrays are contiguous blocks — cache-friendly. Understand pointer arithmetic.
- **Prefix Sum formula:** `prefix[i] = prefix[i-1] + arr[i-1]` with `prefix[0] = 0` (1-indexed trick).
- **Range sum in O(1):** `sum(L, R) = prefix[R+1] - prefix[L]`
- **2D prefix sums:** Extend to matrices for rectangle-sum queries.

### C++ Pattern
```cpp
vector<int> prefix(n + 1, 0);
for (int i = 0; i < n; i++)
    prefix[i + 1] = prefix[i] + arr[i];

// Sum from index l to r (inclusive)
int rangeSum = prefix[r + 1] - prefix[l];
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Product of Array Except Self | Medium | Prefix + Suffix product arrays | O(N) time, O(1) extra |
| Subarray Sum Equals K | Medium | Prefix sum + `unordered_map` | O(N) time, O(N) space |

### Problem Deep-Dives
- **Product of Array Except Self:** Build left-product array forward, then multiply right-product backward in-place. No division needed. The running right-product is a single variable.
- **Subarray Sum Equals K:** If `prefix[j] - prefix[i] = k`, a subarray exists. Store prefix sums in a hash map as you go. Initialize map with `{0: 1}` to handle subarrays starting at index 0.

### ✅ End-of-Day Checklist
- [ ] Can you write prefix sum without looking at notes?
- [ ] Can you explain why Product of Array doesn't need division?
- [ ] Solved both problems in < 30 min combined?

---

## Day 2–3 — Two Pointers

> 💡 **Core Insight:** Replace O(N²) brute force pairs with a single O(N) pass using two indices that converge or race.

### Two Pointer Variants
- **Converging (Opposite ends):** Start at both ends, move inward based on a condition. Works on sorted arrays.
- **Fast & Slow (Floyd's):** Two pointers at different speeds. Detects cycles, finds midpoints.
- **Same-direction sliding:** Both move forward, tracking a window of valid elements.

### C++ Template — Converging
```cpp
int left = 0, right = n - 1;
while (left < right) {
    if (condition)          { /* use pair */ left++; right--; }
    else if (needBigger)    left++;
    else                    right--;
}
```

### C++ Template — Fast & Slow
```cpp
ListNode *slow = head, *fast = head;
while (fast && fast->next) {
    slow = slow->next;
    fast = fast->next->next;
    if (slow == fast) return true; // cycle
}
return false;
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Two Sum II (Sorted Array) | Medium | Converging pointers | O(N) time, O(1) space |
| Container With Most Water | Medium | Converging pointers, greedy | O(N) time, O(1) space |
| 3Sum | Medium | Sort + converging for each pivot | O(N²) time, O(1) extra |
| Trapping Rain Water | Hard | Left/right max prefix arrays | O(N) time, O(1) space |

### Problem Deep-Dives
- **Two Sum II:** Move `left` up if sum < target, `right` down if sum > target.
- **Container With Most Water:** Always move the pointer with the **shorter** bar — moving the taller bar can only decrease height, never increase it.
- **3Sum:** Sort first. For each index `i`, run converging two-pointer on the rest. Skip duplicates with `while (arr[i] == arr[i-1]) i++`. Also skip duplicate results inside the inner loop.
- **Trapping Rain Water:** `water[i] = min(maxLeft[i], maxRight[i]) - height[i]`. Use two-pointer to reduce to O(1) space by maintaining running max from each side.

### ⚠️ Common Mistakes
- **3Sum:** Forgetting to skip duplicate pivot values AND duplicate results after finding a valid triple.
- **Trapping Rain Water:** Off-by-one on the running max — update max **before** computing water.

---

## Day 4–5 — Sliding Window

> 💡 **Core Insight:** Maintain a window `[left, right]`. Expand `right` to explore, shrink `left` to restore validity. Never restart from scratch.

### Fixed vs Variable Window
- **Fixed window (size k):** Move both pointers together. Add new element, remove old element each step.
- **Variable window:** Expand `right` freely. Shrink `left` until window is valid again. Track max/min valid window size.

### C++ Template — Variable Window
```cpp
unordered_map<char, int> freq;
int left = 0, maxLen = 0;

for (int right = 0; right < n; right++) {
    freq[s[right]]++;

    while (/* window invalid */) {
        freq[s[left]]--;
        if (freq[s[left]] == 0) freq.erase(s[left]);
        left++;
    }

    maxLen = max(maxLen, right - left + 1);
}
```

### C++ Template — Fixed Window (size k)
```cpp
int windowSum = 0;
for (int i = 0; i < k; i++) windowSum += arr[i]; // initial window

int maxSum = windowSum;
for (int i = k; i < n; i++) {
    windowSum += arr[i] - arr[i - k];             // slide
    maxSum = max(maxSum, windowSum);
}
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Maximum Average Subarray I | Easy | Fixed sliding window | O(N) time, O(1) space |
| Longest Substring Without Repeating | Medium | Variable window + hash set | O(N) time, O(26) space |
| Minimum Window Substring | Hard | Variable window + two freq maps | O(N+M) time, O(1) space |

### Problem Deep-Dives
- **Minimum Window Substring:** Track `have` vs `need` counters. When `have == need`, try to shrink `left`. A character is "satisfied" when its frequency in window >= its required frequency.
- **Longest Substring:** Use `unordered_set<char>`. Shrink `left` while the new `right` character is already in the set.

---

## Day 6–7 — Hashing (Maps & Sets)

> 💡 **Core Insight:** Trade O(1) space for O(N) space to get O(1) average lookup. Know when hash collisions cause O(N) worst case.

### C++ Hash Containers
```cpp
unordered_map<int, int> freq;   // O(1) avg insert/lookup
unordered_set<int> seen;        // existence check

freq[key]++;                    // default-initializes to 0
freq.count(key);                // 0 or 1, safer than operator[]

// Custom hash for pairs (encode as single int)
auto encode = [&](int a, int b) { return (long long)a * MAX + b; };
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Two Sum | Easy | Value → index map | O(N) time, O(N) space |
| Group Anagrams | Medium | Sorted string as canonical key | O(N·K·logK) time |
| Longest Consecutive Sequence | Medium | Hash set + sequence start check | O(N) time, O(N) space |

### Problem Deep-Dives
- **Two Sum:** For each element, check if `(target - element)` already exists in the map. Insert **after** checking to avoid using the same element twice.
- **Group Anagrams:** Sort each string (O(K log K)) to get a canonical key. Or use an array of 26 counts encoded as a string like `"1#0#2#..."`.
- **Longest Consecutive Sequence:** Only start counting from sequence beginnings (`num-1` not in set). This ensures O(N) total work across all sequences.

---

# WEEK 2: Structures and Memory
> *Focus: Manage state with LIFO/FIFO and pointer reassignment. Master indirection.*

---

## Day 8–9 — Linked Lists

> 💡 **Core Insight:** You can rewire a list without extra space — just carefully reassign `next` pointers. Always save the next node before overwriting.

### Essential Tricks
- **Dummy head node:** Eliminates edge cases for head operations.
  ```cpp
  ListNode* dummy = new ListNode(0);
  dummy->next = head;
  ```
- **Fast & Slow pointer:** `fast` moves 2x. When `fast` reaches end, `slow` is at the middle.
- **Reverse in-place:** Three pointers — `prev`, `curr`, `next`. Zero extra space.

### C++ Reversal Template
```cpp
ListNode *prev = nullptr, *curr = head, *nxt = nullptr;
while (curr) {
    nxt        = curr->next;
    curr->next = prev;
    prev       = curr;
    curr       = nxt;
}
return prev; // new head
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Reverse Linked List | Easy | Three-pointer reversal | O(N) time, O(1) space |
| Linked List Cycle (Floyd's) | Easy | Fast & slow pointer | O(N) time, O(1) space |
| Merge K Sorted Lists | Hard | Min-heap of `(val, node*)` | O(N log K) time |

### Problem Deep-Dives
- **Merge K Sorted Lists:**
  ```cpp
  using T = pair<int, ListNode*>;
  priority_queue<T, vector<T>, greater<T>> pq;
  for (auto* node : lists) if (node) pq.push({node->val, node});
  // Extract min, push its next child
  ```
- **Cycle Detection — Find Start:** If `fast == slow` (cycle exists), reset one pointer to `head`. Move both at speed 1 — they meet exactly at the cycle start.

---

## Day 10–11 — Stacks & Monotonic Stacks

> 💡 **Core Insight:** A monotonic stack maintains sorted order by popping elements that violate the invariant before pushing new ones. Perfect for "next greater/smaller" patterns.

### Monotonic Stack Template
```cpp
stack<int> stk; // stores indices
vector<int> result(n, -1);

for (int i = 0; i < n; i++) {
    while (!stk.empty() && arr[stk.top()] < arr[i]) {
        result[stk.top()] = arr[i]; // next greater element found
        stk.pop();
    }
    stk.push(i);
}
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Valid Parentheses | Easy | Stack for matching brackets | O(N) time, O(N) space |
| Daily Temperatures | Medium | Monotonic decreasing stack | O(N) time, O(N) space |
| Largest Rectangle in Histogram | Hard | Monotonic increasing stack | O(N) time, O(N) space |

### Problem Deep-Dives
- **Daily Temperatures:** Push indices. When current temp > temp at stack top, the answer for top index = `current_index - top_index`. Pop and repeat.
- **Largest Rectangle in Histogram:** Push indices of increasing bars. When you hit a smaller bar, pop and compute: `width = current_index - stack_top_after_pop - 1`. Add **sentinel bars of height 0** at both ends to flush remaining stack.

---

## Day 12–13 — Queues & Deques

> 💡 **Core Insight:** A deque (double-ended queue) lets you maintain a sliding max/min in O(1) amortized by evicting stale and dominated elements from both ends.

### C++ Deque Basics
```cpp
deque<int> dq;
dq.push_back(x);   dq.push_front(x);
dq.pop_back();     dq.pop_front();
dq.front();        dq.back();
```

### Sliding Window Maximum Template
```cpp
deque<int> dq; // stores indices
vector<int> result;

for (int i = 0; i < n; i++) {
    // Remove indices outside window
    while (!dq.empty() && dq.front() < i - k + 1) dq.pop_front();
    // Remove indices with smaller values (they'll never be max)
    while (!dq.empty() && arr[dq.back()] < arr[i]) dq.pop_back();
    dq.push_back(i);
    if (i >= k - 1) result.push_back(arr[dq.front()]);
}
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Implement Queue using Stacks | Easy | 2-stack push/pop lazy transfer | Amortized O(1) per op |
| Sliding Window Maximum | Hard | Monotonic decreasing deque | O(N) time, O(K) space |

---

## Day 14 — Binary Search (On Answer Space)

> 💡 **Core Insight:** Binary search works on any **monotonic decision space** — not just sorted arrays. Define a feasibility function `f(x)` that returns true/false.

### Universal Binary Search Template
```cpp
int lo = minAnswer, hi = maxAnswer;

while (lo < hi) {
    int mid = lo + (hi - lo) / 2;  // NEVER (lo + hi) / 2 — overflow risk!
    if (feasible(mid)) hi = mid;   // search left half for smallest valid
    else lo = mid + 1;             // search right half
}
return lo; // smallest value where feasible() is true
```

> ⚠️ Always use `mid = lo + (hi - lo) / 2` to prevent integer overflow.

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Binary Search (Standard) | Easy | Classic halving on sorted array | O(log N) |
| Find Minimum in Rotated Sorted Array | Medium | Identify which half is sorted | O(log N) |
| Koko Eating Bananas | Medium | Binary search on speed `[1, max_pile]` | O(N log M) |

### Problem Deep-Dives
- **Koko Eating Bananas:** `feasible(speed)` = can she eat all piles in H hours? = `sum(ceil(pile/speed)) <= H`. Binary search speed from 1 to `max(piles)`.
- **Rotated Sorted Array:** The half that doesn't contain the pivot is fully sorted. If `arr[mid] > arr[right]`, the min is in the right half. Otherwise it's in the left.

---

# WEEK 3: Trees, Graphs & Hierarchies
> *Focus: Recursive thinking and traversing complex networks. Build graph intuition.*

---

## Day 15–17 — Binary Trees & BSTs

> 💡 **Core Insight:** Almost every tree problem is a DFS. Ask: *what info do I need from my children? What do I return to my parent?* Build the recursion from there.

### Traversal Patterns
| Traversal | Order | Best For |
|-----------|-------|----------|
| Pre-order | Root → Left → Right | Serialization, copying |
| In-order | Left → Root → Right | BST sorted sequence, validate BST |
| Post-order | Left → Right → Root | Height, diameter, subtree problems |
| Level-order (BFS) | Level by level | Connect same-level nodes, zigzag |

### DFS Template (Post-order)
```cpp
int helper(TreeNode* root) {
    if (!root) return 0;                         // base case
    int left  = helper(root->left);
    int right = helper(root->right);
    // process current node using left, right
    return /* value to return to parent */;
}
```

### BST Properties (Know by Heart)
- In-order traversal of valid BST is **strictly increasing**.
- For every node: ALL left subtree values < node < ALL right subtree values (not just direct children).
- Validate BST by passing `(minVal, maxVal)` bounds down via recursion.

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Maximum Depth of Binary Tree | Easy | Post-order DFS, return max depth | O(N) time, O(H) space |
| Lowest Common Ancestor | Medium | Post-order: find targets in subtrees | O(N) time, O(H) space |
| Validate Binary Search Tree | Medium | DFS with `(min, max)` bounds | O(N) time, O(H) space |
| Serialize & Deserialize Binary Tree | Hard | Pre-order with null markers | O(N) time, O(N) space |

### Problem Deep-Dives
- **LCA:** If both nodes are in different subtrees of current node, current node IS the LCA. If a node IS one of the targets, return it immediately.
- **Serialize:**
  ```cpp
  // Serialize: pre-order, use "#" for null, "," as delimiter
  // Deserialize: use istringstream + getline(ss, val, ',')
  ```
- **Validate BST:** `bool isValid(node, long minVal, long maxVal)` — use `LONG_MIN`/`LONG_MAX` to handle `INT_MIN` edge case.

---

## Day 18–19 — Graph Fundamentals

> 💡 **Core Insight:** A graph is just a tree with possible cycles and multiple components. BFS for shortest paths; DFS for connectivity and cycle detection.

### Representations in C++
```cpp
// Adjacency List (most common)
vector<vector<int>> adj(n);
adj[u].push_back(v);
adj[v].push_back(u); // undirected

// Grid as implicit graph: 4 directions
int dr[] = {0, 0, 1, -1};
int dc[] = {1, -1, 0, 0};
```

### BFS Template
```cpp
vector<bool> visited(n, false);
queue<int> q;
q.push(start);
visited[start] = true;

while (!q.empty()) {
    int node = q.front(); q.pop();
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            visited[neighbor] = true;
            q.push(neighbor);
        }
    }
}
```

### DFS on Grid Template
```cpp
void dfs(vector<vector<char>>& grid, int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    if (grid[r][c] != '1') return;
    grid[r][c] = '0'; // mark visited in-place
    dfs(grid, r+1, c); dfs(grid, r-1, c);
    dfs(grid, r, c+1); dfs(grid, r, c-1);
}
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Number of Islands | Medium | DFS/BFS flood-fill on grid | O(M×N) time and space |
| Max Area of Island | Medium | DFS returns island size | O(M×N) time and space |
| Clone Graph | Medium | BFS + hash map `old → new node` | O(V+E) time and space |

### Grid DFS Tips
- Mark cells visited by setting `grid[r][c] = '0'` in-place to avoid a separate visited array.
- Always check bounds **before** accessing grid cell.

---

## Day 20–21 — Advanced Graphs

> 💡 **Core Insight:** Three essential algorithms — Topological Sort (DAG ordering), Dijkstra's (shortest weighted path), Union-Find (dynamic connectivity).

### Topological Sort (Kahn's BFS)
```cpp
vector<int> indegree(n, 0);
// Fill adj and indegree from edges...

queue<int> q;
for (int i = 0; i < n; i++)
    if (indegree[i] == 0) q.push(i);

vector<int> order;
while (!q.empty()) {
    int node = q.front(); q.pop();
    order.push_back(node);
    for (int nb : adj[node])
        if (--indegree[nb] == 0) q.push(nb);
}
// If order.size() != n → cycle exists (Course Schedule answer)
```

### Union-Find (DSU) with Path Compression + Union by Rank
```cpp
vector<int> parent(n), rank(n, 0);
iota(parent.begin(), parent.end(), 0); // parent[i] = i

function<int(int)> find = [&](int x) {
    if (parent[x] != x)
        parent[x] = find(parent[x]); // path compression
    return parent[x];
};

auto unite = [&](int x, int y) {
    int px = find(x), py = find(y);
    if (px == py) return;
    if (rank[px] < rank[py]) swap(px, py);
    parent[py] = px;
    if (rank[px] == rank[py]) rank[px]++;
};
```

### Dijkstra's Template
```cpp
using pii = pair<int, int>; // {dist, node}
priority_queue<pii, vector<pii>, greater<pii>> pq;
vector<int> dist(n, INT_MAX);

dist[src] = 0;
pq.push({0, src});

while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    if (d > dist[u]) continue; // stale entry

    for (auto [v, w] : adj[u])
        if (dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            pq.push({dist[v], v});
        }
}
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Course Schedule (Topo Sort) | Medium | Kahn's BFS, detect cycle | O(V+E) time |
| Network Delay Time (Dijkstra) | Medium | Min-heap priority queue | O(E log V) time |
| Accounts Merge (Union-Find) | Medium | DSU on emails as IDs | O(N·α(N)) time |

---

# WEEK 4: Optimization and Simulation
> *Focus: Explore state spaces and make optimal choices. DP and backtracking mastery.*

---

## Day 22–23 — Heaps / Priority Queues

> 💡 **Core Insight:** Heaps give O(log N) insert and O(1) peek at min/max. The standard C++ `priority_queue` is a **MAX-heap** — use `greater<>` for min-heap.

### C++ Heap Patterns
```cpp
// Min-heap
priority_queue<int, vector<int>, greater<int>> minHeap;

// Max-heap (default)
priority_queue<int> maxHeap;

// Custom comparator for pairs (min by first element)
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
```

### Top-K Pattern
```cpp
// Keep a min-heap of size K
// If heap.size() > K, pop the min
// Top of heap = Kth largest element
for (int num : nums) {
    minHeap.push(num);
    if ((int)minHeap.size() > k) minHeap.pop();
}
return minHeap.top(); // Kth largest
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Kth Largest Element in Array | Medium | Min-heap of size K | O(N log K) time |
| Top K Frequent Elements | Medium | Freq map + min-heap of size K | O(N log K) time |
| Find Median from Data Stream | Hard | Max-heap (lower) + min-heap (upper) | O(log N) per insert |

### Problem Deep-Dives
- **Find Median from Data Stream:** Lower half in max-heap, upper half in min-heap. Keep sizes balanced (differ by at most 1). Median = top of larger heap, or average of both tops.
- **Top K Frequent:** Build frequency map, then use min-heap of `(freq, element)` of size K. Alternatively, use bucket sort O(N): index by frequency.

---

## Day 24–25 — Backtracking

> 💡 **Core Insight:** Backtracking = DFS on a decision tree. At each node, **choose** an option, **recurse**, then **undo** the choice. The undo step is what makes it safe to explore all paths.

### Universal Backtracking Template
```cpp
void backtrack(vector<int>& state, vector<int>& choices, int start) {
    if (/* base case */) {
        results.push_back(state);
        return;
    }
    for (int i = start; i < choices.size(); i++) {
        // Choose
        state.push_back(choices[i]);
        // Explore
        backtrack(state, choices, i + 1);
        // Unchoose (backtrack)
        state.pop_back();
    }
}
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Subsets | Medium | Include/exclude at each index | O(2^N × N) time |
| Permutations | Medium | Pick unused elements, `used[]` array | O(N! × N) time |
| Word Search | Medium | DFS on grid + in-place visited marker | O(M×N×4^L) time |

### Problem Deep-Dives
- **Subsets:** Two approaches: (1) At each index, branch include vs exclude. (2) Iterate results — for each new element, add it to all existing subsets.
- **Permutations:** Use a `bool used[n]` array. When `result.size() == n`, it's complete. Loop through all elements, skip `used` ones.
- **Word Search:** Mark `grid[r][c] = '#'` before DFS, restore after. Handles visited in-place without extra space.

---

## Day 26–28 — Dynamic Programming

> 💡 **Core Insight:** DP = Recursion + Memoization. Find overlapping subproblems, define state, write the recurrence, then add a cache or convert to a bottom-up table.

### DP Framework (Apply to Every Problem)
1. **Define state:** What does `dp[i]` (or `dp[i][j]`) represent?
2. **Recurrence:** Express `dp[i]` in terms of smaller subproblems.
3. **Base cases:** What are the simplest inputs with known answers?
4. **Order:** Bottom-up fills table so dependencies are always ready.

### Top-Down (Memoization) Template
```cpp
unordered_map<int, int> memo;

int dp(int i) {
    if (/* base case */) return baseValue;
    if (memo.count(i)) return memo[i];
    return memo[i] = dp(i - 1) + dp(i - 2); // example
}
```

### Bottom-Up Template
```cpp
vector<int> dp(n + 1, 0);
dp[0] = base0;
dp[1] = base1;

for (int i = 2; i <= n; i++)
    dp[i] = dp[i - 1] + dp[i - 2]; // example recurrence
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Climbing Stairs | Easy | `dp[i] = dp[i-1] + dp[i-2]` | O(N) time, O(1) space |
| Coin Change | Medium | `dp[amount]` = min coins needed | O(N×amount) time |
| Longest Increasing Subsequence | Medium | `dp[i] = max(dp[j]+1)` for j < i | O(N log N) with patience sort |
| 0/1 Knapsack | Medium | `dp[i][w]` = max value for i items, weight w | O(N×W) time |

### Problem Deep-Dives
- **Coin Change:**
  ```cpp
  vector<int> dp(amount + 1, INT_MAX);
  dp[0] = 0;
  for (int i = 1; i <= amount; i++)
      for (int coin : coins)
          if (i >= coin && dp[i - coin] != INT_MAX)
              dp[i] = min(dp[i], dp[i - coin] + 1);
  ```
- **LIS O(N log N) — Patience Sort:**
  ```cpp
  vector<int> tails; // tails[i] = smallest tail of IS with length i+1
  for (int num : nums) {
      auto it = lower_bound(tails.begin(), tails.end(), num);
      if (it == tails.end()) tails.push_back(num);
      else *it = num;
  }
  return tails.size();
  ```
- **0/1 Knapsack — 1D Space Optimization:** Iterate weight **backwards** to avoid using same item twice:
  ```cpp
  for (int i = 0; i < n; i++)
      for (int w = W; w >= weight[i]; w--)
          dp[w] = max(dp[w], dp[w - weight[i]] + val[i]);
  ```

---

# THE GAUNTLET — Days 29–30
> *Putting it all together under pressure. No notes. No hints. Just you and the problem.*

---

## Day 29 — System-Level DSA

> These problems test whether you can implement complex data structures from scratch — a common final-round interview question.

### Trie (Prefix Tree) — Full C++ Implementation
```cpp
struct TrieNode {
    unordered_map<char, TrieNode*> children;
    bool isEnd = false;
};

class Trie {
    TrieNode* root = new TrieNode();
public:
    void insert(const string& word) {
        auto node = root;
        for (char c : word) {
            if (!node->children.count(c))
                node->children[c] = new TrieNode();
            node = node->children[c];
        }
        node->isEnd = true;
    }

    bool search(const string& word) {
        auto node = root;
        for (char c : word) {
            if (!node->children.count(c)) return false;
            node = node->children[c];
        }
        return node->isEnd;
    }

    bool startsWith(const string& prefix) {
        auto node = root;
        for (char c : prefix) {
            if (!node->children.count(c)) return false;
            node = node->children[c];
        }
        return true;
    }
};
```

### LRU Cache — Design
```
Need O(1) get and O(1) put.
Requires: doubly linked list (O(1) removal/insertion) + unordered_map (O(1) lookup).
Map stores: key → list iterator.
List is ordered most-to-least recently used.
On access: move node to front.
On eviction (capacity exceeded): remove from back.
```

```cpp
class LRUCache {
    int cap;
    list<pair<int,int>> lru; // {key, val}, front = most recent
    unordered_map<int, list<pair<int,int>>::iterator> map;
public:
    LRUCache(int capacity) : cap(capacity) {}

    int get(int key) {
        if (!map.count(key)) return -1;
        lru.splice(lru.begin(), lru, map[key]); // move to front
        return map[key]->second;
    }

    void put(int key, int value) {
        if (map.count(key)) lru.erase(map[key]);
        lru.push_front({key, value});
        map[key] = lru.begin();
        if ((int)lru.size() > cap) {
            map.erase(lru.back().first);
            lru.pop_back();
        }
    }
};
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Implement Trie (Prefix Tree) | Medium | Array[26] or map of child nodes | O(L) per op, O(N×L) space |
| LRU Cache | Medium | `unordered_map` + doubly linked list | O(1) all operations |

---

## Day 30 — Mock Interview Day

> **RULES: No notes. No LeetCode hints. No looking at previous solutions.**

### Setup
- Choose **4 random UNSEEN** Medium/Hard problems from LeetCode
- Set a timer: **35 minutes per problem**
- Talk out loud for the entire session

### Interview Protocol (Per Problem)

| Minutes | Action |
|---------|--------|
| 0–3 | Read carefully. Ask clarifying questions out loud. Confirm constraints. |
| 3–8 | State brute force approach and its complexity out loud. |
| 8–15 | Identify the pattern. Which week's technique applies? State improvement. |
| 15–30 | Code the optimal solution. Narrate every line. |
| 30–35 | Dry-run on 2 test cases (normal + edge). Fix bugs verbally before typing. |

### Post-Problem Reflection
- [ ] Did I define complexity before coding?
- [ ] Did I handle null/empty/single-element edge cases?
- [ ] Could I explain the "why" of each decision?
- [ ] Did I consider integer overflow risks?

### Recommended Sources
- **LeetCode Explore Cards** → Random shuffle within each topic
- **NeetCode 150** → Shuffle the Hard problems for today
- **Pramp / InterviewBit** → Real mock interview partners

---

# APPENDIX: Quick Reference

---

## Complexity Cheat Sheet

| Algorithm / Structure | Time Complexity | Space Complexity |
|-----------------------|-----------------|------------------|
| Prefix Sum (build) | O(N) | O(N) |
| Two Pointers | O(N) | O(1) |
| Sliding Window | O(N) | O(K) window state |
| Hash Map insert/lookup | O(1) avg | O(N) |
| Binary Search | O(log N) | O(1) |
| DFS / BFS (Tree) | O(N) | O(H) / O(W) |
| DFS / BFS (Graph) | O(V+E) | O(V) |
| Dijkstra's | O(E log V) | O(V) |
| Union-Find (path+rank) | O(α(N)) ≈ O(1) | O(N) |
| Heap insert/pop | O(log N) | O(N) |
| DP 1D | O(N) | O(N) or O(1) |
| DP 2D | O(N×M) | O(N×M) |
| Backtracking (subsets) | O(2^N × N) | O(N) |
| Trie insert/search | O(L) | O(N×L) |
| Quicksort avg | O(N log N) | O(log N) |

---

## C++ STL Quick Reference

| Container | Key Operations | Use Case |
|-----------|----------------|----------|
| `vector<int>` | `push_back` O(1) amortized, `[]` O(1), `sort` O(N log N) | Dynamic array, most problems |
| `unordered_map<K,V>` | insert/lookup O(1) avg | Frequency count, index map |
| `map<K,V>` | insert/lookup O(log N), ordered | Need sorted keys |
| `unordered_set<int>` | insert/count O(1) avg | Existence check |
| `priority_queue<int>` | push/pop O(log N), top O(1) | Max-heap; use `greater<>` for min |
| `stack<int>` | push/pop/top O(1) | LIFO, parentheses, monotonic |
| `queue<int>` | push/pop/front O(1) | FIFO, BFS |
| `deque<int>` | push/pop front & back O(1) | Sliding window maximum |
| `set<int>` | `lower_bound`/`upper_bound` O(log N) | Ordered unique elements |

### Useful Snippets
```cpp
// Sort descending
sort(v.begin(), v.end(), greater<int>());

// Fill iota
vector<int> idx(n); iota(idx.begin(), idx.end(), 0);

// Lambda comparator
sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
});

// String stream for parsing
istringstream ss(input);
string token;
while (getline(ss, token, ',')) { /* process token */ }

// Min/Max with edge cases
int lo = INT_MIN, hi = INT_MAX;
// Use LONG_MIN / LONG_MAX for BST validation bounds
```

---

## 🗺️ Pattern Recognition Flowchart

```
When you see...                          → Think...
─────────────────────────────────────────────────────────────────
"subarray sum / range query"             → Prefix Sum
"sorted array, find pair/triplet"        → Two Pointers (converging)
"longest/shortest substring/subarray"    → Sliding Window
"frequency / grouping / lookup"          → Hash Map / Set
"linked list cycle / middle"             → Fast & Slow Pointers
"next greater / smaller element"         → Monotonic Stack
"search in sorted / rotated array"       → Binary Search
"find answer in a range (minimize max)"  → Binary Search on Answer
"tree traversal / depth / path"          → DFS (recursion)
"shortest path (unweighted)"             → BFS
"shortest path (weighted, positive)"     → Dijkstra's
"dependency ordering / cycle in DAG"     → Topological Sort (Kahn's)
"dynamic connectivity / groups/merge"    → Union-Find (DSU)
"K largest / K smallest / median"        → Heap / Priority Queue
"all combinations / subsets / perms"     → Backtracking
"optimal choice with overlapping subs"   → Dynamic Programming
"prefix lookup / autocomplete"           → Trie
"O(1) LRU / cache eviction"              → HashMap + Doubly Linked List
```

---

## 📅 Weekly Progress Tracker

| Week | Focus | Key Patterns |
|------|-------|--------------|
| Week 1 | Arrays, Pointers, Windows | Prefix Sum, Two Pointers, Sliding Window, Hashing |
| Week 2 | Data Structures | Linked Lists, Stacks, Queues, Binary Search |
| Week 3 | Trees & Graphs | DFS, BFS, Dijkstra's, Topo Sort, Union-Find |
| Week 4 | Optimization | Heaps, Backtracking, Dynamic Programming |
| Days 29–30 | Integration | Trie, LRU Cache, Mock Interview |

---

*Built for C++ developers targeting FAANG & top MNC interviews.*
*Pattern recognition > memorization. Every. Single. Time.*