# 🧠 30-Day DSA Interview Prep Roadmap — Language Independent

> **Goal:** Pattern recognition over memorization. Build the mental muscle to crack top-tier MNC and FAANG interviews.

---

## 📋 Universal Principles

| Principle | Details |
|-----------|---------|
| 🗣️ **Talk out loud** | Simulate a real interview. Say "I'm thinking..." if you pause. |
| ⏱️ **Complexity first** | Define Time & Space Complexity **before** writing a single line of code. |
| 🔄 **Stuck > 45 min** | Study the optimal approach, close it, then code from memory. |
| 📝 **Pattern notes** | After every problem: write a 2-sentence "pattern note" in your own words. |
| 🛠️ **Use standard libraries** | Prefer built-in data structures — know their time complexities. |
| ⚠️ **Edge cases** | Always handle: empty input, single element, integer overflow/underflow. |
| 📄 **Dry-run** | Trace through a small example on paper before submitting. |
| 🔁 **Daily consistency** | 30 minutes daily beats 5-hour weekend cramming. |

---

# WEEK 1: Primitives, Pointers & Windows
> *Focus: Iterate efficiently without nested loops. Master O(N) single-pass techniques.*

---

## Day 1 — Array Basics & Prefix Sums

> 💡 **Core Insight:** Pre-compute running totals so any range-sum query becomes O(1) instead of O(N).

### Concepts to Master
- **Memory layout:** Arrays are contiguous blocks — cache-friendly.
- **Prefix Sum formula:** `prefix[i] = prefix[i-1] + arr[i-1]` with `prefix[0] = 0` (1-indexed trick).
- **Range sum in O(1):** `sum(L, R) = prefix[R+1] - prefix[L]`
- **2D prefix sums:** Extend to matrices for rectangle-sum queries.

### Pseudocode Pattern
```
prefix = array of size (n + 1) initialized to 0
for i from 0 to n-1:
    prefix[i + 1] = prefix[i] + arr[i]

// Sum from index L to R (inclusive)
rangeSum = prefix[R + 1] - prefix[L]
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Product of Array Except Self | Medium | Prefix + Suffix product arrays | O(N) time, O(1) extra |
| Subarray Sum Equals K | Medium | Prefix sum + hash map | O(N) time, O(N) space |

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

### Pseudocode — Converging
```
left = 0
right = n - 1

while left < right:
    if condition_met:
        // use pair at (left, right)
        left++
        right--
    else if need_bigger_sum:
        left++
    else:
        right--
```

### Pseudocode — Fast & Slow
```
slow = head
fast = head

while fast and fast.next exist:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        return true  // cycle detected
return false
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
- **3Sum:** Sort first. For each index `i`, run converging two-pointer on the rest. Skip duplicates carefully.
- **Trapping Rain Water:** `water[i] = min(maxLeft[i], maxRight[i]) - height[i]`. Use two-pointer to reduce to O(1) space.

### ⚠️ Common Mistakes
- **3Sum:** Forgetting to skip duplicate pivot values AND duplicate results after finding a valid triple.
- **Trapping Rain Water:** Off-by-one on the running max — update max **before** computing water.

---

## Day 4–5 — Sliding Window

> 💡 **Core Insight:** Maintain a window `[left, right]`. Expand `right` to explore, shrink `left` to restore validity. Never restart from scratch.

### Fixed vs Variable Window
- **Fixed window (size k):** Move both pointers together. Add new element, remove old element each step.
- **Variable window:** Expand `right` freely. Shrink `left` until window is valid again. Track max/min valid window size.

### Pseudocode — Variable Window
```
frequency_map = empty map
left = 0
max_length = 0

for right from 0 to n-1:
    frequency_map[arr[right]]++
    
    while window is invalid:
        frequency_map[arr[left]]--
        if frequency_map[arr[left]] == 0:
            remove arr[left] from map
        left++
    
    max_length = max(max_length, right - left + 1)
```

### Pseudocode — Fixed Window (size k)
```
window_sum = sum of first k elements

max_sum = window_sum
for i from k to n-1:
    window_sum += arr[i] - arr[i - k]  // slide
    max_sum = max(max_sum, window_sum)
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Maximum Average Subarray I | Easy | Fixed sliding window | O(N) time, O(1) space |
| Longest Substring Without Repeating | Medium | Variable window + hash set | O(N) time, O(26) space |
| Minimum Window Substring | Hard | Variable window + two freq maps | O(N+M) time, O(1) space |

### Problem Deep-Dives
- **Minimum Window Substring:** Track `have` vs `need` counters. When `have == need`, try to shrink `left`. A character is "satisfied" when its frequency in window >= its required frequency.
- **Longest Substring:** Use hash set. Shrink `left` while the new `right` character is already in the set.

---

## Day 6–7 — Hashing (Maps & Sets)

> 💡 **Core Insight:** Trade O(1) space for O(N) space to get O(1) average lookup. Know when hash collisions cause O(N) worst case.

### Hash Container Operations
```
// Hash Map (Dictionary)
map[key] = value              // O(1) avg insert
value = map[key]              // O(1) avg lookup
exists = key in map           // O(1) avg existence check

// Hash Set
set.add(element)              // O(1) avg insert
exists = element in set       // O(1) avg existence check
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Two Sum | Easy | Value → index map | O(N) time, O(N) space |
| Group Anagrams | Medium | Sorted string as canonical key | O(N·K·logK) time |
| Longest Consecutive Sequence | Medium | Hash set + sequence start check | O(N) time, O(N) space |

### Problem Deep-Dives
- **Two Sum:** For each element, check if `(target - element)` already exists in the map. Insert **after** checking to avoid using the same element twice.
- **Group Anagrams:** Sort each string to get a canonical key. Or use a character frequency array as key.
- **Longest Consecutive Sequence:** Only start counting from sequence beginnings (`num-1` not in set). This ensures O(N) total work.

---

# WEEK 2: Structures and Memory
> *Focus: Manage state with LIFO/FIFO and pointer reassignment. Master indirection.*

---

## Day 8–9 — Linked Lists

> 💡 **Core Insight:** You can rewire a list without extra space — just carefully reassign `next` pointers. Always save the next node before overwriting.

### Essential Tricks
- **Dummy head node:** Eliminates edge cases for head operations.
- **Fast & Slow pointer:** `fast` moves 2x. When `fast` reaches end, `slow` is at the middle.
- **Reverse in-place:** Three pointers — `prev`, `curr`, `next`. Zero extra space.

### Pseudocode — Reversal
```
prev = null
curr = head

while curr exists:
    next = curr.next
    curr.next = prev
    prev = curr
    curr = next

return prev  // new head
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Reverse Linked List | Easy | Three-pointer reversal | O(N) time, O(1) space |
| Linked List Cycle (Floyd's) | Easy | Fast & slow pointer | O(N) time, O(1) space |
| Merge K Sorted Lists | Hard | Min-heap of node values | O(N log K) time |

### Problem Deep-Dives
- **Merge K Sorted Lists:** Use a min-heap to track the smallest current node across all lists. Extract min, push its next child.
- **Cycle Detection — Find Start:** If `fast == slow` (cycle exists), reset one pointer to head. Move both at speed 1 — they meet exactly at the cycle start.

---

## Day 10–11 — Stacks & Monotonic Stacks

> 💡 **Core Insight:** A monotonic stack maintains sorted order by popping elements that violate the invariant before pushing new ones. Perfect for "next greater/smaller" patterns.

### Monotonic Stack Pseudocode
```
stack = empty stack  // stores indices
result = array of size n, initialized to -1

for i from 0 to n-1:
    while stack is not empty and arr[stack.top()] < arr[i]:
        result[stack.top()] = arr[i]  // next greater element found
        stack.pop()
    stack.push(i)
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Valid Parentheses | Easy | Stack for matching brackets | O(N) time, O(N) space |
| Daily Temperatures | Medium | Monotonic decreasing stack | O(N) time, O(N) space |
| Largest Rectangle in Histogram | Hard | Monotonic increasing stack | O(N) time, O(N) space |

### Problem Deep-Dives
- **Daily Temperatures:** Push indices. When current temp > temp at stack top, the answer for top index = `current_index - top_index`. Pop and repeat.
- **Largest Rectangle in Histogram:** Push indices of increasing bars. When you hit a smaller bar, pop and compute width. Add sentinel bars of height 0 at both ends.

---

## Day 12–13 — Queues & Deques

> 💡 **Core Insight:** A deque (double-ended queue) lets you maintain a sliding max/min in O(1) amortized by evicting stale and dominated elements from both ends.

### Deque Operations
```
deque.push_back(x)      // Add to rear
deque.push_front(x)     // Add to front
deque.pop_back()        // Remove from rear
deque.pop_front()       // Remove from front
deque.front()           // Access front
deque.back()            // Access rear
```

### Sliding Window Maximum Pseudocode
```
deque = empty deque  // stores indices
result = empty array

for i from 0 to n-1:
    // Remove indices outside window
    while deque not empty and deque.front() < i - k + 1:
        deque.pop_front()
    
    // Remove indices with smaller values
    while deque not empty and arr[deque.back()] < arr[i]:
        deque.pop_back()
    
    deque.push_back(i)
    
    if i >= k - 1:
        result.append(arr[deque.front()])
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Implement Queue using Stacks | Easy | 2-stack push/pop lazy transfer | Amortized O(1) per op |
| Sliding Window Maximum | Hard | Monotonic decreasing deque | O(N) time, O(K) space |

---

## Day 14 — Binary Search (On Answer Space)

> 💡 **Core Insight:** Binary search works on any **monotonic decision space** — not just sorted arrays. Define a feasibility function `f(x)` that returns true/false.

### Universal Binary Search Pseudocode
```
low = minimum_possible_answer
high = maximum_possible_answer

while low < high:
    mid = low + (high - low) / 2  // Prevents overflow
    
    if is_feasible(mid):
        high = mid     // search left half for smallest valid
    else:
        low = mid + 1  // search right half

return low  // smallest value where is_feasible() is true
```

> ⚠️ Always use `mid = low + (high - low) / 2` to prevent integer overflow.

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Binary Search (Standard) | Easy | Classic halving on sorted array | O(log N) |
| Find Minimum in Rotated Sorted Array | Medium | Identify which half is sorted | O(log N) |
| Koko Eating Bananas | Medium | Binary search on speed `[1, max_pile]` | O(N log M) |

### Problem Deep-Dives
- **Koko Eating Bananas:** `is_feasible(speed)` = can she eat all piles in H hours? Binary search speed from 1 to max(piles).
- **Rotated Sorted Array:** The half that doesn't contain the pivot is fully sorted. Determine which half is sorted to narrow the search.

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

### DFS Pseudocode (Post-order)
```
function helper(node):
    if node is null:
        return base_case_value
    
    left_result = helper(node.left)
    right_result = helper(node.right)
    
    // Process current node using left_result, right_result
    return value_to_return_to_parent
```

### BST Properties (Know by Heart)
- In-order traversal of valid BST is **strictly increasing**.
- For every node: ALL left subtree values < node < ALL right subtree values.
- Validate BST by passing `(min_value, max_value)` bounds down via recursion.

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Maximum Depth of Binary Tree | Easy | Post-order DFS, return max depth | O(N) time, O(H) space |
| Lowest Common Ancestor | Medium | Post-order: find targets in subtrees | O(N) time, O(H) space |
| Validate Binary Search Tree | Medium | DFS with `(min, max)` bounds | O(N) time, O(H) space |
| Serialize & Deserialize Binary Tree | Hard | Pre-order with null markers | O(N) time, O(N) space |

### Problem Deep-Dives
- **LCA:** If both nodes are in different subtrees of current node, current node IS the LCA. If a node IS one of the targets, return it immediately.
- **Serialize:** Use pre-order traversal with null markers (e.g., "#") and delimiters (e.g., ",").
- **Validate BST:** Use min/max bounds, handling edge cases for minimum and maximum integer values.

---

## Day 18–19 — Graph Fundamentals

> 💡 **Core Insight:** A graph is just a tree with possible cycles and multiple components. BFS for shortest paths; DFS for connectivity and cycle detection.

### Graph Representations
```
// Adjacency List (most common)
adjacency_list = array of lists/arrays
adjacency_list[u].append(v)
adjacency_list[v].append(u)  // for undirected

// Grid as implicit graph: 4 directions
direction_row = [0, 0, 1, -1]
direction_col = [1, -1, 0, 0]
```

### BFS Pseudocode
```
visited = array of booleans, initialized to false
queue = empty queue
queue.enqueue(start)
visited[start] = true

while queue is not empty:
    node = queue.dequeue()
    
    for each neighbor of node:
        if not visited[neighbor]:
            visited[neighbor] = true
            queue.enqueue(neighbor)
```

### DFS on Grid Pseudocode
```
function dfs(grid, row, col):
    if row < 0 or row >= num_rows or col < 0 or col >= num_cols:
        return
    
    if grid[row][col] != target_value:
        return
    
    grid[row][col] = visited_marker  // mark visited in-place
    
    dfs(grid, row+1, col)
    dfs(grid, row-1, col)
    dfs(grid, row, col+1)
    dfs(grid, row, col-1)
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Number of Islands | Medium | DFS/BFS flood-fill on grid | O(M×N) time and space |
| Max Area of Island | Medium | DFS returns island size | O(M×N) time and space |
| Clone Graph | Medium | BFS + hash map `old → new node` | O(V+E) time and space |

### Grid DFS Tips
- Mark cells visited by modifying the grid in-place to avoid a separate visited array.
- Always check bounds **before** accessing grid cell.

---

## Day 20–21 — Advanced Graphs

> 💡 **Core Insight:** Three essential algorithms — Topological Sort (DAG ordering), Dijkstra's (shortest weighted path), Union-Find (dynamic connectivity).

### Topological Sort (Kahn's Algorithm) Pseudocode
```
indegree = array of size n, initialized to 0
// Fill adjacency list and indegree from edges

queue = empty queue
for i from 0 to n-1:
    if indegree[i] == 0:
        queue.enqueue(i)

topological_order = empty array

while queue is not empty:
    node = queue.dequeue()
    topological_order.append(node)
    
    for each neighbor of node:
        indegree[neighbor]--
        if indegree[neighbor] == 0:
            queue.enqueue(neighbor)

// If topological_order.size != n → cycle exists
```

### Union-Find (DSU) Pseudocode
```
parent = array where parent[i] = i
rank = array of zeros

function find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  // path compression
    return parent[x]

function union(x, y):
    root_x = find(x)
    root_y = find(y)
    
    if root_x == root_y:
        return
    
    // Union by rank
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    else if rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x]++
```

### Dijkstra's Algorithm Pseudocode
```
distance = array of size n, initialized to infinity
distance[source] = 0
priority_queue = min-heap initialized with (0, source)

while priority_queue is not empty:
    (dist, node) = priority_queue.extract_min()
    
    if dist > distance[node]:
        continue  // skip stale entry
    
    for each (neighbor, weight) adjacent to node:
        new_dist = distance[node] + weight
        if new_dist < distance[neighbor]:
            distance[neighbor] = new_dist
            priority_queue.insert((new_dist, neighbor))
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

> 💡 **Core Insight:** Heaps give O(log N) insert and O(1) peek at min/max. Understand whether your language provides max-heap or min-heap by default.

### Heap Operations
```
// Min-heap operations
min_heap.insert(element)      // O(log N)
min_element = min_heap.peek() // O(1)
min_heap.extract_min()        // O(log N)

// Max-heap operations
max_heap.insert(element)      // O(log N)
max_element = max_heap.peek() // O(1)
max_heap.extract_max()        // O(log N)
```

### Top-K Pattern Pseudocode
```
// Keep a min-heap of size K to find K largest elements
min_heap = empty min-heap

for each num in array:
    min_heap.insert(num)
    if min_heap.size() > K:
        min_heap.extract_min()

// Top of heap = Kth largest element
return min_heap.peek()
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Kth Largest Element in Array | Medium | Min-heap of size K | O(N log K) time |
| Top K Frequent Elements | Medium | Freq map + min-heap of size K | O(N log K) time |
| Find Median from Data Stream | Hard | Max-heap (lower) + min-heap (upper) | O(log N) per insert |

### Problem Deep-Dives
- **Find Median from Data Stream:** Lower half in max-heap, upper half in min-heap. Keep sizes balanced (differ by at most 1). Median = top of larger heap, or average of both tops.
- **Top K Frequent:** Build frequency map, then use min-heap of size K. Alternatively, use bucket sort for O(N).

---

## Day 24–25 — Backtracking

> 💡 **Core Insight:** Backtracking = DFS on a decision tree. At each node, **choose** an option, **recurse**, then **undo** the choice. The undo step is what makes it safe to explore all paths.

### Universal Backtracking Pseudocode
```
function backtrack(state, choices, start_index):
    if base_case_met:
        results.append(copy of state)
        return
    
    for i from start_index to choices.length - 1:
        // Choose
        state.append(choices[i])
        
        // Explore
        backtrack(state, choices, i + 1)
        
        // Unchoose (backtrack)
        state.remove_last()
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Subsets | Medium | Include/exclude at each index | O(2^N × N) time |
| Permutations | Medium | Pick unused elements, track used | O(N! × N) time |
| Word Search | Medium | DFS on grid + in-place visited marker | O(M×N×4^L) time |

### Problem Deep-Dives
- **Subsets:** Two approaches: (1) At each index, branch include vs exclude. (2) Iterate results — for each new element, add it to all existing subsets.
- **Permutations:** Use a boolean array to track used elements. When result size equals input size, it's complete.
- **Word Search:** Mark cells as visited before DFS, restore after. Handles visited in-place without extra space.

---

## Day 26–28 — Dynamic Programming

> 💡 **Core Insight:** DP = Recursion + Memoization. Find overlapping subproblems, define state, write the recurrence, then add a cache or convert to a bottom-up table.

### DP Framework (Apply to Every Problem)
1. **Define state:** What does `dp[i]` (or `dp[i][j]`) represent?
2. **Recurrence:** Express `dp[i]` in terms of smaller subproblems.
3. **Base cases:** What are the simplest inputs with known answers?
4. **Order:** Bottom-up fills table so dependencies are always ready.

### Top-Down (Memoization) Pseudocode
```
memo = empty map

function dp(i):
    if base_case:
        return base_value
    
    if i exists in memo:
        return memo[i]
    
    // Compute using smaller subproblems
    result = dp(i - 1) + dp(i - 2)  // example
    memo[i] = result
    return result
```

### Bottom-Up Pseudocode
```
dp = array of size (n + 1), initialized appropriately
dp[0] = base_case_0
dp[1] = base_case_1

for i from 2 to n:
    dp[i] = dp[i - 1] + dp[i - 2]  // example recurrence

return dp[n]
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Climbing Stairs | Easy | `dp[i] = dp[i-1] + dp[i-2]` | O(N) time, O(1) space |
| Coin Change | Medium | `dp[amount]` = min coins needed | O(N×amount) time |
| Longest Increasing Subsequence | Medium | `dp[i] = max(dp[j]+1)` for j < i | O(N²) or O(N log N) |
| 0/1 Knapsack | Medium | `dp[i][w]` = max value for i items, weight w | O(N×W) time |

### Problem Deep-Dives
- **Coin Change:**
  ```
  dp = array of size (amount + 1), initialized to infinity
  dp[0] = 0
  
  for i from 1 to amount:
      for each coin in coins:
          if i >= coin and dp[i - coin] is not infinity:
              dp[i] = min(dp[i], dp[i - coin] + 1)
  ```

- **LIS O(N log N) — Patience Sort:**
  ```
  tails = empty array  // tails[i] = smallest tail of IS with length i+1
  
  for each num in nums:
      position = binary_search_lower_bound(tails, num)
      if position == tails.length:
          tails.append(num)
      else:
          tails[position] = num
  
  return tails.length
  ```

- **0/1 Knapsack — 1D Space Optimization:** Iterate weight **backwards** to avoid using same item twice:
  ```
  for i from 0 to n-1:
      for w from W down to weight[i]:
          dp[w] = max(dp[w], dp[w - weight[i]] + value[i])
  ```

---

# THE GAUNTLET — Days 29–30
> *Putting it all together under pressure. No notes. No hints. Just you and the problem.*

---

## Day 29 — System-Level DSA

> These problems test whether you can implement complex data structures from scratch — a common final-round interview question.

### Trie (Prefix Tree) — Conceptual Structure
```
TrieNode:
    children: map of character to TrieNode
    is_end_of_word: boolean

Trie:
    root: TrieNode
    
    insert(word):
        node = root
        for each character in word:
            if character not in node.children:
                node.children[character] = new TrieNode()
            node = node.children[character]
        node.is_end_of_word = true
    
    search(word):
        node = root
        for each character in word:
            if character not in node.children:
                return false
            node = node.children[character]
        return node.is_end_of_word
    
    starts_with(prefix):
        node = root
        for each character in prefix:
            if character not in node.children:
                return false
            node = node.children[character]
        return true
```

### LRU Cache — Design Concept
```
Need O(1) get and O(1) put operations.
Requires:
  - Doubly linked list (O(1) removal/insertion at any position)
  - Hash map (O(1) lookup): key → list node

List is ordered most-to-least recently used.
- On access: move node to front
- On eviction (capacity exceeded): remove from back
```

### Problems

| Problem | Difficulty | Key Pattern | Complexity |
|---------|-----------|-------------|------------|
| Implement Trie (Prefix Tree) | Medium | Map/array of child nodes | O(L) per op, O(N×L) space |
| LRU Cache | Medium | Hash map + doubly linked list | O(1) all operations |

---

## Day 30 — Mock Interview Day

> **RULES: No notes. No hints. No looking at previous solutions.**

### Setup
- Choose **4 random UNSEEN** Medium/Hard problems
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

## Common Data Structures

| Structure | Key Operations | Time Complexity | Use Case |
|-----------|----------------|-----------------|----------|
| Array/List | Access, append | O(1), O(1) amortized | Dynamic array, most problems |
| Hash Map | Insert, lookup | O(1) avg | Frequency count, index map |
| Hash Set | Insert, lookup | O(1) avg | Existence check, duplicates |
| Stack | Push, pop, peek | O(1) | LIFO, parentheses, monotonic |
| Queue | Enqueue, dequeue | O(1) | FIFO, BFS |
| Deque | Push/pop both ends | O(1) | Sliding window maximum |
| Priority Queue (Heap) | Insert, extract min/max | O(log N) | Top-K, median, Dijkstra's |
| Linked List | Insert, delete | O(1) at position | No random access needed |

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
"dependency ordering / cycle in DAG"     → Topological Sort
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

## 📁 Suggested Folder Structure

```
📦 DSA-Daily-Grind/
 ┣ 📂 your-username/
 ┃ ┣ 📂 day-01/
 ┃ ┃ ┣ 📜 README.md              ← daily plan + reflection
 ┃ ┃ ┣ 📜 product_except_self    ← solution files
 ┃ ┃ ┗ 📜 subarray_sum_k
 ┃ ┣ 📂 day-02/
 ┃ ┃ ┗ ...
 ┃ ┗ 📜 progress-log.md          ← running progress tracker
 ┗ 📜 README.md                  ← this file
```

---

## 📆 Daily Workflow

### 1️⃣ PLAN — Start of Day
Create `day-XX/README.md` and write your intent:

```markdown
## Day XX — [Topic Name]

### Plan
- 🎯 Topic: [e.g., Sliding Window — Variable]
- 🕒 Time Target: [e.g., 1.5 hours]
- 💭 Focus: [What concept or pattern you're drilling today]
```

### 2️⃣ GRIND — Do the Work
- Pick the day's problems from the roadmap
- State complexity before coding
- Write clean code using standard library idioms
- Add your solution files to the day's folder

### 3️⃣ REFLECT — End of Day
Update your `README.md`:

```markdown
### Reflection
- ✅ What I solved today:
- 💡 Key pattern/insight I learned:
- 😓 What tripped me up:
- ⏱️ Time taken per problem:
```

### 4️⃣ COMMIT — Lock In Your Streak
```bash
git add .
git commit -m "Day XX — [Topic]: solved [Problem1], [Problem2]"
git push
```

Watch your contribution graph fill up. Your DSA streak is visible proof of your discipline.

---

## 📊 Progress Log Template

Track your progress in `your-username/progress-log.md`:

```markdown
# My DSA Progress Log

| Day | Date | Topic | Problems Solved | Time | Streak |
|-----|------|-------|-----------------|------|--------|
| 1   | YYYY-MM-DD | Prefix Sums | Product of Array, Subarray Sum K | 90 min | 🔥 1 |
| 2   | YYYY-MM-DD | Two Pointers | Two Sum II, Container With Most Water | 75 min | 🔥 2 |
...

## Pattern Mastery Checklist
- [x] Prefix Sum
- [x] Two Pointers
- [ ] Sliding Window
- [ ] Hashing
- [ ] Linked Lists
- [ ] Stacks & Queues
- [ ] Binary Search
- [ ] Trees
- [ ] Graphs
- [ ] Heaps
- [ ] Backtracking
- [ ] Dynamic Programming
```

---

## ✅ End-of-Day Checklist

Before you commit, make sure:

- [ ] Defined time & space complexity before writing code?
- [ ] Handled edge cases (null, empty, single element, overflow)?
- [ ] Wrote a 2-sentence pattern note in your own words?
- [ ] Can re-solve the problem from scratch without looking?

---

## 🤝 Community Guidelines

| Action | Why |
|--------|-----|
| Browse peers' folders | See different approaches — learn by example |
| Comment & help debug | Teaching a concept locks it in for you |
| Celebrate milestones | Week completions, first Hard solved, full streak |
| Share progress publicly | Public commitment becomes public accountability |

**Social media post template:**

> "Day X of #DSADailyGrind 🔥 — Solved [Problem] using [Pattern]. Key insight: [one sentence]. Streak: X days 💪 [repo link]"

---

## 🚀 Getting Started

```bash
# 1. Create your repository or fork an existing one
# 2. Clone it
git clone https://github.com/YOUR_USERNAME/DSA-Daily-Grind.git

# 3. Create your folder
mkdir your-username
cd your-username

# 4. Start Day 1
mkdir day-01
# Create your README.md, write your plan, start grinding
```

---

## 📚 Recommended Resources

- **LeetCode** — Problem practice platform
- **HackerRank** — Problem practice platform
- **Codeforces** — Competitive programming
- **GeeksforGeeks** — Tutorials and practice
- **Striver's A2Z DSA Sheet** — Comprehensive problem list
- **NeetCode 150** — Curated problem list with video solutions
- **Pramp** — Free mock interviews
- **InterviewBit** — Structured interview prep

---

## 💡 Language-Specific Tips

### When Implementing in Any Language:
- **Use standard library data structures** — don't reinvent the wheel
- **Know time/space complexities** of built-in operations in your language
- **Use appropriate naming conventions** for your language
- **Handle edge cases** specific to your language (null vs None vs nil, integer overflow, etc.)
- **Write idiomatic code** that follows your language's conventions

### Common Pitfalls to Avoid:
- Integer overflow in calculations (especially when computing midpoints)
- Off-by-one errors in array indexing
- Not handling null/empty inputs
- Using floating point for operations that should be integers
- Not initializing data structures properly

---

## 🎯 Final Thoughts

**30 days. 1 pattern at a time.**

You're not just solving problems — you're building the identity of a consistent engineer.

Pattern recognition beats memorization. Every. Single. Time.

---

*Built for developers targeting FAANG & top MNC interviews, regardless of their language of choice.*