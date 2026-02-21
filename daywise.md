# 🧠 30-Day DSA Interview Prep — Complete Day-by-Day Roadmap (C++)

> **Goal:** Pattern recognition over memorization. Crack FAANG & top MNC interviews.
> **Language:** C++ with STL
> **Rules:** Talk out loud • Complexity before code • Stuck >45min → read solution, close it, recode from memory

---

## 📋 Master Schedule

| Day | Topic | Problems |
|-----|-------|----------|
| 1 | Array Basics & Prefix Sums | Product of Array Except Self, Subarray Sum Equals K |
| 2 | Two Pointers — Converging | Two Sum II, Container With Most Water |
| 3 | Two Pointers — Advanced | 3Sum, Trapping Rain Water |
| 4 | Sliding Window — Fixed | Maximum Average Subarray I, Maximum Sum Subarray of Size K |
| 5 | Sliding Window — Variable | Longest Substring Without Repeating Characters, Minimum Window Substring |
| 6 | Hashing — Maps | Two Sum, Group Anagrams |
| 7 | Hashing — Sets & Review | Longest Consecutive Sequence, Week 1 Revision |
| 8 | Linked Lists — Basics | Reverse Linked List, Middle of Linked List |
| 9 | Linked Lists — Advanced | Linked List Cycle, Merge K Sorted Lists |
| 10 | Stacks | Valid Parentheses, Min Stack |
| 11 | Monotonic Stacks | Daily Temperatures, Largest Rectangle in Histogram |
| 12 | Queues & Deques | Implement Queue using Stacks, Sliding Window Maximum |
| 13 | Binary Search — Classic | Binary Search, Search in Rotated Sorted Array |
| 14 | Binary Search — On Answer Space | Koko Eating Bananas, Find Minimum in Rotated Sorted Array |
| 15 | Binary Trees — DFS | Maximum Depth, Path Sum, Diameter of Binary Tree |
| 16 | Binary Trees — BFS & Views | Binary Tree Level Order, Right Side View |
| 17 | BST | Validate BST, Lowest Common Ancestor, Serialize & Deserialize |
| 18 | Graph — BFS/DFS Basics | Number of Islands, Max Area of Island |
| 19 | Graph — Clone & Components | Clone Graph, Pacific Atlantic Water Flow |
| 20 | Graph — Topological Sort | Course Schedule I, Course Schedule II |
| 21 | Graph — Dijkstra & Union-Find | Network Delay Time, Accounts Merge |
| 22 | Heaps — Top K Patterns | Kth Largest Element, Top K Frequent Elements |
| 23 | Heaps — Advanced | Find Median from Data Stream, Task Scheduler |
| 24 | Backtracking — Subsets & Combos | Subsets, Combination Sum |
| 25 | Backtracking — Permutations & Grid | Permutations, Word Search |
| 26 | DP — 1D Foundations | Climbing Stairs, House Robber, Coin Change |
| 27 | DP — Strings & Subsequences | Longest Increasing Subsequence, Longest Common Subsequence |
| 28 | DP — 2D & Knapsack | 0/1 Knapsack, Unique Paths, Edit Distance |
| 29 | System-Level DSA | Implement Trie, LRU Cache |
| 30 | Mock Interview Day | 4 random unseen Medium/Hard problems |

---

# 🗓️ DAY 1 — Array Basics & Prefix Sums

> 💡 **Core Insight:** Pre-compute running totals so any range-sum query becomes O(1) instead of O(N).

### Concepts
- Arrays are contiguous in memory — cache-friendly. Know pointer arithmetic.
- `prefix[i] = prefix[i-1] + arr[i-1]` with `prefix[0] = 0`
- Range sum: `sum(L, R) = prefix[R+1] - prefix[L]`

### C++ Pattern
```cpp
vector<int> prefix(n + 1, 0);
for (int i = 0; i < n; i++)
    prefix[i + 1] = prefix[i] + arr[i];

int rangeSum = prefix[r + 1] - prefix[l]; // O(1) query
```

### 🟡 Problem 1: Product of Array Except Self
- **Approach:** Left-product pass forward, right-product pass backward in-place. No division.
- **Complexity:** O(N) time | O(1) extra space (output array doesn't count)
- **Key Trick:** Running right-product is a single variable, not an array.

```cpp
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n, 1);
    // Left pass
    for (int i = 1; i < n; i++)
        res[i] = res[i-1] * nums[i-1];
    // Right pass
    int right = 1;
    for (int i = n-1; i >= 0; i--) {
        res[i] *= right;
        right *= nums[i];
    }
    return res;
}
```

### 🟡 Problem 2: Subarray Sum Equals K
- **Approach:** Prefix sum + hash map. If `prefix[j] - prefix[i] = k`, a valid subarray exists.
- **Complexity:** O(N) time | O(N) space
- **Key Trick:** Initialize map with `{0: 1}` to handle subarrays starting at index 0.

```cpp
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int,int> prefixCount;
    prefixCount[0] = 1;
    int sum = 0, count = 0;
    for (int num : nums) {
        sum += num;
        count += prefixCount[sum - k];
        prefixCount[sum]++;
    }
    return count;
}
```

### ✅ Day 1 Checklist
- [ ] Write prefix sum array from memory
- [ ] Explain why Product of Array needs no division
- [ ] Trace through Subarray Sum with example `[1,2,3], k=3`

---

# 🗓️ DAY 2 — Two Pointers: Converging

> 💡 **Core Insight:** On a sorted array, two indices moving toward each other can find pairs in O(N) instead of O(N²).

### C++ Template
```cpp
int left = 0, right = n - 1;
while (left < right) {
    int sum = arr[left] + arr[right];
    if (sum == target)      { /* found */ left++; right--; }
    else if (sum < target)  left++;
    else                    right--;
}
```

### 🟡 Problem 1: Two Sum II (Input Array Is Sorted)
- **Approach:** Converging two pointers. Move left if sum < target, right if sum > target.
- **Complexity:** O(N) time | O(1) space

```cpp
vector<int> twoSum(vector<int>& numbers, int target) {
    int l = 0, r = numbers.size() - 1;
    while (l < r) {
        int s = numbers[l] + numbers[r];
        if (s == target) return {l+1, r+1};
        else if (s < target) l++;
        else r--;
    }
    return {};
}
```

### 🟡 Problem 2: Container With Most Water
- **Approach:** Always move the pointer with the **shorter** bar inward. Moving the taller one can never increase the area.
- **Complexity:** O(N) time | O(1) space

```cpp
int maxArea(vector<int>& height) {
    int l = 0, r = height.size() - 1, res = 0;
    while (l < r) {
        res = max(res, min(height[l], height[r]) * (r - l));
        if (height[l] < height[r]) l++;
        else r--;
    }
    return res;
}
```

### ✅ Day 2 Checklist
- [ ] Why do we move the shorter bar, not the taller one?
- [ ] Both problems solved < 25 min combined?
- [ ] Handled edge: array of size 2?

---

# 🗓️ DAY 3 — Two Pointers: Advanced

> 💡 **Core Insight:** Sort first to enable pointer movement decisions. Skip duplicates explicitly to avoid repeated answers.

### 🔴 Problem 1: 3Sum
- **Approach:** Sort. For each index `i`, run converging two-pointer on the rest.
- **Complexity:** O(N²) time | O(1) extra space
- **Key Trick:** Skip duplicate pivots with `while (arr[i] == arr[i-1]) i++`

```cpp
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    for (int i = 0; i < (int)nums.size() - 2; i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue; // skip dup pivot
        int l = i+1, r = nums.size()-1;
        while (l < r) {
            int s = nums[i] + nums[l] + nums[r];
            if (s == 0) {
                res.push_back({nums[i], nums[l], nums[r]});
                while (l < r && nums[l] == nums[l+1]) l++;
                while (l < r && nums[r] == nums[r-1]) r--;
                l++; r--;
            } else if (s < 0) l++;
            else r--;
        }
    }
    return res;
}
```

### 🔴 Problem 2: Trapping Rain Water
- **Approach:** `water[i] = min(maxLeft, maxRight) - height[i]`. Use two pointers to maintain running max from each side.
- **Complexity:** O(N) time | O(1) space

```cpp
int trap(vector<int>& height) {
    int l = 0, r = height.size()-1;
    int maxL = 0, maxR = 0, water = 0;
    while (l < r) {
        if (height[l] <= height[r]) {
            maxL = max(maxL, height[l]);
            water += maxL - height[l];
            l++;
        } else {
            maxR = max(maxR, height[r]);
            water += maxR - height[r];
            r--;
        }
    }
    return water;
}
```

### ✅ Day 3 Checklist
- [ ] Explain why 3Sum is O(N²) not O(N³)
- [ ] Trace Trapping Rain Water on `[0,1,0,2,1,0,1,3,2,1,2,1]`
- [ ] Handle edge: all same numbers in 3Sum → should return empty

---

# 🗓️ DAY 4 — Sliding Window: Fixed Size

> 💡 **Core Insight:** For a fixed window of size k, add the new right element and remove the old left element each step. Never recompute the whole window.

### C++ Template — Fixed Window
```cpp
int windowSum = 0;
for (int i = 0; i < k; i++) windowSum += arr[i]; // seed

int maxSum = windowSum;
for (int i = k; i < n; i++) {
    windowSum += arr[i] - arr[i - k]; // slide
    maxSum = max(maxSum, windowSum);
}
```

### 🟢 Problem 1: Maximum Average Subarray I
- **Approach:** Fixed sliding window of size k. Track running sum.
- **Complexity:** O(N) time | O(1) space

```cpp
double findMaxAverage(vector<int>& nums, int k) {
    double sum = 0;
    for (int i = 0; i < k; i++) sum += nums[i];
    double maxSum = sum;
    for (int i = k; i < (int)nums.size(); i++) {
        sum += nums[i] - nums[i - k];
        maxSum = max(maxSum, sum);
    }
    return maxSum / k;
}
```

### 🟡 Problem 2: Maximum Sum Subarray of Size K *(Classic Warmup)*
- **Approach:** Same fixed window. Foundation for harder problems.
- **Complexity:** O(N) time | O(1) space

```cpp
int maxSumSubarray(vector<int>& arr, int k) {
    int sum = 0;
    for (int i = 0; i < k; i++) sum += arr[i];
    int maxSum = sum;
    for (int i = k; i < (int)arr.size(); i++) {
        sum += arr[i] - arr[i - k];
        maxSum = max(maxSum, sum);
    }
    return maxSum;
}
```

### ✅ Day 4 Checklist
- [ ] Can you write the fixed window template in 60 seconds?
- [ ] What happens if k > array size? Handle it.
- [ ] Extend: find the starting index of the max-sum window.

---

# 🗓️ DAY 5 — Sliding Window: Variable Size

> 💡 **Core Insight:** Expand `right` freely. Shrink `left` until the window is valid again. The window never needs to restart.

### C++ Template — Variable Window
```cpp
unordered_map<char, int> freq;
int left = 0, result = 0;

for (int right = 0; right < n; right++) {
    freq[s[right]]++;                           // expand
    while (/* window invalid */) {
        freq[s[left]]--;
        if (freq[s[left]] == 0) freq.erase(s[left]);
        left++;                                 // shrink
    }
    result = max(result, right - left + 1);     // update answer
}
```

### 🟡 Problem 1: Longest Substring Without Repeating Characters
- **Approach:** Variable window + `unordered_set`. Shrink left while duplicate exists.
- **Complexity:** O(N) time | O(min(N, 26)) space

```cpp
int lengthOfLongestSubstring(string s) {
    unordered_set<char> window;
    int left = 0, maxLen = 0;
    for (int right = 0; right < (int)s.size(); right++) {
        while (window.count(s[right]))
            window.erase(s[left++]);
        window.insert(s[right]);
        maxLen = max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

### 🔴 Problem 2: Minimum Window Substring
- **Approach:** Track `have` vs `need`. When satisfied, shrink left greedily to minimize window.
- **Complexity:** O(N + M) time | O(1) space (26 chars)

```cpp
string minWindow(string s, string t) {
    unordered_map<char,int> need, have_map;
    for (char c : t) need[c]++;
    int have = 0, total = need.size();
    int l = 0, minLen = INT_MAX, start = 0;
    for (int r = 0; r < (int)s.size(); r++) {
        have_map[s[r]]++;
        if (need.count(s[r]) && have_map[s[r]] == need[s[r]]) have++;
        while (have == total) {
            if (r - l + 1 < minLen) { minLen = r - l + 1; start = l; }
            have_map[s[l]]--;
            if (need.count(s[l]) && have_map[s[l]] < need[s[l]]) have--;
            l++;
        }
    }
    return minLen == INT_MAX ? "" : s.substr(start, minLen);
}
```

### ✅ Day 5 Checklist
- [ ] Explain the `have` / `need` logic in your own words
- [ ] What does "window valid" mean for each problem?
- [ ] Both solved < 40 min combined?

---

# 🗓️ DAY 6 — Hashing: Maps

> 💡 **Core Insight:** Use a hash map to remember what you've seen. Check "complement exists in map" before inserting.

### C++ Essentials
```cpp
unordered_map<int,int> mp;
mp[key]++;              // auto-initializes to 0
mp.count(key);          // safe existence check
mp.find(key) != mp.end(); // also valid
```

### 🟢 Problem 1: Two Sum
- **Approach:** For each element, check if `target - element` is already in map. Insert **after** checking.
- **Complexity:** O(N) time | O(N) space

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int,int> seen;
    for (int i = 0; i < (int)nums.size(); i++) {
        int comp = target - nums[i];
        if (seen.count(comp)) return {seen[comp], i};
        seen[nums[i]] = i;
    }
    return {};
}
```

### 🟡 Problem 2: Group Anagrams
- **Approach:** Sort each string → canonical key → group in map.
- **Complexity:** O(N × K log K) time | O(N × K) space

```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> mp;
    for (auto& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        mp[key].push_back(s);
    }
    vector<vector<string>> res;
    for (auto& [k, v] : mp) res.push_back(v);
    return res;
}
```

### ✅ Day 6 Checklist
- [ ] Why insert after checking in Two Sum?
- [ ] Alternative key for Group Anagrams without sorting?
- [ ] What if `target - nums[i] == nums[i]`? Does your code handle it?

---

# 🗓️ DAY 7 — Hashing: Sets & Week 1 Review

> 💡 **Core Insight:** Only start counting from the **beginning** of a sequence. This turns what looks O(N²) into O(N).

### 🟡 Problem 1: Longest Consecutive Sequence
- **Approach:** Hash set of all numbers. Only count from `num` if `num-1` is NOT in set.
- **Complexity:** O(N) time | O(N) space

```cpp
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> s(nums.begin(), nums.end());
    int maxLen = 0;
    for (int num : s) {
        if (!s.count(num - 1)) { // start of sequence
            int cur = num, len = 1;
            while (s.count(cur + 1)) { cur++; len++; }
            maxLen = max(maxLen, len);
        }
    }
    return maxLen;
}
```

### 📚 Week 1 Revision Session
Go back through Days 1–6. For each pattern, answer:
- What is the core insight in one sentence?
- What is the time/space complexity?
- What edge case almost tripped you?

| Pattern | One-Line Core Insight |
|---------|----------------------|
| Prefix Sum | Precompute → O(1) range query |
| Two Pointers | Sorted array → converge to eliminate pairs |
| Sliding Window | Expand right, shrink left, never restart |
| Hashing | Pay O(N) space to get O(1) lookup |

### ✅ Day 7 Checklist
- [ ] Longest Consecutive Sequence without looking at notes
- [ ] Re-solve one problem from Days 1–5 you felt shaky on
- [ ] Write all 4 pattern insights from memory

---

# 🗓️ DAY 8 — Linked Lists: Basics

> 💡 **Core Insight:** Always save `next` before overwriting it. Use a dummy head to eliminate edge cases on the head node.

### Essential C++ Setup
```cpp
// Dummy head trick
ListNode* dummy = new ListNode(0);
dummy->next = head;
ListNode* curr = dummy;

// Three-pointer reversal
ListNode *prev = nullptr, *cur = head, *nxt = nullptr;
while (cur) {
    nxt = cur->next;
    cur->next = prev;
    prev = cur;
    cur = nxt;
}
// prev is new head
```

### 🟢 Problem 1: Reverse Linked List
- **Complexity:** O(N) time | O(1) space

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *cur = head;
    while (cur) {
        ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}
```

### 🟢 Problem 2: Middle of the Linked List
- **Approach:** Fast & slow pointer. When fast reaches end, slow is at middle.
- **Complexity:** O(N) time | O(1) space

```cpp
ListNode* middleNode(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}
```

### ✅ Day 8 Checklist
- [ ] Reverse iteratively AND recursively
- [ ] What does slow point to when list has even nodes?
- [ ] Memory: who owns the dummy node? (Clean it up)

---

# 🗓️ DAY 9 — Linked Lists: Advanced

> 💡 **Core Insight:** For cycle detection — if two runners ever meet, there's a loop. Reset one to head, walk both at speed 1 to find the cycle start.

### 🟢 Problem 1: Linked List Cycle (Floyd's)
- **Complexity:** O(N) time | O(1) space

```cpp
bool hasCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}
```

### 🔴 Problem 2: Merge K Sorted Lists
- **Approach:** Min-heap of `(value, node_ptr)`. Extract min, push its next child.
- **Complexity:** O(N log K) time | O(K) space

```cpp
ListNode* mergeKLists(vector<ListNode*>& lists) {
    using T = pair<int, ListNode*>;
    priority_queue<T, vector<T>, greater<T>> pq;
    for (auto* node : lists)
        if (node) pq.push({node->val, node});

    ListNode dummy(0);
    ListNode* tail = &dummy;
    while (!pq.empty()) {
        auto [val, node] = pq.top(); pq.pop();
        tail->next = node;
        tail = tail->next;
        if (node->next) pq.push({node->next->val, node->next});
    }
    return dummy.next;
}
```

### ✅ Day 9 Checklist
- [ ] Prove Floyd's algorithm works (why does meeting imply cycle?)
- [ ] What if `lists` contains nullptr entries? Handled?
- [ ] Time complexity: why log K and not log N?

---

# 🗓️ DAY 10 — Stacks

> 💡 **Core Insight:** Stack = LIFO. When you need to "undo" or "match" the most recent thing, reach for a stack first.

### 🟢 Problem 1: Valid Parentheses
- **Approach:** Push opening brackets. On closing, check if top matches.
- **Complexity:** O(N) time | O(N) space

```cpp
bool isValid(string s) {
    stack<char> stk;
    unordered_map<char,char> match = {{')', '('}, {']', '['}, {'}', '{'}};
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') stk.push(c);
        else {
            if (stk.empty() || stk.top() != match[c]) return false;
            stk.pop();
        }
    }
    return stk.empty();
}
```

### 🟡 Problem 2: Min Stack
- **Approach:** Two stacks — one regular, one tracking current minimum at each level.
- **Complexity:** O(1) all operations | O(N) space

```cpp
class MinStack {
    stack<int> stk, minStk;
public:
    void push(int val) {
        stk.push(val);
        int m = minStk.empty() ? val : min(val, minStk.top());
        minStk.push(m);
    }
    void pop() { stk.pop(); minStk.pop(); }
    int top() { return stk.top(); }
    int getMin() { return minStk.top(); }
};
```

### ✅ Day 10 Checklist
- [ ] What happens if you pop from an empty stack? Defend your code.
- [ ] Why does `minStk` always push a value, not just on new minimums?
- [ ] Edge case: `"(]"` — expected false. Trace it.

---

# 🗓️ DAY 11 — Monotonic Stacks

> 💡 **Core Insight:** Maintain sorted order in a stack by popping anything that violates the invariant before pushing. Each element enters and exits the stack exactly once → O(N).

### Template — Next Greater Element
```cpp
stack<int> stk; // stores indices
vector<int> result(n, -1);
for (int i = 0; i < n; i++) {
    while (!stk.empty() && arr[stk.top()] < arr[i]) {
        result[stk.top()] = arr[i];
        stk.pop();
    }
    stk.push(i);
}
```

### 🟡 Problem 1: Daily Temperatures
- **Approach:** Monotonic decreasing stack of indices. When current temp is hotter, resolve all colder waiting entries.
- **Complexity:** O(N) time | O(N) space

```cpp
vector<int> dailyTemperatures(vector<int>& T) {
    int n = T.size();
    vector<int> res(n, 0);
    stack<int> stk; // indices
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && T[stk.top()] < T[i]) {
            res[stk.top()] = i - stk.top();
            stk.pop();
        }
        stk.push(i);
    }
    return res;
}
```

### 🔴 Problem 2: Largest Rectangle in Histogram
- **Approach:** Monotonic increasing stack. Add **sentinel** bars of height 0 at both ends to flush the stack.
- **Complexity:** O(N) time | O(N) space

```cpp
int largestRectangleArea(vector<int>& heights) {
    heights.insert(heights.begin(), 0); // left sentinel
    heights.push_back(0);               // right sentinel
    stack<int> stk;
    int maxArea = 0;
    for (int i = 0; i < (int)heights.size(); i++) {
        while (!stk.empty() && heights[stk.top()] > heights[i]) {
            int h = heights[stk.top()]; stk.pop();
            int w = i - stk.top() - 1;
            maxArea = max(maxArea, h * w);
        }
        stk.push(i);
    }
    return maxArea;
}
```

### ✅ Day 11 Checklist
- [ ] Why does each element get pushed/popped exactly once?
- [ ] Trace Histogram on `[2,1,5,6,2,3]` — expected 10
- [ ] What does the width formula `i - stk.top() - 1` compute?

---

# 🗓️ DAY 12 — Queues & Deques

> 💡 **Core Insight:** A monotonic deque lets you track the max/min of a sliding window in O(1) amortized by removing stale and dominated elements.

### 🟢 Problem 1: Implement Queue using Stacks
- **Approach:** Two stacks. `inbox` for push, `outbox` for pop. Transfer lazily.
- **Complexity:** Amortized O(1) per operation

```cpp
class MyQueue {
    stack<int> inbox, outbox;
    void transfer() {
        if (outbox.empty())
            while (!inbox.empty()) { outbox.push(inbox.top()); inbox.pop(); }
    }
public:
    void push(int x) { inbox.push(x); }
    int pop()  { transfer(); int v = outbox.top(); outbox.pop(); return v; }
    int peek() { transfer(); return outbox.top(); }
    bool empty() { return inbox.empty() && outbox.empty(); }
};
```

### 🔴 Problem 2: Sliding Window Maximum
- **Approach:** Deque stores indices. Remove stale (out of window) from front. Remove dominated (smaller) from back.
- **Complexity:** O(N) time | O(K) space

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> dq; // indices
    vector<int> res;
    for (int i = 0; i < (int)nums.size(); i++) {
        // Remove out-of-window from front
        if (!dq.empty() && dq.front() <= i - k) dq.pop_front();
        // Remove smaller elements from back
        while (!dq.empty() && nums[dq.back()] < nums[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) res.push_back(nums[dq.front()]);
    }
    return res;
}
```

### ✅ Day 12 Checklist
- [ ] Why is the two-stack queue amortized O(1), not O(N)?
- [ ] Trace Sliding Window Maximum on `[1,3,-1,-3,5,3,6,7]`, k=3
- [ ] What does "dominated" mean in the deque context?

---

# 🗓️ DAY 13 — Binary Search: Classic

> 💡 **Core Insight:** Halve the search space each step. Always use `mid = lo + (hi - lo) / 2` to prevent overflow.

### Template
```cpp
int lo = 0, hi = n - 1;
while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (arr[mid] == target) return mid;
    else if (arr[mid] < target) lo = mid + 1;
    else hi = mid - 1;
}
return -1;
```

### 🟢 Problem 1: Binary Search (Standard)
- **Complexity:** O(log N) time | O(1) space

```cpp
int search(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
```

### 🟡 Problem 2: Search in Rotated Sorted Array
- **Approach:** One half is always fully sorted. Identify which half and check if target is in it.
- **Complexity:** O(log N) time | O(1) space

```cpp
int search(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) return mid;
        if (nums[lo] <= nums[mid]) { // left half sorted
            if (nums[lo] <= target && target < nums[mid]) hi = mid - 1;
            else lo = mid + 1;
        } else {                     // right half sorted
            if (nums[mid] < target && target <= nums[hi]) lo = mid + 1;
            else hi = mid - 1;
        }
    }
    return -1;
}
```

### ✅ Day 13 Checklist
- [ ] What overflow does `(lo+hi)/2` cause? When?
- [ ] Trace rotated array `[4,5,6,7,0,1,2]`, target=0
- [ ] What if the array has duplicates? Does binary search break?

---

# 🗓️ DAY 14 — Binary Search: On Answer Space

> 💡 **Core Insight:** Binary search works on any monotonic decision space. Define `feasible(x)` → true/false and binary search on x.

### Template — Smallest Valid Answer
```cpp
int lo = minPossible, hi = maxPossible;
while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (feasible(mid)) hi = mid;    // try smaller
    else lo = mid + 1;              // need bigger
}
return lo;
```

### 🟡 Problem 1: Koko Eating Bananas
- **feasible(speed):** Can she eat all piles within H hours?
- **Search range:** `[1, max(piles)]`
- **Complexity:** O(N log M) time | O(1) space

```cpp
int minEatingSpeed(vector<int>& piles, int h) {
    int lo = 1, hi = *max_element(piles.begin(), piles.end());
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        long hours = 0;
        for (int p : piles) hours += (p + mid - 1) / mid; // ceil(p/mid)
        if (hours <= h) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}
```

### 🟡 Problem 2: Find Minimum in Rotated Sorted Array
- **Approach:** If `arr[mid] > arr[right]`, the minimum is in the right half.
- **Complexity:** O(log N) time | O(1) space

```cpp
int findMin(vector<int>& nums) {
    int lo = 0, hi = nums.size() - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] > nums[hi]) lo = mid + 1;
        else hi = mid;
    }
    return nums[lo];
}
```

### ✅ Day 14 Checklist
- [ ] Why is `hi = mid` (not `mid - 1`) in Find Minimum?
- [ ] What makes a problem suitable for binary search on answer space?
- [ ] Implement `feasible` for Koko from scratch, including the ceil division trick

---

# 🗓️ DAY 15 — Binary Trees: DFS

> 💡 **Core Insight:** Post-order DFS is the workhorse — compute left/right subtree answers first, then combine at the current node.

### DFS Template
```cpp
int solve(TreeNode* root) {
    if (!root) return 0;              // base case
    int left  = solve(root->left);
    int right = solve(root->right);
    // combine and return to parent
    return 1 + max(left, right);
}
```

### 🟢 Problem 1: Maximum Depth of Binary Tree
- **Complexity:** O(N) time | O(H) space (call stack)

```cpp
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

### 🟢 Problem 2: Path Sum
- **Approach:** Pass remaining sum down. At leaf, check if remaining == 0.
- **Complexity:** O(N) time | O(H) space

```cpp
bool hasPathSum(TreeNode* root, int targetSum) {
    if (!root) return false;
    if (!root->left && !root->right) return root->val == targetSum;
    return hasPathSum(root->left,  targetSum - root->val) ||
           hasPathSum(root->right, targetSum - root->val);
}
```

### 🟡 Problem 3: Diameter of Binary Tree
- **Approach:** For each node, diameter through it = left height + right height. Track global max.
- **Complexity:** O(N) time | O(H) space

```cpp
int diameterOfBinaryTree(TreeNode* root) {
    int maxDiam = 0;
    function<int(TreeNode*)> height = [&](TreeNode* node) -> int {
        if (!node) return 0;
        int l = height(node->left), r = height(node->right);
        maxDiam = max(maxDiam, l + r);
        return 1 + max(l, r);
    };
    height(root);
    return maxDiam;
}
```

### ✅ Day 15 Checklist
- [ ] Difference between height and depth of a node?
- [ ] What order does post-order DFS visit nodes?
- [ ] Diameter problem: why do we need a global variable?

---

# 🗓️ DAY 16 — Binary Trees: BFS & Views

> 💡 **Core Insight:** Level-order BFS processes all nodes at depth d before depth d+1. Use queue size to separate levels.

### BFS Template
```cpp
queue<TreeNode*> q;
q.push(root);
while (!q.empty()) {
    int levelSize = q.size();           // nodes at current level
    for (int i = 0; i < levelSize; i++) {
        auto node = q.front(); q.pop();
        // process node
        if (node->left)  q.push(node->left);
        if (node->right) q.push(node->right);
    }
}
```

### 🟡 Problem 1: Binary Tree Level Order Traversal
- **Complexity:** O(N) time | O(W) space (max width)

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    vector<vector<int>> res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();
        vector<int> level;
        for (int i = 0; i < sz; i++) {
            auto node = q.front(); q.pop();
            level.push_back(node->val);
            if (node->left)  q.push(node->left);
            if (node->right) q.push(node->right);
        }
        res.push_back(level);
    }
    return res;
}
```

### 🟡 Problem 2: Binary Tree Right Side View
- **Approach:** Level-order BFS. Last node at each level is visible from the right.
- **Complexity:** O(N) time | O(W) space

```cpp
vector<int> rightSideView(TreeNode* root) {
    if (!root) return {};
    vector<int> res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();
        for (int i = 0; i < sz; i++) {
            auto node = q.front(); q.pop();
            if (i == sz - 1) res.push_back(node->val); // last in level
            if (node->left)  q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return res;
}
```

### ✅ Day 16 Checklist
- [ ] BFS vs DFS — when do you choose each for trees?
- [ ] What is the maximum queue size during BFS on a complete binary tree?
- [ ] How would you get the LEFT side view instead?

---

# 🗓️ DAY 17 — BST

> 💡 **Core Insight:** BST invariant applies to ALL ancestors, not just the parent. Pass bounds `(minVal, maxVal)` down via recursion.

### 🟡 Problem 1: Validate Binary Search Tree
- **Complexity:** O(N) time | O(H) space

```cpp
bool isValidBST(TreeNode* root) {
    function<bool(TreeNode*, long, long)> valid =
        [&](TreeNode* node, long lo, long hi) -> bool {
        if (!node) return true;
        if (node->val <= lo || node->val >= hi) return false;
        return valid(node->left,  lo, node->val) &&
               valid(node->right, node->val, hi);
    };
    return valid(root, LONG_MIN, LONG_MAX);
}
```

### 🟡 Problem 2: Lowest Common Ancestor of BST
- **Approach:** Use BST property: if both targets < node, go left. If both > node, go right. Else current is LCA.
- **Complexity:** O(H) time | O(1) space (iterative)

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    while (root) {
        if (p->val < root->val && q->val < root->val)
            root = root->left;
        else if (p->val > root->val && q->val > root->val)
            root = root->right;
        else return root;
    }
    return nullptr;
}
```

### 🔴 Problem 3: Serialize and Deserialize Binary Tree
- **Complexity:** O(N) time | O(N) space

```cpp
string serialize(TreeNode* root) {
    if (!root) return "#,";
    return to_string(root->val) + "," +
           serialize(root->left) + serialize(root->right);
}

TreeNode* deserialize(string data) {
    istringstream ss(data);
    function<TreeNode*()> build = [&]() -> TreeNode* {
        string val;
        getline(ss, val, ',');
        if (val == "#") return nullptr;
        auto node = new TreeNode(stoi(val));
        node->left  = build();
        node->right = build();
        return node;
    };
    return build();
}
```

### ✅ Day 17 Checklist
- [ ] Why use `LONG_MIN`/`LONG_MAX` instead of `INT_MIN`/`INT_MAX` for BST validation?
- [ ] What is the time complexity of BST operations on a skewed tree?
- [ ] Trace serialize/deserialize on a 3-node tree

---

# 🗓️ DAY 18 — Graph: BFS/DFS Basics

> 💡 **Core Insight:** A grid IS a graph. Each cell is a node; its 4 neighbors are edges. Mark visited IN-PLACE to save space.

### Grid DFS Template
```cpp
int rows = grid.size(), cols = grid[0].size();
int dr[] = {0, 0, 1, -1};
int dc[] = {1, -1, 0, 0};

void dfs(vector<vector<char>>& grid, int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    if (grid[r][c] != '1') return;
    grid[r][c] = '0'; // mark visited
    for (int d = 0; d < 4; d++)
        dfs(grid, r + dr[d], c + dc[d]);
}
```

### 🟡 Problem 1: Number of Islands
- **Approach:** For each unvisited '1', DFS to mark the whole island, increment counter.
- **Complexity:** O(M×N) time | O(M×N) space

```cpp
int numIslands(vector<vector<char>>& grid) {
    int count = 0;
    for (int r = 0; r < (int)grid.size(); r++)
        for (int c = 0; c < (int)grid[0].size(); c++)
            if (grid[r][c] == '1') {
                dfs(grid, r, c);
                count++;
            }
    return count;
}
```

### 🟡 Problem 2: Max Area of Island
- **Approach:** DFS returns the size of the island it explored.
- **Complexity:** O(M×N) time | O(M×N) space

```cpp
int dfs(vector<vector<int>>& grid, int r, int c) {
    if (r < 0 || r >= (int)grid.size() ||
        c < 0 || c >= (int)grid[0].size() || grid[r][c] == 0)
        return 0;
    grid[r][c] = 0;
    return 1 + dfs(grid,r+1,c) + dfs(grid,r-1,c)
             + dfs(grid,r,c+1) + dfs(grid,r,c-1);
}

int maxAreaOfIsland(vector<vector<int>>& grid) {
    int maxArea = 0;
    for (int r = 0; r < (int)grid.size(); r++)
        for (int c = 0; c < (int)grid[0].size(); c++)
            maxArea = max(maxArea, dfs(grid, r, c));
    return maxArea;
}
```

### ✅ Day 18 Checklist
- [ ] What is the difference between BFS and DFS on a graph?
- [ ] Why modify the grid in-place instead of a visited array?
- [ ] If you couldn't modify the grid, what would you use?

---

# 🗓️ DAY 19 — Graph: Clone & Multi-Source

> 💡 **Core Insight:** Clone Graph requires a map from original node → clone to handle cycles. Pacific Atlantic requires multi-source BFS from both coasts simultaneously.

### 🟡 Problem 1: Clone Graph
- **Approach:** BFS + hash map `original → clone`.
- **Complexity:** O(V+E) time | O(V) space

```cpp
Node* cloneGraph(Node* node) {
    if (!node) return nullptr;
    unordered_map<Node*, Node*> cloned;
    queue<Node*> q;
    q.push(node);
    cloned[node] = new Node(node->val);
    while (!q.empty()) {
        Node* cur = q.front(); q.pop();
        for (Node* nb : cur->neighbors) {
            if (!cloned.count(nb)) {
                cloned[nb] = new Node(nb->val);
                q.push(nb);
            }
            cloned[cur]->neighbors.push_back(cloned[nb]);
        }
    }
    return cloned[node];
}
```

### 🟡 Problem 2: Pacific Atlantic Water Flow
- **Approach:** Multi-source BFS. Start from Pacific border, then Atlantic border. Answer = intersection.
- **Complexity:** O(M×N) time | O(M×N) space

```cpp
vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
    int m = heights.size(), n = heights[0].size();
    vector<vector<bool>> pac(m, vector<bool>(n, false));
    vector<vector<bool>> atl(m, vector<bool>(n, false));
    auto bfs = [&](queue<pair<int,int>>& q, vector<vector<bool>>& visited) {
        int dr[] = {0,0,1,-1}, dc[] = {1,-1,0,0};
        while (!q.empty()) {
            auto [r, c] = q.front(); q.pop();
            for (int d = 0; d < 4; d++) {
                int nr = r+dr[d], nc = c+dc[d];
                if (nr<0||nr>=m||nc<0||nc>=n||visited[nr][nc]) continue;
                if (heights[nr][nc] >= heights[r][c]) {
                    visited[nr][nc] = true;
                    q.push({nr, nc});
                }
            }
        }
    };
    queue<pair<int,int>> pq, aq;
    for (int i = 0; i < m; i++) { pac[i][0]=true; pq.push({i,0}); atl[i][n-1]=true; aq.push({i,n-1}); }
    for (int j = 0; j < n; j++) { pac[0][j]=true; pq.push({0,j}); atl[m-1][j]=true; aq.push({m-1,j}); }
    bfs(pq, pac); bfs(aq, atl);
    vector<vector<int>> res;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (pac[i][j] && atl[i][j]) res.push_back({i,j});
    return res;
}
```

### ✅ Day 19 Checklist
- [ ] Why does the Clone Graph map prevent infinite loops?
- [ ] In Pacific Atlantic, why search BACKWARDS from the borders?
- [ ] What is multi-source BFS and when do you use it?

---

# 🗓️ DAY 20 — Topological Sort

> 💡 **Core Insight:** Topological sort only works on DAGs. Build an indegree array. Start with all 0-indegree nodes. If not all nodes are processed → cycle exists.

### Kahn's BFS Template
```cpp
vector<int> indegree(n, 0);
// Build adj and indegree from edges...

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
// order.size() == n → no cycle
```

### 🟡 Problem 1: Course Schedule I
- **Answer:** Can you finish all courses? = Does a valid topological order exist?
- **Complexity:** O(V+E) time | O(V+E) space

```cpp
bool canFinish(int n, vector<vector<int>>& prereqs) {
    vector<vector<int>> adj(n);
    vector<int> indegree(n, 0);
    for (auto& e : prereqs) { adj[e[1]].push_back(e[0]); indegree[e[0]]++; }
    queue<int> q;
    for (int i = 0; i < n; i++) if (indegree[i] == 0) q.push(i);
    int count = 0;
    while (!q.empty()) {
        int node = q.front(); q.pop(); count++;
        for (int nb : adj[node]) if (--indegree[nb] == 0) q.push(nb);
    }
    return count == n;
}
```

### 🟡 Problem 2: Course Schedule II
- **Answer:** Return the actual topological order, or empty if cycle.
- **Complexity:** O(V+E) time | O(V+E) space

```cpp
vector<int> findOrder(int n, vector<vector<int>>& prereqs) {
    vector<vector<int>> adj(n);
    vector<int> indegree(n, 0);
    for (auto& e : prereqs) { adj[e[1]].push_back(e[0]); indegree[e[0]]++; }
    queue<int> q;
    for (int i = 0; i < n; i++) if (indegree[i] == 0) q.push(i);
    vector<int> order;
    while (!q.empty()) {
        int node = q.front(); q.pop();
        order.push_back(node);
        for (int nb : adj[node]) if (--indegree[nb] == 0) q.push(nb);
    }
    return order.size() == n ? order : vector<int>{};
}
```

### ✅ Day 20 Checklist
- [ ] What does indegree represent? Why start with indegree 0?
- [ ] DFS-based topo sort vs Kahn's BFS — when would you prefer each?
- [ ] Prove: if order.size() != n, a cycle must exist.

---

# 🗓️ DAY 21 — Dijkstra's & Union-Find

> 💡 **Core Insight:** Dijkstra = greedy shortest path with a min-heap. Union-Find = near O(1) dynamic connectivity with path compression + union by rank.

### 🟡 Problem 1: Network Delay Time (Dijkstra's)
- **Complexity:** O(E log V) time | O(V+E) space

```cpp
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<vector<pair<int,int>>> adj(n+1);
    for (auto& t : times) adj[t[0]].push_back({t[1], t[2]});

    vector<int> dist(n+1, INT_MAX);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    dist[k] = 0; pq.push({0, k});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u])
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
    }
    int ans = *max_element(dist.begin()+1, dist.end());
    return ans == INT_MAX ? -1 : ans;
}
```

### 🟡 Problem 2: Accounts Merge (Union-Find)
- **Approach:** Union all emails belonging to the same account. Group by root.
- **Complexity:** O(N α(N)) ≈ O(N) time

```cpp
// Union-Find (Path Compression + Union by Rank)
vector<int> parent, rnk;
int find(int x) {
    if (parent[x] != x) parent[x] = find(parent[x]);
    return parent[x];
}
void unite(int x, int y) {
    int px = find(x), py = find(y);
    if (px == py) return;
    if (rnk[px] < rnk[py]) swap(px, py);
    parent[py] = px;
    if (rnk[px] == rnk[py]) rnk[px]++;
}
```

### ✅ Day 21 Checklist
- [ ] Why do we skip stale Dijkstra entries with `if (d > dist[u]) continue`?
- [ ] What is α(N) (inverse Ackermann)? Why is it "almost O(1)"?
- [ ] What is the difference between path compression and union by rank?

---

# 🗓️ DAY 22 — Heaps: Top K Patterns

> 💡 **Core Insight:** To find K largest elements, keep a MIN-heap of size K. The smallest in the heap is the Kth largest overall.

### C++ Heap Quick Reference
```cpp
priority_queue<int> maxHeap;                           // default max
priority_queue<int, vector<int>, greater<int>> minHeap; // min-heap
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq; // pairs min
```

### 🟡 Problem 1: Kth Largest Element in an Array
- **Approach:** Min-heap of size K. If size > K, pop the min.
- **Complexity:** O(N log K) time | O(K) space

```cpp
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    for (int num : nums) {
        minHeap.push(num);
        if ((int)minHeap.size() > k) minHeap.pop();
    }
    return minHeap.top();
}
```

### 🟡 Problem 2: Top K Frequent Elements
- **Approach:** Frequency map + min-heap on `(freq, element)` of size K.
- **Complexity:** O(N log K) time | O(N) space

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int,int> freq;
    for (int n : nums) freq[n]++;
    using P = pair<int,int>;
    priority_queue<P, vector<P>, greater<P>> pq; // min-heap by freq
    for (auto& [val, cnt] : freq) {
        pq.push({cnt, val});
        if ((int)pq.size() > k) pq.pop();
    }
    vector<int> res;
    while (!pq.empty()) { res.push_back(pq.top().second); pq.pop(); }
    return res;
}
```

### ✅ Day 22 Checklist
- [ ] Why a min-heap to find K LARGEST? (Counterintuitive — explain it)
- [ ] Bucket sort alternative for Top K Frequent — implement it O(N)
- [ ] What if K == N? What degenerates?

---

# 🗓️ DAY 23 — Heaps: Advanced

> 💡 **Core Insight:** Two heaps partition a data stream into two halves. Max-heap holds lower half, min-heap holds upper half. Median = top(s) of the heap(s).

### 🔴 Problem 1: Find Median from Data Stream
- **Complexity:** O(log N) insert | O(1) findMedian

```cpp
class MedianFinder {
    priority_queue<int> lower;                          // max-heap: lower half
    priority_queue<int, vector<int>, greater<int>> upper; // min-heap: upper half
public:
    void addNum(int num) {
        lower.push(num);
        upper.push(lower.top()); lower.pop();           // balance
        if (lower.size() < upper.size()) {
            lower.push(upper.top()); upper.pop();
        }
    }
    double findMedian() {
        if (lower.size() > upper.size()) return lower.top();
        return (lower.top() + upper.top()) / 2.0;
    }
};
```

### 🟡 Problem 2: Task Scheduler
- **Approach:** Most frequent task determines idle time. Use a max-heap + cooldown queue.
- **Complexity:** O(N log N) time

```cpp
int leastInterval(vector<char>& tasks, int n) {
    unordered_map<char,int> freq;
    for (char t : tasks) freq[t]++;
    priority_queue<int> pq;
    for (auto& [c, f] : freq) pq.push(f);

    int time = 0;
    queue<pair<int,int>> cooldown; // {freq, available_time}
    while (!pq.empty() || !cooldown.empty()) {
        time++;
        if (!cooldown.empty() && cooldown.front().second <= time) {
            pq.push(cooldown.front().first);
            cooldown.pop();
        }
        if (!pq.empty()) {
            int f = pq.top() - 1; pq.pop();
            if (f > 0) cooldown.push({f, time + n + 1});
        }
    }
    return time;
}
```

### ✅ Day 23 Checklist
- [ ] Why does lower.size() >= upper.size() always hold after addNum?
- [ ] Trace MedianFinder with inputs: 1, 2, 3
- [ ] What does the cooldown queue represent in Task Scheduler?

---

# 🗓️ DAY 24 — Backtracking: Subsets & Combinations

> 💡 **Core Insight:** At each index, make a binary decision: include or exclude. The recursion tree has 2^N leaves. Undo your choice (pop) before the next branch.

### Universal Template
```cpp
void backtrack(int start, vector<int>& current, vector<vector<int>>& result,
               vector<int>& nums) {
    result.push_back(current);  // collect at every node (subsets)
    for (int i = start; i < (int)nums.size(); i++) {
        current.push_back(nums[i]);           // choose
        backtrack(i + 1, current, result, nums); // explore
        current.pop_back();                   // unchoose
    }
}
```

### 🟡 Problem 1: Subsets
- **Complexity:** O(2^N × N) time | O(N) space (recursion)

```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> curr;
    function<void(int)> bt = [&](int start) {
        res.push_back(curr);
        for (int i = start; i < (int)nums.size(); i++) {
            curr.push_back(nums[i]);
            bt(i + 1);
            curr.pop_back();
        }
    };
    bt(0);
    return res;
}
```

### 🟡 Problem 2: Combination Sum
- **Approach:** Same element can be reused — don't increment `start` when recursing.
- **Complexity:** O(N^(T/M)) where T=target, M=min element

```cpp
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> res;
    vector<int> curr;
    function<void(int, int)> bt = [&](int start, int rem) {
        if (rem == 0) { res.push_back(curr); return; }
        for (int i = start; i < (int)candidates.size(); i++) {
            if (candidates[i] > rem) break; // prune (sort first)
            curr.push_back(candidates[i]);
            bt(i, rem - candidates[i]); // i not i+1 (reuse allowed)
            curr.pop_back();
        }
    };
    sort(candidates.begin(), candidates.end());
    bt(0, target);
    return res;
}
```

### ✅ Day 24 Checklist
- [ ] What does `pop_back()` undo exactly?
- [ ] Why does Combination Sum pass `i` not `i+1` to allow reuse?
- [ ] Subsets: how many total subsets for an array of size 4?

---

# 🗓️ DAY 25 — Backtracking: Permutations & Grid

> 💡 **Core Insight:** Permutations use a `used[]` array to track which elements are taken. Grid DFS marks cells in-place to simulate the "used" state.

### 🟡 Problem 1: Permutations
- **Complexity:** O(N! × N) time

```cpp
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> curr;
    vector<bool> used(nums.size(), false);
    function<void()> bt = [&]() {
        if (curr.size() == nums.size()) { res.push_back(curr); return; }
        for (int i = 0; i < (int)nums.size(); i++) {
            if (used[i]) continue;
            used[i] = true;
            curr.push_back(nums[i]);
            bt();
            curr.pop_back();
            used[i] = false;
        }
    };
    bt();
    return res;
}
```

### 🟡 Problem 2: Word Search
- **Approach:** DFS on grid. Mark cell `'#'` before recursing, restore after.
- **Complexity:** O(M×N×4^L) time | O(L) space

```cpp
bool exist(vector<vector<char>>& board, string word) {
    int m = board.size(), n = board[0].size();
    function<bool(int,int,int)> dfs = [&](int r, int c, int idx) -> bool {
        if (idx == (int)word.size()) return true;
        if (r<0||r>=m||c<0||c>=n||board[r][c]!=word[idx]) return false;
        char tmp = board[r][c];
        board[r][c] = '#';
        bool found = dfs(r+1,c,idx+1)||dfs(r-1,c,idx+1)||
                     dfs(r,c+1,idx+1)||dfs(r,c-1,idx+1);
        board[r][c] = tmp;  // restore
        return found;
    };
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++)
            if (dfs(r, c, 0)) return true;
    return false;
}
```

### ✅ Day 25 Checklist
- [ ] Why restore `board[r][c]` in Word Search after DFS?
- [ ] How many permutations does an array of 4 elements have?
- [ ] How would you handle duplicate numbers in Permutations?

---

# 🗓️ DAY 26 — Dynamic Programming: 1D Foundations

> 💡 **Core Insight:** DP Framework: (1) Define state. (2) Write recurrence. (3) Set base cases. (4) Fill in bottom-up order.

### 🟢 Problem 1: Climbing Stairs
- `dp[i]` = number of ways to reach step i
- Recurrence: `dp[i] = dp[i-1] + dp[i-2]`
- **Complexity:** O(N) time | O(1) space

```cpp
int climbStairs(int n) {
    if (n <= 2) return n;
    int a = 1, b = 2;
    for (int i = 3; i <= n; i++) { int c = a + b; a = b; b = c; }
    return b;
}
```

### 🟡 Problem 2: House Robber
- `dp[i]` = max money robbing from houses 0..i
- Recurrence: `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
- **Complexity:** O(N) time | O(1) space

```cpp
int rob(vector<int>& nums) {
    int prev2 = 0, prev1 = 0;
    for (int n : nums) {
        int cur = max(prev1, prev2 + n);
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}
```

### 🟡 Problem 3: Coin Change
- `dp[i]` = min coins to make amount i
- Recurrence: `dp[i] = min(dp[i], dp[i-coin] + 1)` for each coin
- **Complexity:** O(N × amount) time | O(amount) space

```cpp
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++)
        for (int coin : coins)
            if (i >= coin && dp[i - coin] != INT_MAX)
                dp[i] = min(dp[i], dp[i - coin] + 1);
    return dp[amount] == INT_MAX ? -1 : dp[amount];
}
```

### ✅ Day 26 Checklist
- [ ] What DP pattern do Climbing Stairs and House Robber share?
- [ ] Why initialize `dp[i] = INT_MAX` in Coin Change?
- [ ] All three problems solved from memory?

---

# 🗓️ DAY 27 — DP: Strings & Subsequences

> 💡 **Core Insight:** For subsequence problems on two strings, the DP table is 2D. `dp[i][j]` captures the relationship between the first i chars of s1 and first j chars of s2.

### 🟡 Problem 1: Longest Increasing Subsequence
- `dp[i]` = length of LIS ending at index i
- Recurrence: `dp[i] = max(dp[j]+1)` for all j < i where `nums[j] < nums[i]`
- **O(N²) approach**, then O(N log N) with patience sort.

```cpp
// O(N log N) — patience sort with binary search
int lengthOfLIS(vector<int>& nums) {
    vector<int> tails;
    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) tails.push_back(num);
        else *it = num;
    }
    return tails.size();
}
```

### 🟡 Problem 2: Longest Common Subsequence
- `dp[i][j]` = LCS of first i chars of text1 and first j chars of text2
- Recurrence: if chars match → `dp[i-1][j-1] + 1`, else `max(dp[i-1][j], dp[i][j-1])`
- **Complexity:** O(M×N) time | O(M×N) space

```cpp
int longestCommonSubsequence(string t1, string t2) {
    int m = t1.size(), n = t2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            dp[i][j] = (t1[i-1] == t2[j-1])
                ? dp[i-1][j-1] + 1
                : max(dp[i-1][j], dp[i][j-1]);
    return dp[m][n];
}
```

### ✅ Day 27 Checklist
- [ ] What does `tails` represent in the patience sort LIS?
- [ ] Trace LCS for "abcde" and "ace" — expected 3
- [ ] Can you reduce LCS to O(N) space? How?

---

# 🗓️ DAY 28 — DP: 2D & Knapsack

> 💡 **Core Insight:** 0/1 Knapsack iterates weights BACKWARDS to prevent using the same item twice within the same pass.

### 🟡 Problem 1: 0/1 Knapsack
- **Complexity:** O(N×W) time | O(W) space (1D optimization)

```cpp
int knapsack(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < n; i++)
        for (int w = W; w >= weights[i]; w--)  // BACKWARDS!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
    return dp[W];
}
```

### 🟡 Problem 2: Unique Paths
- `dp[i][j]` = number of paths to reach cell (i, j)
- Recurrence: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`
- **Complexity:** O(M×N) time | O(N) space

```cpp
int uniquePaths(int m, int n) {
    vector<int> dp(n, 1);
    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++)
            dp[j] += dp[j - 1];
    return dp[n - 1];
}
```

### 🔴 Problem 3: Edit Distance
- `dp[i][j]` = min edits to convert s1[0..i] to s2[0..j]
- **Complexity:** O(M×N) time | O(M×N) space

```cpp
int minDistance(string w1, string w2) {
    int m = w1.size(), n = w2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1));
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            dp[i][j] = (w1[i-1] == w2[j-1])
                ? dp[i-1][j-1]
                : 1 + min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
    return dp[m][n];
}
```

### ✅ Day 28 Checklist
- [ ] Why iterate weights backwards in Knapsack?
- [ ] Three operations in Edit Distance — what are they?
- [ ] Reduce Edit Distance to O(N) space.

---

# 🗓️ DAY 29 — System-Level DSA

> 💡 **Core Insight:** These problems test if you can engineer data structures from scratch — a final-round staple. Know WHY each component is needed.

### 🟡 Problem 1: Implement Trie (Prefix Tree)
- Each node has 26 children (or a map) + `isEnd` flag.
- **Complexity:** O(L) per operation | O(N×L) total space

```cpp
struct TrieNode {
    TrieNode* children[26] = {};
    bool isEnd = false;
};

class Trie {
    TrieNode* root = new TrieNode();
public:
    void insert(string word) {
        auto node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx])
                node->children[idx] = new TrieNode();
            node = node->children[idx];
        }
        node->isEnd = true;
    }
    bool search(string word) {
        auto node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return node->isEnd;
    }
    bool startsWith(string prefix) {
        auto node = root;
        for (char c : prefix) {
            int idx = c - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return true;
    }
};
```

### 🟡 Problem 2: LRU Cache
- `unordered_map<key, list::iterator>` + `list<pair<key,val>>`
- **Complexity:** O(1) get and put

```cpp
class LRUCache {
    int cap;
    list<pair<int,int>> lru;
    unordered_map<int, list<pair<int,int>>::iterator> mp;
public:
    LRUCache(int capacity) : cap(capacity) {}
    int get(int key) {
        if (!mp.count(key)) return -1;
        lru.splice(lru.begin(), lru, mp[key]);
        return mp[key]->second;
    }
    void put(int key, int value) {
        if (mp.count(key)) lru.erase(mp[key]);
        lru.push_front({key, value});
        mp[key] = lru.begin();
        if ((int)lru.size() > cap) {
            mp.erase(lru.back().first);
            lru.pop_back();
        }
    }
};
```

### ✅ Day 29 Checklist
- [ ] Why use `list` (not `vector`) for LRU? What's special about splice?
- [ ] What happens to the Trie's memory? Who cleans up TrieNode?
- [ ] Can you implement Trie with `unordered_map` instead of array? Trade-offs?

---

# 🗓️ DAY 30 — Mock Interview Day

> **NO notes. NO hints. NO peeking at previous solutions.**

### Setup
- Pick **4 random UNSEEN** Medium/Hard problems
- Set a timer: **35 minutes per problem**
- Talk out loud for the ENTIRE session — every thought

### The Protocol (Per Problem)

| Time | What To Do |
|------|-----------|
| 0–3 min | Read twice. Clarify constraints. State input/output clearly. |
| 3–8 min | State the brute force and its complexity out loud. |
| 8–15 min | Identify the pattern — name the week/technique. State optimized complexity. |
| 15–30 min | Code it while narrating every line and decision. |
| 30–35 min | Dry-run 2 test cases (normal + edge). Fix bugs verbally before typing. |

### Post-Problem Self-Assessment

| Check | Did you? |
|-------|---------|
| ✅ | Define time & space complexity before starting? |
| ✅ | Handle empty input / null / single element? |
| ✅ | Avoid integer overflow (used `long` where needed)? |
| ✅ | Explain the key insight, not just the code? |
| ✅ | Finish within 35 minutes? |

### Scoring Rubric

| Score | Criteria |
|-------|---------|
| 5/5 | Optimal solution, clean code, all edge cases, < 25 min |
| 4/5 | Optimal solution, minor code issues, < 35 min |
| 3/5 | Correct but suboptimal, or correct but no edge cases |
| 2/5 | On the right track but incomplete |
| 1/5 | Brute force only |

### Recommended Mock Sources
- LeetCode Weekly Contest (simulate under pressure)
- NeetCode 150 — Hard problems shuffled
- Pramp.com — Real peer mock interviews
- InterviewBit — Timed problem sets

---

# 📊 APPENDIX

## Pattern Recognition Flowchart

```
What type of problem is it?
│
├── Array / subarray
│   ├── Range sum query?                  → Prefix Sum
│   ├── Find pair in sorted array?        → Two Pointers (converging)
│   └── Longest/shortest substring?       → Sliding Window
│
├── Lookup / grouping
│   └── Frequency, existence, index map?  → HashMap / HashSet
│
├── Linked list
│   ├── Reverse / modify?                 → Three-pointer trick
│   └── Cycle / middle?                   → Fast & Slow Pointers
│
├── Stack / queue
│   ├── Matching brackets, undo?          → Stack
│   ├── Next greater/smaller?             → Monotonic Stack
│   └── Sliding window max/min?           → Monotonic Deque
│
├── Search
│   ├── Sorted array?                     → Binary Search
│   └── Search for answer in a range?     → Binary Search on Answer Space
│
├── Tree
│   ├── Height / diameter / path?         → DFS Post-order
│   ├── Level-by-level?                   → BFS
│   └── BST validate / find?              → Bounds-based DFS
│
├── Graph
│   ├── Connectivity / flood fill?        → DFS / BFS
│   ├── Shortest path (unweighted)?       → BFS
│   ├── Shortest path (weighted)?         → Dijkstra's
│   ├── Ordering with dependencies?       → Topological Sort
│   └── Dynamic groups / merge?           → Union-Find
│
├── K-th / Top-K / Scheduling?            → Heap / Priority Queue
├── All combinations / explore choices?   → Backtracking
├── Optimal substructure + overlapping?   → Dynamic Programming
├── Prefix search / autocomplete?         → Trie
└── O(1) cache / recency tracking?        → HashMap + Doubly Linked List
```

---

## Complexity Cheat Sheet

| Algorithm | Time | Space |
|-----------|------|-------|
| Prefix Sum | O(N) | O(N) |
| Two Pointers | O(N) | O(1) |
| Sliding Window | O(N) | O(K) |
| Hash Map ops | O(1) avg | O(N) |
| Binary Search | O(log N) | O(1) |
| DFS/BFS Tree | O(N) | O(H) |
| DFS/BFS Graph | O(V+E) | O(V) |
| Dijkstra's | O(E log V) | O(V) |
| Topo Sort (Kahn's) | O(V+E) | O(V) |
| Union-Find | O(α(N)) | O(N) |
| Heap push/pop | O(log N) | O(N) |
| DP 1D | O(N) | O(1)–O(N) |
| DP 2D | O(M×N) | O(M×N) |
| Backtracking | O(2^N) or O(N!) | O(N) |
| Trie ops | O(L) | O(N×L) |

---

## C++ STL Pitfalls

```cpp
// ❌ Overflow in binary search
int mid = (lo + hi) / 2;      // BAD — overflow if lo+hi > INT_MAX
int mid = lo + (hi - lo) / 2; // ✅ GOOD

// ❌ Accessing non-existent map key creates it
if (mp[key] > 0) { ... }       // BAD — inserts key with value 0
if (mp.count(key) && mp[key] > 0) { ... } // ✅ GOOD

// ❌ Iterator invalidation
for (auto& p : mp) mp.erase(p.first); // BAD — undefined behavior
// ✅ Collect keys first, then erase

// ❌ INT_MAX + 1 overflows
if (dp[i-coin] != INT_MAX) dp[i] = min(dp[i], dp[i-coin] + 1); // ✅ Guard first

// ❌ BST bounds overflow
bool valid(node, int lo, int hi) // BAD — INT_MIN - 1 overflows
bool valid(node, long lo, long hi) // ✅ Use long
```

---

*30 days. 1 pattern at a time. You've got this.*