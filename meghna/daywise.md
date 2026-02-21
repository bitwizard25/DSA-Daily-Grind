# 🧠 30-Day DSA Interview Prep — Complete Day-by-Day Roadmap (Java)

> **Goal:** Pattern recognition over memorization. Crack FAANG & top MNC interviews.
> **Language:** Java with Collections Framework
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
- Arrays are contiguous in memory — cache-friendly
- `prefix[i] = prefix[i-1] + arr[i-1]` with `prefix[0] = 0`
- Range sum: `sum(L, R) = prefix[R+1] - prefix[L]`

### Java Pattern
```java
int[] prefix = new int[n + 1];
for (int i = 0; i < n; i++) {
    prefix[i + 1] = prefix[i] + arr[i];
}

int rangeSum = prefix[r + 1] - prefix[l]; // O(1) query
```

### 🟡 Problem 1: Product of Array Except Self
- **Approach:** Left-product pass forward, right-product pass backward in-place. No division.
- **Complexity:** O(N) time | O(1) extra space (output array doesn't count)
- **Key Trick:** Running right-product is a single variable, not an array.

```java
public int[] productExceptSelf(int[] nums) {
    int n = nums.length;
    int[] res = new int[n];
    Arrays.fill(res, 1);
    
    // Left pass
    for (int i = 1; i < n; i++) {
        res[i] = res[i - 1] * nums[i - 1];
    }
    
    // Right pass
    int right = 1;
    for (int i = n - 1; i >= 0; i--) {
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

```java
public int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> prefixCount = new HashMap<>();
    prefixCount.put(0, 1);
    int sum = 0, count = 0;
    
    for (int num : nums) {
        sum += num;
        count += prefixCount.getOrDefault(sum - k, 0);
        prefixCount.put(sum, prefixCount.getOrDefault(sum, 0) + 1);
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

### Java Template
```java
int left = 0, right = n - 1;
while (left < right) {
    int sum = arr[left] + arr[right];
    if (sum == target) {
        // found
        left++;
        right--;
    } else if (sum < target) {
        left++;
    } else {
        right--;
    }
}
```

### 🟡 Problem 1: Two Sum II (Input Array Is Sorted)
- **Approach:** Converging two pointers. Move left if sum < target, right if sum > target.
- **Complexity:** O(N) time | O(1) space

```java
public int[] twoSum(int[] numbers, int target) {
    int l = 0, r = numbers.length - 1;
    
    while (l < r) {
        int sum = numbers[l] + numbers[r];
        if (sum == target) {
            return new int[]{l + 1, r + 1};
        } else if (sum < target) {
            l++;
        } else {
            r--;
        }
    }
    
    return new int[]{};
}
```

### 🟡 Problem 2: Container With Most Water
- **Approach:** Always move the pointer with the **shorter** bar inward. Moving the taller one can never increase the area.
- **Complexity:** O(N) time | O(1) space

```java
public int maxArea(int[] height) {
    int l = 0, r = height.length - 1;
    int res = 0;
    
    while (l < r) {
        res = Math.max(res, Math.min(height[l], height[r]) * (r - l));
        if (height[l] < height[r]) {
            l++;
        } else {
            r--;
        }
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
- **Complexity:** O(N²) time | O(1) extra space (excluding result)
- **Key Trick:** Skip duplicate pivots with `while (arr[i] == arr[i-1]) i++`

```java
public List<List<Integer>> threeSum(int[] nums) {
    Arrays.sort(nums);
    List<List<Integer>> res = new ArrayList<>();
    
    for (int i = 0; i < nums.length - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue; // skip dup pivot
        
        int l = i + 1, r = nums.length - 1;
        while (l < r) {
            int sum = nums[i] + nums[l] + nums[r];
            if (sum == 0) {
                res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                while (l < r && nums[l] == nums[l + 1]) l++;
                while (l < r && nums[r] == nums[r - 1]) r--;
                l++;
                r--;
            } else if (sum < 0) {
                l++;
            } else {
                r--;
            }
        }
    }
    
    return res;
}
```

### 🔴 Problem 2: Trapping Rain Water
- **Approach:** `water[i] = min(maxLeft, maxRight) - height[i]`. Use two pointers to maintain running max from each side.
- **Complexity:** O(N) time | O(1) space

```java
public int trap(int[] height) {
    int l = 0, r = height.length - 1;
    int maxL = 0, maxR = 0;
    int water = 0;
    
    while (l < r) {
        if (height[l] <= height[r]) {
            maxL = Math.max(maxL, height[l]);
            water += maxL - height[l];
            l++;
        } else {
            maxR = Math.max(maxR, height[r]);
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

### Java Template — Fixed Window
```java
int windowSum = 0;
for (int i = 0; i < k; i++) {
    windowSum += arr[i]; // seed
}

int maxSum = windowSum;
for (int i = k; i < n; i++) {
    windowSum += arr[i] - arr[i - k]; // slide
    maxSum = Math.max(maxSum, windowSum);
}
```

### 🟢 Problem 1: Maximum Average Subarray I
- **Approach:** Fixed sliding window of size k. Track running sum.
- **Complexity:** O(N) time | O(1) space

```java
public double findMaxAverage(int[] nums, int k) {
    double sum = 0;
    for (int i = 0; i < k; i++) {
        sum += nums[i];
    }
    
    double maxSum = sum;
    for (int i = k; i < nums.length; i++) {
        sum += nums[i] - nums[i - k];
        maxSum = Math.max(maxSum, sum);
    }
    
    return maxSum / k;
}
```

### 🟡 Problem 2: Maximum Sum Subarray of Size K *(Classic Warmup)*
- **Approach:** Same fixed window. Foundation for harder problems.
- **Complexity:** O(N) time | O(1) space

```java
public int maxSumSubarray(int[] arr, int k) {
    int sum = 0;
    for (int i = 0; i < k; i++) {
        sum += arr[i];
    }
    
    int maxSum = sum;
    for (int i = k; i < arr.length; i++) {
        sum += arr[i] - arr[i - k];
        maxSum = Math.max(maxSum, sum);
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

### Java Template — Variable Window
```java
Map<Character, Integer> freq = new HashMap<>();
int left = 0, result = 0;

for (int right = 0; right < n; right++) {
    freq.put(s.charAt(right), freq.getOrDefault(s.charAt(right), 0) + 1);
    
    while (/* window invalid */) {
        freq.put(s.charAt(left), freq.get(s.charAt(left)) - 1);
        if (freq.get(s.charAt(left)) == 0) {
            freq.remove(s.charAt(left));
        }
        left++;
    }
    
    result = Math.max(result, right - left + 1);
}
```

### 🟡 Problem 1: Longest Substring Without Repeating Characters
- **Approach:** Variable window + `HashSet`. Shrink left while duplicate exists.
- **Complexity:** O(N) time | O(min(N, 26)) space

```java
public int lengthOfLongestSubstring(String s) {
    Set<Character> window = new HashSet<>();
    int left = 0, maxLen = 0;
    
    for (int right = 0; right < s.length(); right++) {
        while (window.contains(s.charAt(right))) {
            window.remove(s.charAt(left));
            left++;
        }
        window.add(s.charAt(right));
        maxLen = Math.max(maxLen, right - left + 1);
    }
    
    return maxLen;
}
```

### 🔴 Problem 2: Minimum Window Substring
- **Approach:** Track `have` vs `need`. When satisfied, shrink left greedily to minimize window.
- **Complexity:** O(N + M) time | O(1) space (26 chars)

```java
public String minWindow(String s, String t) {
    Map<Character, Integer> need = new HashMap<>();
    Map<Character, Integer> have = new HashMap<>();
    
    for (char c : t.toCharArray()) {
        need.put(c, need.getOrDefault(c, 0) + 1);
    }
    
    int formed = 0, required = need.size();
    int l = 0, minLen = Integer.MAX_VALUE, start = 0;
    
    for (int r = 0; r < s.length(); r++) {
        char c = s.charAt(r);
        have.put(c, have.getOrDefault(c, 0) + 1);
        
        if (need.containsKey(c) && have.get(c).intValue() == need.get(c).intValue()) {
            formed++;
        }
        
        while (formed == required) {
            if (r - l + 1 < minLen) {
                minLen = r - l + 1;
                start = l;
            }
            
            char leftChar = s.charAt(l);
            have.put(leftChar, have.get(leftChar) - 1);
            if (need.containsKey(leftChar) && have.get(leftChar) < need.get(leftChar)) {
                formed--;
            }
            l++;
        }
    }
    
    return minLen == Integer.MAX_VALUE ? "" : s.substring(start, start + minLen);
}
```

### ✅ Day 5 Checklist
- [ ] Explain the `formed` / `required` logic in your own words
- [ ] What does "window valid" mean for each problem?
- [ ] Both solved < 40 min combined?

---

# 🗓️ DAY 6 — Hashing: Maps

> 💡 **Core Insight:** Use a hash map to remember what you've seen. Check "complement exists in map" before inserting.

### Java Essentials
```java
Map<Integer, Integer> map = new HashMap<>();
map.put(key, map.getOrDefault(key, 0) + 1);
map.containsKey(key);  // check existence
map.get(key);          // retrieve value
```

### 🟢 Problem 1: Two Sum
- **Approach:** For each element, check if `target - element` is already in map. Insert **after** checking.
- **Complexity:** O(N) time | O(N) space

```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> seen = new HashMap<>();
    
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (seen.containsKey(complement)) {
            return new int[]{seen.get(complement), i};
        }
        seen.put(nums[i], i);
    }
    
    return new int[]{};
}
```

### 🟡 Problem 2: Group Anagrams
- **Approach:** Sort each string → canonical key → group in map.
- **Complexity:** O(N × K log K) time | O(N × K) space

```java
public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> map = new HashMap<>();
    
    for (String s : strs) {
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        String key = new String(chars);
        
        map.putIfAbsent(key, new ArrayList<>());
        map.get(key).add(s);
    }
    
    return new ArrayList<>(map.values());
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

```java
public int longestConsecutive(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int num : nums) {
        set.add(num);
    }
    
    int maxLen = 0;
    for (int num : set) {
        if (!set.contains(num - 1)) { // start of sequence
            int cur = num, len = 1;
            while (set.contains(cur + 1)) {
                cur++;
                len++;
            }
            maxLen = Math.max(maxLen, len);
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

### Essential Java Setup
```java
// Dummy head trick
ListNode dummy = new ListNode(0);
dummy.next = head;
ListNode curr = dummy;

// Three-pointer reversal
ListNode prev = null, cur = head, nxt = null;
while (cur != null) {
    nxt = cur.next;
    cur.next = prev;
    prev = cur;
    cur = nxt;
}
// prev is new head
```

### 🟢 Problem 1: Reverse Linked List
- **Complexity:** O(N) time | O(1) space

```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null, cur = head;
    
    while (cur != null) {
        ListNode nxt = cur.next;
        cur.next = prev;
        prev = cur;
        cur = nxt;
    }
    
    return prev;
}
```

### 🟢 Problem 2: Middle of the Linked List
- **Approach:** Fast & slow pointer. When fast reaches end, slow is at middle.
- **Complexity:** O(N) time | O(1) space

```java
public ListNode middleNode(ListNode head) {
    ListNode slow = head, fast = head;
    
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    
    return slow;
}
```

### ✅ Day 8 Checklist
- [ ] Reverse iteratively AND recursively
- [ ] What does slow point to when list has even nodes?
- [ ] Memory: who owns the dummy node? (GC handles it in Java)

---

# 🗓️ DAY 9 — Linked Lists: Advanced

> 💡 **Core Insight:** For cycle detection — if two runners ever meet, there's a loop. Reset one to head, walk both at speed 1 to find the cycle start.

### 🟢 Problem 1: Linked List Cycle (Floyd's)
- **Complexity:** O(N) time | O(1) space

```java
public boolean hasCycle(ListNode head) {
    ListNode slow = head, fast = head;
    
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) {
            return true;
        }
    }
    
    return false;
}
```

### 🔴 Problem 2: Merge K Sorted Lists
- **Approach:** Min-heap of `ListNode`. Extract min, push its next child.
- **Complexity:** O(N log K) time | O(K) space

```java
public ListNode mergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
    
    for (ListNode node : lists) {
        if (node != null) {
            pq.offer(node);
        }
    }
    
    ListNode dummy = new ListNode(0);
    ListNode tail = dummy;
    
    while (!pq.isEmpty()) {
        ListNode node = pq.poll();
        tail.next = node;
        tail = tail.next;
        if (node.next != null) {
            pq.offer(node.next);
        }
    }
    
    return dummy.next;
}
```

### ✅ Day 9 Checklist
- [ ] Prove Floyd's algorithm works (why does meeting imply cycle?)
- [ ] What if `lists` contains null entries? Handled?
- [ ] Time complexity: why log K and not log N?

---

# 🗓️ DAY 10 — Stacks

> 💡 **Core Insight:** Stack = LIFO. When you need to "undo" or "match" the most recent thing, reach for a stack first.

### 🟢 Problem 1: Valid Parentheses
- **Approach:** Push opening brackets. On closing, check if top matches.
- **Complexity:** O(N) time | O(N) space

```java
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    Map<Character, Character> match = Map.of(')', '(', ']', '[', '}', '{');
    
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '[' || c == '{') {
            stack.push(c);
        } else {
            if (stack.isEmpty() || stack.pop() != match.get(c)) {
                return false;
            }
        }
    }
    
    return stack.isEmpty();
}
```

### 🟡 Problem 2: Min Stack
- **Approach:** Two stacks — one regular, one tracking current minimum at each level.
- **Complexity:** O(1) all operations | O(N) space

```java
class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> minStack;
    
    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }
    
    public void push(int val) {
        stack.push(val);
        int min = minStack.isEmpty() ? val : Math.min(val, minStack.peek());
        minStack.push(min);
    }
    
    public void pop() {
        stack.pop();
        minStack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int getMin() {
        return minStack.peek();
    }
}
```

### ✅ Day 10 Checklist
- [ ] What happens if you pop from an empty stack? Defend your code.
- [ ] Why does `minStack` always push a value, not just on new minimums?
- [ ] Edge case: `"(]"` — expected false. Trace it.

---

# 🗓️ DAY 11 — Monotonic Stacks

> 💡 **Core Insight:** Maintain sorted order in a stack by popping anything that violates the invariant before pushing. Each element enters and exits the stack exactly once → O(N).

### Template — Next Greater Element
```java
Stack<Integer> stack = new Stack<>(); // stores indices
int[] result = new int[n];
Arrays.fill(result, -1);

for (int i = 0; i < n; i++) {
    while (!stack.isEmpty() && arr[stack.peek()] < arr[i]) {
        result[stack.pop()] = arr[i];
    }
    stack.push(i);
}
```

### 🟡 Problem 1: Daily Temperatures
- **Approach:** Monotonic decreasing stack of indices. When current temp is hotter, resolve all colder waiting entries.
- **Complexity:** O(N) time | O(N) space

```java
public int[] dailyTemperatures(int[] temperatures) {
    int n = temperatures.length;
    int[] res = new int[n];
    Stack<Integer> stack = new Stack<>();
    
    for (int i = 0; i < n; i++) {
        while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
            int idx = stack.pop();
            res[idx] = i - idx;
        }
        stack.push(i);
    }
    
    return res;
}
```

### 🔴 Problem 2: Largest Rectangle in Histogram
- **Approach:** Monotonic increasing stack. Add **sentinel** bars of height 0 at both ends to flush the stack.
- **Complexity:** O(N) time | O(N) space

```java
public int largestRectangleArea(int[] heights) {
    Stack<Integer> stack = new Stack<>();
    int maxArea = 0;
    
    // Add left sentinel
    int[] h = new int[heights.length + 2];
    h[0] = 0;
    h[h.length - 1] = 0;
    for (int i = 0; i < heights.length; i++) {
        h[i + 1] = heights[i];
    }
    
    for (int i = 0; i < h.length; i++) {
        while (!stack.isEmpty() && h[stack.peek()] > h[i]) {
            int height = h[stack.pop()];
            int width = i - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }
        stack.push(i);
    }
    
    return maxArea;
}
```

### ✅ Day 11 Checklist
- [ ] Why does each element get pushed/popped exactly once?
- [ ] Trace Histogram on `[2,1,5,6,2,3]` — expected 10
- [ ] What does the width formula `i - stack.peek() - 1` compute?

---

# 🗓️ DAY 12 — Queues & Deques

> 💡 **Core Insight:** A monotonic deque lets you track the max/min of a sliding window in O(1) amortized by removing stale and dominated elements.

### 🟢 Problem 1: Implement Queue using Stacks
- **Approach:** Two stacks. `inbox` for push, `outbox` for pop. Transfer lazily.
- **Complexity:** Amortized O(1) per operation

```java
class MyQueue {
    private Stack<Integer> inbox;
    private Stack<Integer> outbox;
    
    public MyQueue() {
        inbox = new Stack<>();
        outbox = new Stack<>();
    }
    
    private void transfer() {
        if (outbox.isEmpty()) {
            while (!inbox.isEmpty()) {
                outbox.push(inbox.pop());
            }
        }
    }
    
    public void push(int x) {
        inbox.push(x);
    }
    
    public int pop() {
        transfer();
        return outbox.pop();
    }
    
    public int peek() {
        transfer();
        return outbox.peek();
    }
    
    public boolean empty() {
        return inbox.isEmpty() && outbox.isEmpty();
    }
}
```

### 🔴 Problem 2: Sliding Window Maximum
- **Approach:** Deque stores indices. Remove stale (out of window) from front. Remove dominated (smaller) from back.
- **Complexity:** O(N) time | O(K) space

```java
public int[] maxSlidingWindow(int[] nums, int k) {
    Deque<Integer> dq = new ArrayDeque<>();
    int[] res = new int[nums.length - k + 1];
    int idx = 0;
    
    for (int i = 0; i < nums.length; i++) {
        // Remove out-of-window from front
        if (!dq.isEmpty() && dq.peekFirst() <= i - k) {
            dq.pollFirst();
        }
        
        // Remove smaller elements from back
        while (!dq.isEmpty() && nums[dq.peekLast()] < nums[i]) {
            dq.pollLast();
        }
        
        dq.offerLast(i);
        
        if (i >= k - 1) {
            res[idx++] = nums[dq.peekFirst()];
        }
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
```java
int lo = 0, hi = n - 1;
while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (arr[mid] == target) {
        return mid;
    } else if (arr[mid] < target) {
        lo = mid + 1;
    } else {
        hi = mid - 1;
    }
}
return -1;
```

### 🟢 Problem 1: Binary Search (Standard)
- **Complexity:** O(log N) time | O(1) space

```java
public int search(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    
    return -1;
}
```

### 🟡 Problem 2: Search in Rotated Sorted Array
- **Approach:** One half is always fully sorted. Identify which half and check if target is in it.
- **Complexity:** O(log N) time | O(1) space

```java
public int search(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) {
            return mid;
        }
        
        if (nums[lo] <= nums[mid]) { // left half sorted
            if (nums[lo] <= target && target < nums[mid]) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        } else { // right half sorted
            if (nums[mid] < target && target <= nums[hi]) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
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
```java
int lo = minPossible, hi = maxPossible;
while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (feasible(mid)) {
        hi = mid;    // try smaller
    } else {
        lo = mid + 1; // need bigger
    }
}
return lo;
```

### 🟡 Problem 1: Koko Eating Bananas
- **feasible(speed):** Can she eat all piles within H hours?
- **Search range:** `[1, max(piles)]`
- **Complexity:** O(N log M) time | O(1) space

```java
public int minEatingSpeed(int[] piles, int h) {
    int lo = 1, hi = 0;
    for (int pile : piles) {
        hi = Math.max(hi, pile);
    }
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        long hours = 0;
        for (int pile : piles) {
            hours += (pile + mid - 1) / mid; // ceil(pile/mid)
        }
        
        if (hours <= h) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    
    return lo;
}
```

### 🟡 Problem 2: Find Minimum in Rotated Sorted Array
- **Approach:** If `arr[mid] > arr[right]`, the minimum is in the right half.
- **Complexity:** O(log N) time | O(1) space

```java
public int findMin(int[] nums) {
    int lo = 0, hi = nums.length - 1;
    
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] > nums[hi]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
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
```java
public int solve(TreeNode root) {
    if (root == null) return 0;              // base case
    int left  = solve(root.left);
    int right = solve(root.right);
    // combine and return to parent
    return 1 + Math.max(left, right);
}
```

### 🟢 Problem 1: Maximum Depth of Binary Tree
- **Complexity:** O(N) time | O(H) space (call stack)

```java
public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
}
```

### 🟢 Problem 2: Path Sum
- **Approach:** Pass remaining sum down. At leaf, check if remaining == 0.
- **Complexity:** O(N) time | O(H) space

```java
public boolean hasPathSum(TreeNode root, int targetSum) {
    if (root == null) return false;
    if (root.left == null && root.right == null) {
        return root.val == targetSum;
    }
    return hasPathSum(root.left, targetSum - root.val) ||
           hasPathSum(root.right, targetSum - root.val);
}
```

### 🟡 Problem 3: Diameter of Binary Tree
- **Approach:** For each node, diameter through it = left height + right height. Track global max.
- **Complexity:** O(N) time | O(H) space

```java
private int maxDiam = 0;

public int diameterOfBinaryTree(TreeNode root) {
    height(root);
    return maxDiam;
}

private int height(TreeNode node) {
    if (node == null) return 0;
    int l = height(node.left);
    int r = height(node.right);
    maxDiam = Math.max(maxDiam, l + r);
    return 1 + Math.max(l, r);
}
```

### ✅ Day 15 Checklist
- [ ] Difference between height and depth of a node?
- [ ] What order does post-order DFS visit nodes?
- [ ] Diameter problem: why do we need an instance variable?

---

# 🗓️ DAY 16 — Binary Trees: BFS & Views

> 💡 **Core Insight:** Level-order BFS processes all nodes at depth d before depth d+1. Use queue size to separate levels.

### BFS Template
```java
Queue<TreeNode> q = new LinkedList<>();
q.offer(root);

while (!q.isEmpty()) {
    int levelSize = q.size();
    for (int i = 0; i < levelSize; i++) {
        TreeNode node = q.poll();
        // process node
        if (node.left != null) q.offer(node.left);
        if (node.right != null) q.offer(node.right);
    }
}
```

### 🟡 Problem 1: Binary Tree Level Order Traversal
- **Complexity:** O(N) time | O(W) space (max width)

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    if (root == null) return res;
    
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    
    while (!q.isEmpty()) {
        int size = q.size();
        List<Integer> level = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            TreeNode node = q.poll();
            level.add(node.val);
            if (node.left != null) q.offer(node.left);
            if (node.right != null) q.offer(node.right);
        }
        res.add(level);
    }
    
    return res;
}
```

### 🟡 Problem 2: Binary Tree Right Side View
- **Approach:** Level-order BFS. Last node at each level is visible from the right.
- **Complexity:** O(N) time | O(W) space

```java
public List<Integer> rightSideView(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if (root == null) return res;
    
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    
    while (!q.isEmpty()) {
        int size = q.size();
        for (int i = 0; i < size; i++) {
            TreeNode node = q.poll();
            if (i == size - 1) res.add(node.val); // last in level
            if (node.left != null) q.offer(node.left);
            if (node.right != null) q.offer(node.right);
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

```java
public boolean isValidBST(TreeNode root) {
    return valid(root, Long.MIN_VALUE, Long.MAX_VALUE);
}

private boolean valid(TreeNode node, long lo, long hi) {
    if (node == null) return true;
    if (node.val <= lo || node.val >= hi) return false;
    return valid(node.left, lo, node.val) &&
           valid(node.right, node.val, hi);
}
```

### 🟡 Problem 2: Lowest Common Ancestor of BST
- **Approach:** Use BST property: if both targets < node, go left. If both > node, go right. Else current is LCA.
- **Complexity:** O(H) time | O(1) space (iterative)

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    while (root != null) {
        if (p.val < root.val && q.val < root.val) {
            root = root.left;
        } else if (p.val > root.val && q.val > root.val) {
            root = root.right;
        } else {
            return root;
        }
    }
    return null;
}
```

### 🔴 Problem 3: Serialize and Deserialize Binary Tree
- **Complexity:** O(N) time | O(N) space

```java
// Encodes a tree to a single string
public String serialize(TreeNode root) {
    StringBuilder sb = new StringBuilder();
    serializeHelper(root, sb);
    return sb.toString();
}

private void serializeHelper(TreeNode node, StringBuilder sb) {
    if (node == null) {
        sb.append("#,");
        return;
    }
    sb.append(node.val).append(",");
    serializeHelper(node.left, sb);
    serializeHelper(node.right, sb);
}

// Decodes your encoded data to tree
public TreeNode deserialize(String data) {
    Queue<String> nodes = new LinkedList<>(Arrays.asList(data.split(",")));
    return deserializeHelper(nodes);
}

private TreeNode deserializeHelper(Queue<String> nodes) {
    String val = nodes.poll();
    if (val.equals("#")) return null;
    TreeNode node = new TreeNode(Integer.parseInt(val));
    node.left = deserializeHelper(nodes);
    node.right = deserializeHelper(nodes);
    return node;
}
```

### ✅ Day 17 Checklist
- [ ] Why use `Long.MIN_VALUE`/`Long.MAX_VALUE` instead of `Integer.MIN_VALUE`/`Integer.MAX_VALUE` for BST validation?
- [ ] What is the time complexity of BST operations on a skewed tree?
- [ ] Trace serialize/deserialize on a 3-node tree

---

# 🗓️ DAY 18 — Graph: BFS/DFS Basics

> 💡 **Core Insight:** A grid IS a graph. Each cell is a node; its 4 neighbors are edges. Mark visited IN-PLACE to save space.

### Grid DFS Template
```java
int rows = grid.length, cols = grid[0].length;
int[] dr = {0, 0, 1, -1};
int[] dc = {1, -1, 0, 0};

void dfs(char[][] grid, int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    if (grid[r][c] != '1') return;
    grid[r][c] = '0'; // mark visited
    for (int d = 0; d < 4; d++) {
        dfs(grid, r + dr[d], c + dc[d]);
    }
}
```

### 🟡 Problem 1: Number of Islands
- **Approach:** For each unvisited '1', DFS to mark the whole island, increment counter.
- **Complexity:** O(M×N) time | O(M×N) space

```java
public int numIslands(char[][] grid) {
    int count = 0;
    for (int r = 0; r < grid.length; r++) {
        for (int c = 0; c < grid[0].length; c++) {
            if (grid[r][c] == '1') {
                dfs(grid, r, c);
                count++;
            }
        }
    }
    return count;
}

private void dfs(char[][] grid, int r, int c) {
    if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] != '1') {
        return;
    }
    grid[r][c] = '0';
    dfs(grid, r + 1, c);
    dfs(grid, r - 1, c);
    dfs(grid, r, c + 1);
    dfs(grid, r, c - 1);
}
```

### 🟡 Problem 2: Max Area of Island
- **Approach:** DFS returns the size of the island it explored.
- **Complexity:** O(M×N) time | O(M×N) space

```java
public int maxAreaOfIsland(int[][] grid) {
    int maxArea = 0;
    for (int r = 0; r < grid.length; r++) {
        for (int c = 0; c < grid[0].length; c++) {
            maxArea = Math.max(maxArea, dfs(grid, r, c));
        }
    }
    return maxArea;
}

private int dfs(int[][] grid, int r, int c) {
    if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] == 0) {
        return 0;
    }
    grid[r][c] = 0;
    return 1 + dfs(grid, r + 1, c) + dfs(grid, r - 1, c) +
           dfs(grid, r, c + 1) + dfs(grid, r, c - 1);
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

```java
public Node cloneGraph(Node node) {
    if (node == null) return null;
    
    Map<Node, Node> cloned = new HashMap<>();
    Queue<Node> q = new LinkedList<>();
    q.offer(node);
    cloned.put(node, new Node(node.val));
    
    while (!q.isEmpty()) {
        Node cur = q.poll();
        for (Node neighbor : cur.neighbors) {
            if (!cloned.containsKey(neighbor)) {
                cloned.put(neighbor, new Node(neighbor.val));
                q.offer(neighbor);
            }
            cloned.get(cur).neighbors.add(cloned.get(neighbor));
        }
    }
    
    return cloned.get(node);
}
```

### 🟡 Problem 2: Pacific Atlantic Water Flow
- **Approach:** Multi-source BFS. Start from Pacific border, then Atlantic border. Answer = intersection.
- **Complexity:** O(M×N) time | O(M×N) space

```java
public List<List<Integer>> pacificAtlantic(int[][] heights) {
    int m = heights.length, n = heights[0].length;
    boolean[][] pac = new boolean[m][n];
    boolean[][] atl = new boolean[m][n];
    
    Queue<int[]> pq = new LinkedList<>();
    Queue<int[]> aq = new LinkedList<>();
    
    for (int i = 0; i < m; i++) {
        pac[i][0] = true;
        pq.offer(new int[]{i, 0});
        atl[i][n - 1] = true;
        aq.offer(new int[]{i, n - 1});
    }
    
    for (int j = 0; j < n; j++) {
        pac[0][j] = true;
        pq.offer(new int[]{0, j});
        atl[m - 1][j] = true;
        aq.offer(new int[]{m - 1, j});
    }
    
    bfs(heights, pq, pac);
    bfs(heights, aq, atl);
    
    List<List<Integer>> res = new ArrayList<>();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (pac[i][j] && atl[i][j]) {
                res.add(Arrays.asList(i, j));
            }
        }
    }
    
    return res;
}

private void bfs(int[][] heights, Queue<int[]> q, boolean[][] visited) {
    int[] dr = {0, 0, 1, -1};
    int[] dc = {1, -1, 0, 0};
    int m = heights.length, n = heights[0].length;
    
    while (!q.isEmpty()) {
        int[] cell = q.poll();
        int r = cell[0], c = cell[1];
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && 
                !visited[nr][nc] && heights[nr][nc] >= heights[r][c]) {
                visited[nr][nc] = true;
                q.offer(new int[]{nr, nc});
            }
        }
    }
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
```java
int[] indegree = new int[n];
// Build adj and indegree from edges...

Queue<Integer> q = new LinkedList<>();
for (int i = 0; i < n; i++) {
    if (indegree[i] == 0) q.offer(i);
}

List<Integer> order = new ArrayList<>();
while (!q.isEmpty()) {
    int node = q.poll();
    order.add(node);
    for (int neighbor : adj.get(node)) {
        if (--indegree[neighbor] == 0) {
            q.offer(neighbor);
        }
    }
}
// order.size() == n → no cycle
```

### 🟡 Problem 1: Course Schedule I
- **Answer:** Can you finish all courses? = Does a valid topological order exist?
- **Complexity:** O(V+E) time | O(V+E) space

```java
public boolean canFinish(int numCourses, int[][] prerequisites) {
    List<List<Integer>> adj = new ArrayList<>();
    int[] indegree = new int[numCourses];
    
    for (int i = 0; i < numCourses; i++) {
        adj.add(new ArrayList<>());
    }
    
    for (int[] prereq : prerequisites) {
        adj.get(prereq[1]).add(prereq[0]);
        indegree[prereq[0]]++;
    }
    
    Queue<Integer> q = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) q.offer(i);
    }
    
    int count = 0;
    while (!q.isEmpty()) {
        int node = q.poll();
        count++;
        for (int neighbor : adj.get(node)) {
            if (--indegree[neighbor] == 0) {
                q.offer(neighbor);
            }
        }
    }
    
    return count == numCourses;
}
```

### 🟡 Problem 2: Course Schedule II
- **Answer:** Return the actual topological order, or empty if cycle.
- **Complexity:** O(V+E) time | O(V+E) space

```java
public int[] findOrder(int numCourses, int[][] prerequisites) {
    List<List<Integer>> adj = new ArrayList<>();
    int[] indegree = new int[numCourses];
    
    for (int i = 0; i < numCourses; i++) {
        adj.add(new ArrayList<>());
    }
    
    for (int[] prereq : prerequisites) {
        adj.get(prereq[1]).add(prereq[0]);
        indegree[prereq[0]]++;
    }
    
    Queue<Integer> q = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) q.offer(i);
    }
    
    int[] order = new int[numCourses];
    int idx = 0;
    
    while (!q.isEmpty()) {
        int node = q.poll();
        order[idx++] = node;
        for (int neighbor : adj.get(node)) {
            if (--indegree[neighbor] == 0) {
                q.offer(neighbor);
            }
        }
    }
    
    return idx == numCourses ? order : new int[]{};
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

```java
public int networkDelayTime(int[][] times, int n, int k) {
    List<List<int[]>> adj = new ArrayList<>();
    for (int i = 0; i <= n; i++) {
        adj.add(new ArrayList<>());
    }
    
    for (int[] time : times) {
        adj.get(time[0]).add(new int[]{time[1], time[2]});
    }
    
    int[] dist = new int[n + 1];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[k] = 0;
    
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
    pq.offer(new int[]{0, k});
    
    while (!pq.isEmpty()) {
        int[] curr = pq.poll();
        int d = curr[0], u = curr[1];
        
        if (d > dist[u]) continue;
        
        for (int[] edge : adj.get(u)) {
            int v = edge[0], w = edge[1];
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.offer(new int[]{dist[v], v});
            }
        }
    }
    
    int ans = 0;
    for (int i = 1; i <= n; i++) {
        if (dist[i] == Integer.MAX_VALUE) return -1;
        ans = Math.max(ans, dist[i]);
    }
    
    return ans;
}
```

### 🟡 Problem 2: Accounts Merge (Union-Find)
- **Approach:** Union all emails belonging to the same account. Group by root.
- **Complexity:** O(N α(N)) ≈ O(N) time

```java
class UnionFind {
    int[] parent, rank;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // path compression
        }
        return parent[x];
    }
    
    public void union(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        if (rank[px] < rank[py]) {
            int temp = px; px = py; py = temp;
        }
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
    }
}

public List<List<String>> accountsMerge(List<List<String>> accounts) {
    Map<String, Integer> emailToId = new HashMap<>();
    Map<String, String> emailToName = new HashMap<>();
    int id = 0;
    
    for (List<String> account : accounts) {
        String name = account.get(0);
        for (int i = 1; i < account.size(); i++) {
            String email = account.get(i);
            if (!emailToId.containsKey(email)) {
                emailToId.put(email, id++);
                emailToName.put(email, name);
            }
        }
    }
    
    UnionFind uf = new UnionFind(id);
    for (List<String> account : accounts) {
        int firstId = emailToId.get(account.get(1));
        for (int i = 2; i < account.size(); i++) {
            uf.union(firstId, emailToId.get(account.get(i)));
        }
    }
    
    Map<Integer, List<String>> groups = new HashMap<>();
    for (String email : emailToId.keySet()) {
        int root = uf.find(emailToId.get(email));
        groups.putIfAbsent(root, new ArrayList<>());
        groups.get(root).add(email);
    }
    
    List<List<String>> res = new ArrayList<>();
    for (List<String> emails : groups.values()) {
        Collections.sort(emails);
        List<String> account = new ArrayList<>();
        account.add(emailToName.get(emails.get(0)));
        account.addAll(emails);
        res.add(account);
    }
    
    return res;
}
```

### ✅ Day 21 Checklist
- [ ] Why do we skip stale Dijkstra entries with `if (d > dist[u]) continue`?
- [ ] What is α(N) (inverse Ackermann)? Why is it "almost O(1)"?
- [ ] What is the difference between path compression and union by rank?

---

# 🗓️ DAY 22 — Heaps: Top K Patterns

> 💡 **Core Insight:** To find K largest elements, keep a MIN-heap of size K. The smallest in the heap is the Kth largest overall.

### Java Heap Quick Reference
```java
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]); // min by first element
```

### 🟡 Problem 1: Kth Largest Element in an Array
- **Approach:** Min-heap of size K. If size > K, poll the min.
- **Complexity:** O(N log K) time | O(K) space

```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    
    for (int num : nums) {
        minHeap.offer(num);
        if (minHeap.size() > k) {
            minHeap.poll();
        }
    }
    
    return minHeap.peek();
}
```

### 🟡 Problem 2: Top K Frequent Elements
- **Approach:** Frequency map + min-heap on `(freq, element)` of size K.
- **Complexity:** O(N log K) time | O(N) space

```java
public int[] topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> freq = new HashMap<>();
    for (int num : nums) {
        freq.put(num, freq.getOrDefault(num, 0) + 1);
    }
    
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
    
    for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
        pq.offer(new int[]{entry.getValue(), entry.getKey()});
        if (pq.size() > k) {
            pq.poll();
        }
    }
    
    int[] res = new int[k];
    int idx = 0;
    while (!pq.isEmpty()) {
        res[idx++] = pq.poll()[1];
    }
    
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

```java
class MedianFinder {
    private PriorityQueue<Integer> lower; // max-heap
    private PriorityQueue<Integer> upper; // min-heap
    
    public MedianFinder() {
        lower = new PriorityQueue<>(Collections.reverseOrder());
        upper = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        lower.offer(num);
        upper.offer(lower.poll());
        if (lower.size() < upper.size()) {
            lower.offer(upper.poll());
        }
    }
    
    public double findMedian() {
        if (lower.size() > upper.size()) {
            return lower.peek();
        }
        return (lower.peek() + upper.peek()) / 2.0;
    }
}
```

### 🟡 Problem 2: Task Scheduler
- **Approach:** Most frequent task determines idle time. Use a max-heap + cooldown queue.
- **Complexity:** O(N log N) time

```java
public int leastInterval(char[] tasks, int n) {
    Map<Character, Integer> freq = new HashMap<>();
    for (char task : tasks) {
        freq.put(task, freq.getOrDefault(task, 0) + 1);
    }
    
    PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
    pq.addAll(freq.values());
    
    int time = 0;
    Queue<int[]> cooldown = new LinkedList<>();
    
    while (!pq.isEmpty() || !cooldown.isEmpty()) {
        time++;
        
        if (!cooldown.isEmpty() && cooldown.peek()[1] <= time) {
            pq.offer(cooldown.poll()[0]);
        }
        
        if (!pq.isEmpty()) {
            int f = pq.poll() - 1;
            if (f > 0) {
                cooldown.offer(new int[]{f, time + n + 1});
            }
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

> 💡 **Core Insight:** At each index, make a binary decision: include or exclude. The recursion tree has 2^N leaves. Undo your choice (remove) before the next branch.

### Universal Template
```java
void backtrack(int start, List<Integer> current, List<List<Integer>> result, int[] nums) {
    result.add(new ArrayList<>(current));  // collect at every node
    for (int i = start; i < nums.length; i++) {
        current.add(nums[i]);              // choose
        backtrack(i + 1, current, result, nums); // explore
        current.remove(current.size() - 1); // unchoose
    }
}
```

### 🟡 Problem 1: Subsets
- **Complexity:** O(2^N × N) time | O(N) space (recursion)

```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    backtrack(0, new ArrayList<>(), res, nums);
    return res;
}

private void backtrack(int start, List<Integer> curr, List<List<Integer>> res, int[] nums) {
    res.add(new ArrayList<>(curr));
    for (int i = start; i < nums.length; i++) {
        curr.add(nums[i]);
        backtrack(i + 1, curr, res, nums);
        curr.remove(curr.size() - 1);
    }
}
```

### 🟡 Problem 2: Combination Sum
- **Approach:** Same element can be reused — don't increment `start` when recursing.
- **Complexity:** O(N^(T/M)) where T=target, M=min element

```java
public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(candidates);
    backtrack(0, target, new ArrayList<>(), res, candidates);
    return res;
}

private void backtrack(int start, int remaining, List<Integer> curr, 
                      List<List<Integer>> res, int[] candidates) {
    if (remaining == 0) {
        res.add(new ArrayList<>(curr));
        return;
    }
    
    for (int i = start; i < candidates.length; i++) {
        if (candidates[i] > remaining) break; // prune
        curr.add(candidates[i]);
        backtrack(i, remaining - candidates[i], curr, res, candidates); // i not i+1
        curr.remove(curr.size() - 1);
    }
}
```

### ✅ Day 24 Checklist
- [ ] What does `remove(size - 1)` undo exactly?
- [ ] Why does Combination Sum pass `i` not `i+1` to allow reuse?
- [ ] Subsets: how many total subsets for an array of size 4?

---

# 🗓️ DAY 25 — Backtracking: Permutations & Grid

> 💡 **Core Insight:** Permutations use a `used[]` array to track which elements are taken. Grid DFS marks cells in-place to simulate the "used" state.

### 🟡 Problem 1: Permutations
- **Complexity:** O(N! × N) time

```java
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    backtrack(new ArrayList<>(), res, nums, new boolean[nums.length]);
    return res;
}

private void backtrack(List<Integer> curr, List<List<Integer>> res, 
                      int[] nums, boolean[] used) {
    if (curr.size() == nums.length) {
        res.add(new ArrayList<>(curr));
        return;
    }
    
    for (int i = 0; i < nums.length; i++) {
        if (used[i]) continue;
        used[i] = true;
        curr.add(nums[i]);
        backtrack(curr, res, nums, used);
        curr.remove(curr.size() - 1);
        used[i] = false;
    }
}
```

### 🟡 Problem 2: Word Search
- **Approach:** DFS on grid. Mark cell with a special character before recursing, restore after.
- **Complexity:** O(M×N×4^L) time | O(L) space

```java
public boolean exist(char[][] board, String word) {
    for (int r = 0; r < board.length; r++) {
        for (int c = 0; c < board[0].length; c++) {
            if (dfs(board, word, r, c, 0)) {
                return true;
            }
        }
    }
    return false;
}

private boolean dfs(char[][] board, String word, int r, int c, int idx) {
    if (idx == word.length()) return true;
    if (r < 0 || r >= board.length || c < 0 || c >= board[0].length || 
        board[r][c] != word.charAt(idx)) {
        return false;
    }
    
    char tmp = board[r][c];
    board[r][c] = '#';
    
    boolean found = dfs(board, word, r + 1, c, idx + 1) ||
                    dfs(board, word, r - 1, c, idx + 1) ||
                    dfs(board, word, r, c + 1, idx + 1) ||
                    dfs(board, word, r, c - 1, idx + 1);
    
    board[r][c] = tmp; // restore
    return found;
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

```java
public int climbStairs(int n) {
    if (n <= 2) return n;
    int a = 1, b = 2;
    for (int i = 3; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}
```

### 🟡 Problem 2: House Robber
- `dp[i]` = max money robbing from houses 0..i
- Recurrence: `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
- **Complexity:** O(N) time | O(1) space

```java
public int rob(int[] nums) {
    int prev2 = 0, prev1 = 0;
    for (int num : nums) {
        int cur = Math.max(prev1, prev2 + num);
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

```java
public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (i >= coin) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    
    return dp[amount] > amount ? -1 : dp[amount];
}
```

### ✅ Day 26 Checklist
- [ ] What DP pattern do Climbing Stairs and House Robber share?
- [ ] Why initialize `dp[i] = amount + 1` (not Integer.MAX_VALUE) in Coin Change?
- [ ] All three problems solved from memory?

---

# 🗓️ DAY 27 — DP: Strings & Subsequences

> 💡 **Core Insight:** For subsequence problems on two strings, the DP table is 2D. `dp[i][j]` captures the relationship between the first i chars of s1 and first j chars of s2.

### 🟡 Problem 1: Longest Increasing Subsequence
- `dp[i]` = length of LIS ending at index i
- **O(N²) approach**, then O(N log N) with patience sort (binary search)

```java
// O(N log N) — patience sort with binary search
public int lengthOfLIS(int[] nums) {
    List<Integer> tails = new ArrayList<>();
    
    for (int num : nums) {
        int pos = binarySearch(tails, num);
        if (pos == tails.size()) {
            tails.add(num);
        } else {
            tails.set(pos, num);
        }
    }
    
    return tails.size();
}

private int binarySearch(List<Integer> tails, int target) {
    int lo = 0, hi = tails.size();
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (tails.get(mid) < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}
```

### 🟡 Problem 2: Longest Common Subsequence
- `dp[i][j]` = LCS of first i chars of text1 and first j chars of text2
- Recurrence: if chars match → `dp[i-1][j-1] + 1`, else `max(dp[i-1][j], dp[i][j-1])`
- **Complexity:** O(M×N) time | O(M×N) space

```java
public int longestCommonSubsequence(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    
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

```java
public int knapsack(int[] weights, int[] values, int W) {
    int n = weights.length;
    int[] dp = new int[W + 1];
    
    for (int i = 0; i < n; i++) {
        for (int w = W; w >= weights[i]; w--) {  // BACKWARDS!
            dp[w] = Math.max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }
    
    return dp[W];
}
```

### 🟡 Problem 2: Unique Paths
- `dp[i][j]` = number of paths to reach cell (i, j)
- Recurrence: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`
- **Complexity:** O(M×N) time | O(N) space

```java
public int uniquePaths(int m, int n) {
    int[] dp = new int[n];
    Arrays.fill(dp, 1);
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[j] += dp[j - 1];
        }
    }
    
    return dp[n - 1];
}
```

### 🔴 Problem 3: Edit Distance
- `dp[i][j]` = min edits to convert word1[0..i] to word2[0..j]
- **Complexity:** O(M×N) time | O(M×N) space

```java
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + Math.min(dp[i - 1][j],
                           Math.min(dp[i][j - 1], dp[i - 1][j - 1]));
            }
        }
    }
    
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

```java
class TrieNode {
    TrieNode[] children = new TrieNode[26];
    boolean isEnd = false;
}

class Trie {
    private TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null) {
                node.children[idx] = new TrieNode();
            }
            node = node.children[idx];
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null) {
                return false;
            }
            node = node.children[idx];
        }
        return node.isEnd;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char c : prefix.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null) {
                return false;
            }
            node = node.children[idx];
        }
        return true;
    }
}
```

### 🟡 Problem 2: LRU Cache
- `LinkedHashMap` OR `HashMap + Doubly Linked List`
- **Complexity:** O(1) get and put

```java
class LRUCache {
    private int capacity;
    private Map<Integer, Integer> map;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.map = new LinkedHashMap<>(capacity, 0.75f, true) {
            protected boolean removeEldestEntry(Map.Entry eldest) {
                return size() > capacity;
            }
        };
    }
    
    public int get(int key) {
        return map.getOrDefault(key, -1);
    }
    
    public void put(int key, int value) {
        map.put(key, value);
    }
}

// Alternative: Manual Doubly Linked List implementation
class LRUCacheManual {
    class Node {
        int key, val;
        Node prev, next;
        Node(int k, int v) { key = k; val = v; }
    }
    
    private int capacity;
    private Map<Integer, Node> map;
    private Node head, tail;
    
    public LRUCacheManual(int capacity) {
        this.capacity = capacity;
        this.map = new HashMap<>();
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }
    
    private void remove(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    private void insert(Node node) {
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
    }
    
    public int get(int key) {
        if (!map.containsKey(key)) return -1;
        Node node = map.get(key);
        remove(node);
        insert(node);
        return node.val;
    }
    
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            remove(map.get(key));
        }
        Node node = new Node(key, value);
        map.put(key, node);
        insert(node);
        if (map.size() > capacity) {
            Node lru = tail.prev;
            remove(lru);
            map.remove(lru.key);
        }
    }
}
```

### ✅ Day 29 Checklist
- [ ] Why use doubly linked list for LRU? What's special about it?
- [ ] What happens to the Trie's memory? Who cleans up TrieNode? (GC in Java)
- [ ] Can you implement Trie with `HashMap` instead of array? Trade-offs?

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
├── K-th / Top-K / Scheduling?            → Heap / PriorityQueue
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
| HashMap ops | O(1) avg | O(N) |
| Binary Search | O(log N) | O(1) |
| DFS/BFS Tree | O(N) | O(H) |
| DFS/BFS Graph | O(V+E) | O(V) |
| Dijkstra's | O(E log V) | O(V) |
| Topo Sort (Kahn's) | O(V+E) | O(V) |
| Union-Find | O(α(N)) | O(N) |
| Heap push/poll | O(log N) | O(N) |
| DP 1D | O(N) | O(1)–O(N) |
| DP 2D | O(M×N) | O(M×N) |
| Backtracking | O(2^N) or O(N!) | O(N) |
| Trie ops | O(L) | O(N×L) |

---

## Java Collections Pitfalls

```java
// ❌ Overflow in binary search
int mid = (lo + hi) / 2;      // BAD — overflow if lo+hi > Integer.MAX_VALUE
int mid = lo + (hi - lo) / 2; // ✅ GOOD

// ❌ ConcurrentModificationException
for (Integer key : map.keySet()) {
    map.remove(key); // BAD — modifying while iterating
}
// ✅ Use Iterator or collect keys first
Iterator<Integer> it = map.keySet().iterator();
while (it.hasNext()) {
    it.next();
    it.remove();
}

// ❌ Autoboxing performance trap
Map<Integer, Integer> map = new HashMap<>();
for (int i = 0; i < 1000000; i++) {
    map.put(i, i); // Creates Integer objects — slow
}

// ❌ Integer overflow
if (sum + num > Integer.MAX_VALUE) // BAD — sum+num may overflow first
if (num > Integer.MAX_VALUE - sum) // ✅ GOOD — rearrange

// ❌ Array to List pitfall
int[] arr = {1, 2, 3};
List<Integer> list = Arrays.asList(arr); // BAD — creates List<int[]>
List<Integer> list = Arrays.stream(arr).boxed().collect(Collectors.toList()); // ✅ GOOD

// ❌ PriorityQueue doesn't maintain order during iteration
PriorityQueue<Integer> pq = new PriorityQueue<>();
pq.addAll(Arrays.asList(3, 1, 2));
for (int x : pq) System.out.print(x); // Prints 1, 3, 2 (heap order, not sorted)
// ✅ Use poll() to get elements in priority order
```

---

*30 days. 1 pattern at a time. You've got this.*