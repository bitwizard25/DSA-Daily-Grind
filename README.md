# 🧠 DSA Daily Grind — 30-Day C++ Interview Prep

> *"Consistency beats intensity — every single time."*

A structured **30-day DSA habit system** for C++ developers targeting **FAANG & top MNC interviews** — built around pattern recognition, daily accountability, and the [Striver's A2Z DSA Sheet](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2).

---

## ⚡ The Core Philosophy

This repo is not a problem dump. It's a **habit system**.

- **Pattern recognition > memorization** — understand *why* a solution works, not just *how*
- **Show up daily** — even 30 minutes beats a 5-hour cramming weekend
- **Reflect & share** — public accountability turns practice into identity

---

## 📋 Global Rules

| Rule | Detail |
|------|--------|
| 🗣️ **Talk out loud** | Narrate every thought — simulate the real interview |
| ⏱️ **Complexity first** | State Time & Space complexity **before** writing any code |
| 🔄 **Stuck > 45 min** | Read the optimal approach → close it → recode from memory |
| 📝 **Pattern note** | After every problem: write a 2-sentence insight in your own words |
| ⚠️ **Edge cases always** | Empty input, single element, `INT_MIN`/`INT_MAX` overflow |
| 🛠️ **Use STL** | Prefer `vector`, `unordered_map`, `priority_queue` — know their internals |

---

## 🗓️ The 30-Day Roadmap

### Week 1 — Primitives, Pointers & Windows
*Master O(N) single-pass techniques. No nested loops.*

| Day | Topic | Problems |
|-----|-------|----------|
| **1** | Array Basics & Prefix Sums | Product of Array Except Self · Subarray Sum Equals K |
| **2** | Two Pointers — Converging | Two Sum II · Container With Most Water |
| **3** | Two Pointers — Advanced | 3Sum · Trapping Rain Water |
| **4** | Sliding Window — Fixed | Maximum Average Subarray I · Max Sum Subarray of Size K |
| **5** | Sliding Window — Variable | Longest Substring Without Repeating · Minimum Window Substring |
| **6** | Hashing — Maps | Two Sum · Group Anagrams |
| **7** | Hashing — Sets & Review | Longest Consecutive Sequence · Week 1 Full Revision |

---

### Week 2 — Structures & Memory
*Manage state with LIFO/FIFO and pointer reassignment.*

| Day | Topic | Problems |
|-----|-------|----------|
| **8** | Linked Lists — Basics | Reverse Linked List · Middle of the Linked List |
| **9** | Linked Lists — Advanced | Linked List Cycle (Floyd's) · Merge K Sorted Lists |
| **10** | Stacks | Valid Parentheses · Min Stack |
| **11** | Monotonic Stacks | Daily Temperatures · Largest Rectangle in Histogram |
| **12** | Queues & Deques | Implement Queue using Stacks · Sliding Window Maximum |
| **13** | Binary Search — Classic | Binary Search · Search in Rotated Sorted Array |
| **14** | Binary Search — Answer Space | Koko Eating Bananas · Find Minimum in Rotated Sorted Array |

---

### Week 3 — Trees, Graphs & Hierarchies
*Recursive thinking and traversing complex networks.*

| Day | Topic | Problems |
|-----|-------|----------|
| **15** | Binary Trees — DFS | Max Depth · Path Sum · Diameter of Binary Tree |
| **16** | Binary Trees — BFS & Views | Level Order Traversal · Right Side View |
| **17** | BST | Validate BST · LCA of BST · Serialize & Deserialize |
| **18** | Graph — BFS/DFS Basics | Number of Islands · Max Area of Island |
| **19** | Graph — Clone & Multi-Source | Clone Graph · Pacific Atlantic Water Flow |
| **20** | Graph — Topological Sort | Course Schedule I · Course Schedule II |
| **21** | Graph — Dijkstra & Union-Find | Network Delay Time · Accounts Merge |

---

### Week 4 — Optimization & Simulation
*Explore state spaces and make optimal choices.*

| Day | Topic | Problems |
|-----|-------|----------|
| **22** | Heaps — Top K Patterns | Kth Largest Element · Top K Frequent Elements |
| **23** | Heaps — Advanced | Find Median from Data Stream · Task Scheduler |
| **24** | Backtracking — Subsets & Combos | Subsets · Combination Sum |
| **25** | Backtracking — Permutations & Grid | Permutations · Word Search |
| **26** | DP — 1D Foundations | Climbing Stairs · House Robber · Coin Change |
| **27** | DP — Strings & Subsequences | Longest Increasing Subsequence · LCS |
| **28** | DP — 2D & Knapsack | 0/1 Knapsack · Unique Paths · Edit Distance |

---

### The Gauntlet — Integration
*Build from scratch. Perform under pressure.*

| Day | Topic | Problems |
|-----|-------|----------|
| **29** | System-Level DSA | Implement Trie · LRU Cache |
| **30** | 🎯 Mock Interview Day | 4 random unseen Medium/Hard problems · 35 min each |

---

## 🗺️ Pattern Recognition Cheatsheet

```
What you see in the problem          →   Pattern to reach for
─────────────────────────────────────────────────────────────
Range sum / subarray query           →   Prefix Sum
Find pair in sorted array            →   Two Pointers (converging)
Longest / shortest substring         →   Sliding Window
Frequency, grouping, lookup          →   HashMap / HashSet
Linked list cycle / middle           →   Fast & Slow Pointers
Next greater / smaller element       →   Monotonic Stack
Search in sorted / rotated array     →   Binary Search
Minimize the maximum (answer range)  →   Binary Search on Answer Space
Tree height / path / diameter        →   DFS Post-order
Level-by-level tree traversal        →   BFS
Shortest path (unweighted graph)     →   BFS
Shortest path (weighted graph)       →   Dijkstra's
Ordering with dependencies / cycle   →   Topological Sort (Kahn's)
Dynamic connectivity / group merge   →   Union-Find (DSU)
K-th largest / Top-K / median        →   Heap / Priority Queue
All combinations / permutations      →   Backtracking
Optimal substructure + overlapping   →   Dynamic Programming
Prefix lookup / autocomplete         →   Trie
O(1) cache with recency eviction     →   HashMap + Doubly Linked List
```

---

## 📁 Folder Structure

```
📦 DSA-Daily-Grind/
 ┣ 📂 your-github-username/
 ┃ ┣ 📂 day-01/
 ┃ ┃ ┣ 📜 README.md          ← daily plan + reflection
 ┃ ┃ ┣ 📜 product_except_self.cpp
 ┃ ┃ ┗ 📜 subarray_sum_k.cpp
 ┃ ┣ 📂 day-02/
 ┃ ┃ ┗ ...
 ┃ ┗ 📜 streak-log.md        ← running progress tracker
 ┗ 📜 README.md              ← this file
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
- Pick the day's problems from the roadmap above
- **State complexity before coding**
- Write clean C++ using STL idioms
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

> Watch your contribution graph fill up. Your **DSA streak** is visible proof of your discipline.

---

## 📊 Streak Log Template

Track your progress in `your-username/streak-log.md`:

```markdown
# My DSA Streak Log

| Day | Date | Topic | Problems Solved | Time | Streak |
|-----|------|-------|-----------------|------|--------|
| 1   | YYYY-MM-DD | Prefix Sums | Product of Array, Subarray Sum K | 90 min | 🔥 1 |
| 2   | YYYY-MM-DD | Two Pointers | Two Sum II, Container With Most Water | 75 min | 🔥 2 |
...

## Striver A2Z Progress
- [x] Arrays Level 1
- [x] Arrays Level 2
- [ ] Binary Search
- [ ] Strings
- [ ] Recursion
- [ ] Linked List
- [ ] Stack & Queue
- [ ] Trees
- [ ] Graphs
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

## 🤝 Community Rules

| Action | Why |
|--------|-----|
| Browse peers' folders | See different approaches — learn by example |
| Comment & help debug | Teaching a concept locks it in for you |
| Celebrate milestones | Week completions, first Hard solved, full streak |
| Share on LinkedIn | Public commitment becomes public accountability |

**LinkedIn post template:**
> "Day X of #DSADailyGrind 🔥 — Solved [Problem] using [Pattern] in C++. Key insight: [one sentence]. Streak: X days 💪 [repo link]"

---

## 🚀 Getting Started

```bash
# 1. Fork this repo
# 2. Clone it
git clone https://github.com/YOUR_USERNAME/DSA-Daily-Grind.git

# 3. Create your folder
mkdir your-github-username
cd your-github-username

# 4. Start Day 1
mkdir day-01
# Create your README.md, write your plan, start grinding
```

---

## 📚 Resources

| Resource | Link |
|----------|------|
| Striver's A2Z DSA Sheet | [takeuforward.org →](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2) |
| LeetCode | [leetcode.com →](https://leetcode.com) |
| C++ STL Reference | [cppreference.com →](https://cppreference.com) |
| NeetCode 150 | [neetcode.io →](https://neetcode.io) |
| Pramp Mock Interviews | [pramp.com →](https://pramp.com) |

---

<div align="center">

**30 days. 1 pattern at a time.**

*You're not just solving problems — you're building the identity of a consistent engineer.*

⭐ Star this repo if it helps you stay consistent

</div>