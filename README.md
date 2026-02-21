# 🧠 DSA Daily Grind — 30-Day Interview Prep

> *"Consistency beats intensity — every single time."*

A structured **30-day DSA habit system** targeting **FAANG & top MNC interviews** — built around pattern recognition, daily accountability, and the [Striver's A2Z DSA Sheet](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2).

---

## ⚡ Core Philosophy

This repo is not a problem dump. It's a **habit system**.

- **Pattern recognition > memorization** — understand *why* a solution works, not just *how*
- **Show up daily** — even 30 minutes beats a 5-hour cramming weekend
- **Reflect & share** — public accountability turns practice into identity

---

## 🗓️ The Roadmap

A full 30-day roadmap is available — covering 4 structured weeks plus a final Gauntlet:

- **Week 1** — Arrays, Two Pointers, Sliding Window, Hashing
- **Week 2** — Linked Lists, Stacks, Queues, Binary Search
- **Week 3** — Trees, Graphs, Dijkstra's, Topological Sort, Union-Find
- **Week 4** — Heaps, Backtracking, Dynamic Programming
- **Days 29–30** — Trie, LRU Cache, and a full Mock Interview

> 📄 See [`ROADMAP.md`](./ROADMAP.md) for the complete day-by-day breakdown with problems, pseudocode, and deep-dives.

---

## 📋 Global Rules

| Rule | Detail |
|------|--------|
| 🗣️ **Talk out loud** | Narrate every thought — simulate the real interview |
| ⏱️ **Complexity first** | State Time & Space complexity **before** writing any code |
| 🔄 **Stuck > 45 min** | Read the optimal approach → close it → recode from memory |
| 📝 **Pattern note** | After every problem: write a 2-sentence insight in your own words |
| ⚠️ **Edge cases always** | Empty input, single element, overflow/underflow |
| 🛠️ **Use STL / stdlib** | Prefer built-in structures — know their internals |

---

## 🗺️ Pattern Recognition Cheatsheet

```
What you see in the problem              →   Pattern to reach for
─────────────────────────────────────────────────────────────────
Range sum / subarray query               →   Prefix Sum
Find pair in sorted array                →   Two Pointers (converging)
Longest / shortest substring             →   Sliding Window
Frequency, grouping, lookup              →   HashMap / HashSet
Linked list cycle / middle               →   Fast & Slow Pointers
Next greater / smaller element           →   Monotonic Stack
Search in sorted / rotated array         →   Binary Search
Minimize the maximum (answer range)      →   Binary Search on Answer Space
Tree height / path / diameter            →   DFS Post-order
Level-by-level tree traversal            →   BFS
Shortest path (unweighted graph)         →   BFS
Shortest path (weighted graph)           →   Dijkstra's
Ordering with dependencies / cycle       →   Topological Sort (Kahn's)
Dynamic connectivity / group merge       →   Union-Find (DSU)
K-th largest / Top-K / median            →   Heap / Priority Queue
All combinations / permutations          →   Backtracking
Optimal substructure + overlapping subs  →   Dynamic Programming
Prefix lookup / autocomplete             →   Trie
O(1) cache with recency eviction         →   HashMap + Doubly Linked List
```

---

## 📁 Folder Structure

```
📦 DSA-Daily-Grind/
 ┣ 📂 your-github-username/
 ┃ ┣ 📂 day-01/
 ┃ ┃ ┣ 📜 README.md          ← daily plan + reflection
 ┃ ┃ ┗ 📜 solution files
 ┃ ┣ 📂 day-02/
 ┃ ┃ ┗ ...
 ┃ ┗ 📜 streak-log.md        ← running progress tracker
 ┣ 📜 ROADMAP.md             ← full 30-day breakdown
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
- Open `ROADMAP.md` and pick the day's problems
- **State complexity before coding**
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

> Watch your contribution graph fill up. Your streak is visible proof of your discipline.

---

## 📊 Streak Log Template

Track your progress in `your-username/streak-log.md`:

```markdown
# My DSA Streak Log

| Day | Date | Topic | Problems Solved | Time | Streak |
|-----|------|-------|-----------------|------|--------|
| 1   | YYYY-MM-DD | Prefix Sums | Product of Array, Subarray Sum K | 90 min | 🔥 1 |
| 2   | YYYY-MM-DD | Two Pointers | Two Sum II, Container With Most Water | 75 min | 🔥 2 |

## Pattern Mastery
- [ ] Prefix Sum
- [ ] Two Pointers
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

**Post template:**
> "Day X of #DSADailyGrind 🔥 — Solved [Problem] using [Pattern]. Key insight: [one sentence]. Streak: X days 💪 [repo link]"

---

## 🚀 Getting Started

```bash
# 1. Fork this repo
# 2. Clone it
git clone https://github.com/YOUR_USERNAME/DSA-Daily-Grind.git

# 3. Create your folder
mkdir your-github-username && cd your-github-username

# 4. Start Day 1
mkdir day-01
# Write your plan in README.md, then start grinding
```

---

## 📚 Resources

| Resource | Link |
|----------|------|
| Striver's A2Z DSA Sheet | [takeuforward.org →](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2) |
| LeetCode | [leetcode.com →](https://leetcode.com) |
| NeetCode 150 | [neetcode.io →](https://neetcode.io) |
| C++ STL Reference | [cppreference.com →](https://cppreference.com) |
| Pramp Mock Interviews | [pramp.com →](https://pramp.com) |

---

<div align="center">

**30 days. 1 pattern at a time.**

*You're not just solving problems — you're building the identity of a consistent engineer.*

⭐ Star this repo if it helps you stay consistent

</div>