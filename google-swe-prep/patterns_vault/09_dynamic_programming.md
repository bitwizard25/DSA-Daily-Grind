# Dynamic Programming

**Cue:** "number of ways", "min/max cost", choices that overlap, string alignment.

Recipe:
1. State: what does dp[i] (or dp[i][j]) mean in words?
2. Transition: how does it depend on smaller states?
3. Base cases.
4. Iteration order, then answer location.
5. Compress space if only the previous row is used.

- 0/1 knapsack: loop capacity descending. Unbounded: ascending.
- Interval DP (Burst Balloons): choose the *last* action in the interval.
- Strings (Edit Distance, LCS): dp over prefixes of both strings.
