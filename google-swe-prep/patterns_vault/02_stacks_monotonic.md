# Stacks and the monotonic stack

> In one line: a stack gives back the last item you put in; for "next bigger" questions, it holds the items still waiting for an answer.

## When to use it
- **Matching pairs** or nesting: brackets like `([]{})`, open and close tags, "undo the last step".
- **Next greater** or **next smaller** element: "how many days until a warmer day?"
- "The first taller person to the right", "who can see whom", "which bar blocks the view".
- **Largest rectangle** in a bar chart, or rain water trapped between bars.

## The idea in plain words
A **stack** is like a pile of plates. You put a plate on top (**push**) and take the top plate off (**pop**).
The last plate in is the first plate out: **LIFO** (last in, first out).
In Python a plain `list` is the stack: `append(x)` pushes, `pop()` pops and returns the item, `st[-1]` reads the top.

**Monotonic** means "always increasing or always decreasing".
A **monotonic stack** is a stack whose values, from bottom to top, always keep that one order.

**Next greater element**: for each number, find the first bigger number to its right.
1. Walk through the list from left to right.
2. The stack holds the **indexes** of items that still wait for their answer.
3. A new value `x` arrives. Look at the value of the index on top. If `x` is bigger, `x` is its answer.
4. Pop that index and write its answer. Repeat while the top value is smaller than `x`.
5. Push the index of `x`: now it waits too.
6. Indexes still on the stack at the end have no bigger value to the right. Their answer stays -1.

Step 4 removes every smaller value before a push. So the values on the stack always go down from bottom to top.
That is why it is called a **decreasing** monotonic stack.

Example: `nums = [2, 1, 5, 3]`.
- `2` arrives: the stack is empty, push it. Waiting values: 2.
- `1` arrives: 1 is not bigger than 2, push it. Waiting values: 2, 1.
- `5` arrives: 5 answers 1, then 5 answers 2. Push 5. Waiting values: 5.
- `3` arrives: 3 is not bigger than 5, push it. Waiting values: 5, 3.
- End: 5 and 3 never got an answer. Result: `[5, 5, -1, -1]`.

The inner `while` loop looks slow, but each index is pushed once and popped at most once.
So the whole walk is O(N): about N steps, not N × N.

## Python template

### A list as a stack

```python
stack = []
stack.append(10)
stack.append(20)
print(stack[-1])  # 20
print(stack.pop())  # 20
print(stack)  # [10]
# an empty list counts as False, so "not stack" means "the stack is empty"
print(not stack)  # False
```

### Matching brackets

```python
def is_valid(text):
    # a dict: each closing bracket -> its opening bracket
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in text:
        if ch in pairs:
            # a closing bracket must match the last opening bracket
            if not stack:
                return False
            if stack.pop() != pairs[ch]:
                return False
        else:
            stack.append(ch)
    # opening brackets that were never closed make the text invalid
    return not stack


print(is_valid("([]{})"))  # True
print(is_valid("(]"))  # False
print(is_valid("(("))  # False
```

### Monotonic stack: next greater element

```python
def next_greater(nums):
    answer = [-1] * len(nums)
    # indexes still waiting; their values go down from bottom to top
    stack = []
    for i in range(len(nums)):
        x = nums[i]
        while stack and nums[stack[-1]] < x:
            j = stack.pop()
            answer[j] = x
        stack.append(i)
    return answer


print(next_greater([2, 1, 5, 3]))  # [5, 5, -1, -1]
print(next_greater([4, 4]))  # [-1, -1]
```

### Store indexes to get a distance

Days to wait for a warmer day (0 if it never comes). The index tells you **how far** the answer is.

```python
def days_to_warmer(temps):
    answer = [0] * len(temps)
    stack = []
    for i in range(len(temps)):
        while stack and temps[stack[-1]] < temps[i]:
            j = stack.pop()
            answer[j] = i - j
        stack.append(i)
    return answer


print(days_to_warmer([73, 74, 75, 71, 69, 72, 76, 73]))  # [1, 1, 4, 2, 1, 1, 0, 0]
```

### Pick the pop rule

| You want, for each item | Pop while the top value is | Stack values, bottom to top |
|---|---|---|
| next **greater** | smaller than `x` (`<`) | going down (decreasing) |
| next **smaller** | bigger than `x` (`>`) | going up (increasing) |
| previous **greater** | smaller than or equal to `x` (`<=`) | going down |
| previous **smaller** | bigger than or equal to `x` (`>=`) | going up |

For "next": the answer is written when an item is **popped**.
For "previous": after the pops, the top item (if any) is the answer for `x`. Read it, then push `x`.

## Coming from C++
| C++ you remember | Python you write | Note |
|---|---|---|
| `stack<int> st;` | `st = []` | a list is the stack |
| `st.push(x);` | `st.append(x)` | adds on top |
| `st.top()` | `st[-1]` | index -1 is the last item |
| `int v = st.top(); st.pop();` | `v = st.pop()` | Python `pop()` returns the item |
| `st.empty()` | `not st` | an empty list counts as False |
| `st.size()` | `len(st)` | the number of items |
| `vector<int> ans(n, -1);` | `ans = [-1] * n` | n copies of -1 |
| `while (!st.empty() && a[st.top()] < x)` | `while st and a[st[-1]] < x:` | `and` stops early if `st` is empty |

## How fast is it?
| What you do | Time | In plain words |
|---|---|---|
| `st.append(x)`, `st.pop()`, `st[-1]` | O(1) | the same speed for any size |
| `st.pop(0)`, `st.insert(0, x)` | O(N) | every other item moves; a stack never needs it |
| Next greater with two nested loops | O(N²) | for each item, look at everything to its right |
| Next greater with a monotonic stack | O(N) | each index is pushed once and popped at most once |
| Matching brackets | O(N) | one pass over the text |

## Common mistakes
- Mistake: `st.top()` as in C++. A Python list has no `top`. Fix: read the top with `st[-1]`.
- Mistake: `st[-1]` or `st.pop()` on an empty list stops the program with an IndexError. Fix: check first with `while st and ...`.
- Mistake: pushing values when you need a distance or a position. Fix: push **indexes**, and read the value with `nums[i]`.
- Mistake: `if` instead of `while` for popping. One new value can answer many waiting items. Fix: use `while`.
- Mistake: mixing up `<` and `<=` in the pop rule, so equal values get the wrong answer. Fix: test with a repeat like `[4, 4]`.
- Mistake: forgetting the items left on the stack at the end. Fix: fill the answer list first with the "no answer" value (-1 or 0).

## Practice in the app
- **G-739 Daily Temperatures**: decreasing stack of indexes; when day `i` pops day `j`, the answer for `j` is `i - j`.
- **G-503 Next Greater Element II**: the list is circular (after the end comes the start), so walk it twice and use index `i % n`; push only in the first round.
- **G-1944 Number of Visible People in a Queue**: walk from right to left; a person sees everyone shorter that they pop, plus the next person who is not shorter, if there is one.
- **G-84 Largest Rectangle in Histogram**: increasing stack; when a lower bar comes, pop and measure the popped bar's rectangle.
- **G-42 Trapping Rain Water**: decreasing stack of bars; each pop fills one layer of water between two walls.
