# Google mode: engineering practice

Approved direction: add comprehension, independent retries and practical recovery to the existing Google study flow. Implement inline in the existing isolated worktree. Keep compiler execution separate.

## Contract
- Three original beginner-to-intermediate exercises, each in Python and C++: duplicate events, expiring cache, dependency ordering. Read, debug, change, verify and an optional extension. Label these practice exercises, never an official Google format.
- Persist separate attempts and drafts by exercise/language in an additive SQLModel table. Autosave with revision checks. Explicit assistance levels; showing a reference permanently records solution help for that attempt. Completion is self-reported and cannot generate DSA grades, streaks or readiness.
- Independent retry queue derives from real saved Attempt rows: failed, hinted, answer-assisted or unknown-help latest attempts, available tomorrow. A subsequent independently solved attempt clears the item. Show recovery evidence separately from feelings.
- Add one optional card to Google Today and links in Interview prep. Keep primary work and break handling intact. Failures loading the optional card must not block Today. Preserve unfinished mocks.
- Improve stuck dialog with a concrete two-minute reset and pause/save option. Preserve existing timed hint policy.
- Existing General mode, study data, updater and compiler plans stay separate.

## Tasks
1. Add backend tests for draft round trips, revision conflicts, validation, monotonic assistance, truthful completion and retry ordering; observe failures, then implement model, core module and RPCs.
2. Add three bilingual exercises and independent fixture tests proving the starter bugs and reference fixes. No learner code is executed.
3. Add responsive practice UI, explicit save/recovery feedback, assistance choices, reference disclosure, retry links and optional Today card. Exercise real shipped handlers in frontend regressions, including failed saves and navigation.
4. Run targeted and full backend suite, existing frontend suites and the new frontend suite; inspect a source preview with an isolated profile. Review diff, push feature branch and open PR. Do not merge or install.

## Failure cases
- Save fails: keep text, show unsaved status, block navigation until saved or explicitly confirmed; allow downloading draft.
- Second window edits: reject stale revision instead of overwriting.
- Duplicate completion: keep one immutable completed attempt, no extra credit.
- Language switch: save first, keep each language draft separate.
- Empty history, legacy unknown help, same-day retries, missing retired problems: handle explicitly; never infer independent success from missing data.
- Reference examples: test empty input, boundaries, duplicates, cycles; C++ examples require compilation in CI rather than claiming local validation without a compiler.

## Execution ledger
- Existing worktree verified. Unfinished CI visibility changes safely stashed separately before branching from fetched main.
