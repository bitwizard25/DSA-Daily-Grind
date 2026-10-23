# Planned study breaks

Purpose: a deliberate travel/rest day should preserve progress and consistency without becoming a fake study day or creating a catch-up quota.

## Behavior

- One profile-wide break, shared by General and Google modes. Choose a start date and an exclusive **Resume on** date. Resume automatically on that date; allow cancelling an upcoming break or returning early.
- Store dated intervals independently of the original start date. Keep elapsed intervals after resuming so roadmap arithmetic never jumps forward again. Reject past starts, empty/reversed ranges, overlaps, and breaks longer than 365 days.
- Exclude break days from study-day/phase/pace arithmetic. Preserve the fixed interview target and keep its countdown based on real dates. Spaced repetition dates remain real dates because rest does not stop forgetting.
- Break days neither increase nor break the existing streak. A genuinely missed study day after returning still breaks it. Actual optional study remains accessible from the navigation and counts only when completed.
- Preserve unfinished sessions, code, notes and all graded work. An immediate break pauses an active General-mode focus block. Saving a break must run the current screen's save-before-leave guard first.
- Today shows a short rest card, the return date and a manage/resume action. Keep the control accessible from the sidebar and Settings in both modes.
- Generated reminders check break state before opening the app. Existing external OS tasks must be recreated once to use the new entrypoint; do not silently rewrite unrelated OS tasks. Manual app launches always work.

## Verification

Check travel Fri→Sat/Sun→Mon, early return, cancellation before starting, repeated breaks, overlap rejection, date boundaries, unchanged interview date, no phantom completion, saved sessions across modes, silent scheduled launches, and frontend failed-save retention. Verify the date controls and rest card in an isolated profile before pushing. Platform CI must pass before merging/releasing the installed version.
