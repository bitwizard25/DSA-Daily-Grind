# google-swe-prep

CLI coach for a Google SWE II prep plan: 3 phases / 8 weeks, daily time-boxing, readiness index.

Note: `google_swe_prep_spec.md` was not provided; the design comes from the plan diagram
(phases, 20m warmup + 60-90m evening block, 10m micro-mode, Readiness = Mastery + Mock Speed + Googlyness).

## Install and run
    pip install -e ".[dev]"
    google-swe-prep today            # daily plan (add --micro for 10-minute days)
    google-swe-prep log 6 -m 35      # log a solve (--no-solved if not)
    google-swe-prep recall           # spaced-repetition drills
    google-swe-prep mock --limit 45  # timed mock
    google-swe-prep roadmap
    google-swe-prep star add -c Leadership -t "..." -s "..." --task "..." -a "..." -r "..."
    google-swe-prep star list

Data: `~/.google-swe-prep/prep.db` (override with `GSP_DB`).

## Readiness index
50% Mastery (frequency-weighted problems solved 70%, recall boxes 30%)
+ 30% Mock speed (last 5 mocks: time vs limit 70%, communication 30%, unsolved = 0)
+ 20% Googlyness (6 competencies with at least one STAR story).

## Executable
`python build_exe.py` builds for the OS you run it on. A Windows `.exe` must be built on Windows:
run the `build-exe` GitHub Actions workflow and download the artifact.

## Tests
    pytest
