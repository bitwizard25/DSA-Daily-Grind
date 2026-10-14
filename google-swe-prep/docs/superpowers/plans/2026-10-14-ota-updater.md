# OTA Updater Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the installed Windows EXE receive verified updates from successful GitHub CI builds automatically, without interrupting study or losing data.

**Architecture:** A GitHub release contains a Windows installer and an update manifest. A Python worker checks and stages downloads while the app is open; a separate helper installs after the app exits. The helper independently checks the cache, registered installation and running-app mutex.

**Tech Stack:** Python 3.12/stdlib, existing SQLModel application, plain JavaScript, PyInstaller, Inno Setup 6, GitHub Actions and GitHub Releases. No new runtime third-party dependencies.

**Spec:** [Approved OTA design](./OTA-Updater-Design.md), approved by the user on 14 October 2026.

**Execution status:** Planning only. Product code has not been modified. Recommend native execution in this chat because the launcher, helper and release format depend on the same small set of interfaces.

## Global Constraints

- “Publish updates only from successful `main` builds, after every matrix job and the Windows installer smoke test succeeds.”
- “Development branch builds cannot become automatic updates.”
- “Run network work in a Python background worker, independently of browser timers and heartbeats.”
- “Check at launch and at most every six hours after a successful check.”
- “After a connection failure, retry with increasing delays: one minute, five minutes, fifteen minutes, then hourly.”
- “Never accept an arbitrary installer URL supplied by the web interface.”
- “Automatic installation supports the registered Windows x64 Inno Setup installation.”
- “It does not kill the app or unrelated browser processes.”
- “Clear the pending update only after success and confirm the new build on the next launch.”
- The current application AppId remains `{03EDDFF2-4DC8-4E0E-84DE-FA73BFFCC5D8}` and app mutex remains `GoogleSWEIIOS.Running`.
- Preserve `.google_prep` and respect `GOOGLE_PREP_HOME` in tests and normal use.

## Review Focus

1. Two running app instances: stage at most one download/install; the helper must defer until both close. Task 2 owns the worker lock; Task 3 owns the app-mutex test.
2. A study window loses heartbeats: updater installation must not be triggered by the heartbeat shutdown path. Task 4 explicitly tests that path and defers the update.
3. Same source version, newer CI build: detect the new build; never downgrade to a lower source version or an older build. Tasks 1 and 5 test ordering.
4. Windows paths with spaces or non-ASCII characters: preserve arguments, installation path, logs and profile data without shell interpolation. Task 3 tests the argument list.
5. Interrupted publication or download: ignore draft/incomplete releases; never install a partial file; a rerun must not overwrite published assets. Tasks 1, 2 and 5 test these cases.

## Working tree and file map

At execution time, read the using-git-worktrees skill and create/reuse an isolated checkout from current `origin/main`; use branch `codex/ota-updates`. Preserve existing uncommitted work. Keep approved design and plan under `google-swe-prep/docs/superpowers/specs/2026-10-14-ota-updater-design.md` and `google-swe-prep/docs/superpowers/plans/2026-10-14-ota-updater.md` in the execution checkout.

New Python package `google-swe-prep/google_prep/updates/`:

| File | Responsibility |
| --- | --- |
| `manifest.py` | Build identity, candidate validation, update ordering and fixed product constants |
| `client.py` | Restricted GitHub release lookup and streamed, verified downloads |
| `manager.py` | Preferences, background checks, state snapshots and pending-update persistence |
| `windows.py` | Registered installation detection, Windows mutex/process checks and helper launch |
| `helper.py` | Standalone install helper; checksum recheck, SQLite backup, silent installer and result log |

Existing integration files: `google_prep/gui/launcher.py`, `google_prep/gui/api.py`, `google_prep/gui/static/app.js`, `google_prep/gui/static/training.css`, `build.py`, `installer/app.spec`, `installer/google-prep.iss`, `README.md`, `__init__.py`, `pyproject.toml`, `.github/workflows/build-exe.yml`.

New CI scripts: `google-swe-prep/scripts/release_update.py` and `google-swe-prep/scripts/ota_smoke.py`. Unit tests: `tests/test_updates.py`, `tests/test_update_manager.py`, `tests/test_update_install.py`, `tests/test_update_release.py`; frontend regression added to `tests/frontend-flows.test.cjs`.

All commands below run from the execution checkout's `google-swe-prep/`, using its Python environment. On Windows, set `TEMP`, `TMP` and pytest `--basetemp` to task-local paths. No test uses the user's real progress folder.

## Task 1: Define the release contract and verified download client

**Interfaces produced**

```python
@dataclass(frozen=True)
class Candidate:
    version: str
    build_number: int
    commit: str
    tag: str
    installer_url: str
    installer_size: int
    installer_sha256: str

def read_build_info(resource_root: Path) -> dict: ...
def validate_candidate(release: dict, manifest: dict, installed: dict) -> Candidate | None: ...
def fetch_candidate(installed: dict) -> Candidate | None: ...
def download_candidate(candidate: Candidate, cache_dir: Path) -> Path: ...
```

The signatures above describe the contract, not implementation placeholders. Implement these functions fully in this task.

**Exact manifest format**

```json
{
  "schema": 1,
  "app_id": "google-swe-prep",
  "repository": "bitwizard25/DSA-Daily-Grind",
  "platform": "windows-x64",
  "version": "1.6.0",
  "build_number": 100,
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "tag": "google-prep-v1.6.0-b100",
  "installer": {
    "name": "GoogleSWEPrep-Setup.exe",
    "size": 3,
    "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
  }
}
```

This is a test fixture: three bytes `abc` produce the displayed checksum. Production size and checksum come from the built installer. The manifest does not provide an executable command or an arbitrary destination URL.

- [ ] Write `test_updates.py` with fixtures for the manifest and a GitHub release whose exact installer asset URL is `https://github.com/bitwizard25/DSA-Daily-Grind/releases/download/google-prep-v1.6.0-b100/GoogleSWEPrep-Setup.exe`. Include tests like:

```python
def test_new_build_with_same_version_is_an_update(release, manifest):
    candidate = validate_candidate(release, manifest, {"version": "1.6.0", "build_number": 99})
    assert candidate.build_number == 100

def test_older_version_is_never_an_automatic_update(release, manifest):
    assert validate_candidate(release, manifest, {"version": "1.7.0", "build_number": 1}) is None

@pytest.mark.parametrize("field,value", [("build_number", True), ("build_number", -1),
                                       ("version", "1.6.0-rc1"), ("platform", "other")])
def test_bad_manifest_is_rejected(release, manifest, field, value):
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_candidate(release, manifest, {"version": "1.6.0", "build_number": 99})
```

- [ ] Run `python -m pytest -q tests/test_updates.py`; confirm failure because the new package/functions do not exist.
- [ ] Implement `manifest.py` with strict three-integer source-version parsing; positive integer build numbers excluding booleans; 40-character hexadecimal commits; exact app/repository/platform/schema; SHA-256 regex `[0-9a-f]{64}`; installer size `1 <= size <= 500 * 1024 * 1024`; and a tag that matches version/build. Ignore draft, prerelease and unrelated-product releases. Require exact asset name, uploaded state, size and tag/repository URL. Compare `(major, minor, patch, build_number)` tuples.

```python
def version_key(version: str, build_number: int) -> tuple[int, int, int, int]:
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Invalid update version")
    if type(build_number) is not int or build_number < 0:
        raise ValueError("Invalid build number")
    return (*map(int, version.split(".")), build_number)
```

`read_build_info` reads `data/build_info.json`. Missing metadata identifies source/local builds with build number zero and must disable automatic installation, not invent a CI identity.

- [ ] Implement `client.py` with `urllib.request` and fixed repository endpoints. List up to three pages of 100 releases, filter the product prefix, choose the newest candidate identity rather than response order, fetch its `google-prep-update.json`, then validate. Bound API JSON to 2 MiB and manifest to 64 KiB; use finite network timeouts. Validate every redirect hop: HTTPS, no user-info, expected GitHub origin/path initially, then only GitHub's asset CDN hosts. Do not let the default opener bypass redirect checks.
- [ ] Stream the installer to `installer.part`, enforce manifest size during streaming, calculate SHA-256, fsync and atomically rename only after validation. On failure remove the partial file, never the previously verified installer. Restrict writes to the updater cache directory. A completed download still does not install anything.

```python
digest = hashlib.sha256()
count = 0
with response, partial.open("wb") as out:
    while chunk := response.read(64 * 1024):
        count += len(chunk)
        if count > candidate.installer_size:
            raise ValueError("Update download exceeds its declared size")
        digest.update(chunk)
        out.write(chunk)
    out.flush()
    os.fsync(out.fileno())
if count != candidate.installer_size or digest.hexdigest() != candidate.installer_sha256:
    raise ValueError("Update download could not be verified")
partial.replace(destination)
```

- [ ] Add mocked HTTP tests for timeout, HTTP 429/403 with Retry-After/reset headers, unrelated latest release, missing assets, oversized JSON, cross-host redirect, digest mismatch and interrupted stream. Pin rate-limit retry behavior in a typed client exception exposing `retry_after_seconds`.
- [ ] Run all Task 1 tests and commit only the manifest/client and their tests.

## Task 2: Background checks, preferences and persistent pending state

**Consumes:** Task 1's `read_build_info`, `fetch_candidate`, `download_candidate` and `Candidate`.

**Produces:**

```python
class UpdateManager:
    def __init__(self, home: Path, build: dict, installation, client=None): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> dict: ...
    def check(self) -> dict: ...
    def set_enabled(self, enabled: bool) -> dict: ...
    def pending(self) -> Path | None: ...
```

`installation` is Task 3's `Installation` object or `None`. Inject a client and clocks/events for deterministic tests. No updater thread starts just by constructing `Api()` in source tests.

- [ ] Write manager tests that block a fake client on an event, then confirm `check()` returns immediately and a second `check()` does not start another request. Confirm fake source mode never downloads an installer. Test disabled preference persists after reconstructing the manager.

```python
def test_off_setting_survives_restart(home, build, installation, fake_client):
    manager = UpdateManager(home, build, installation, client=fake_client)
    assert manager.set_enabled(False)["enabled"] is False
    reopened = UpdateManager(home, build, installation, client=fake_client)
    assert reopened.status()["enabled"] is False

def test_bad_preference_file_does_not_reenable_updates(home, build, installation):
    path = home / "updates" / "preferences.json"
    path.parent.mkdir()
    path.write_text("not json")
    assert UpdateManager(home, build, installation).status()["enabled"] is False
```

- [ ] Run `python -m pytest -q tests/test_update_manager.py`; confirm failure before adding the manager.
- [ ] Implement `manager.py` with one daemon worker, a thread-safe status snapshot, wake/stop events and no network calls while holding the GUI RPC lock. Persist `preferences.json`, `pending.json` and result metadata under `home/updates/` via temporary files and `os.replace`. Default automatic updates to true only when no preference exists; corrupted preference data disables automation until the user explicitly enables it.
- [ ] Use states `unsupported`, `idle`, `checking`, `available`, `downloading`, `ready`, `offline`, `failed`. Status exposes current version/build, enabled flag, support reason, candidate version/build, last check time, downloaded/total bytes, last result and a plain message. It never exposes a GitHub token or accepts a web-provided installer URL.

```python
RETRY_SECONDS = (60, 300, 900, 3600)
CHECK_SECONDS = 6 * 3600
delay = CHECK_SECONDS if succeeded else RETRY_SECONDS[min(failures - 1, 3)]
if rate_limit_seconds is not None:
    delay = max(delay, rate_limit_seconds)
stop_event.wait(delay)
```

- [ ] While automation is disabled, manual checks may inspect release metadata but cannot automatically download/install. Provide the validated release link for a manual download. Switching off during a download prevents promotion to installable pending state. Check enabled state again before launching a helper.
- [ ] Add a named cross-process update mutex for installed Windows instances. A second instance can report the shared cache/result but cannot overwrite staging files or start a second install. A crash leaves no stale permanent lock; OS handles are closed on every exit.
- [ ] Add tests for exact retry schedule, GitHub rate limits, preference toggle during download, pending state after restart, corruption/stale pending identity, shutdown during download and two-instance lock contention. Run manager/client tests and commit.

## Task 3: Safe Windows installation and standalone helper

**Consumes:** Task 1 validation and Task 2's `pending.json`.

**Produces:** `Installation(path: Path, version: str, build_number: int)`; `registered_installation() -> Installation | None`; `detect_installation(executable: Path | None = None) -> Installation | None`; `launch_pending(home: Path, parent_pid: int, reopen: bool = False) -> bool`; `install_pending(home: Path, parent_pid: int, reopen: bool = False) -> int`; and the packaged `google-prep-update.exe`.

Files: create `updates/windows.py`, `updates/helper.py`, `tests/test_update_install.py`; modify `build.py`, `installer/google-prep.iss` and the installed folder's packaging.

- [ ] Write tests with fake registry, mutex, process and subprocess adapters. Test registered installed mode, unregistered portable mode, source mode, two running app instances, reparse/symlink cache paths, altered pending JSON, wrong hash, pending older version, Windows paths containing spaces/non-ASCII, timeout and nonzero installer exit. Check no subprocess launches until every prerequisite passes.

```python
def test_open_app_defers_install(monkeypatch, home, pending_update):
    calls = []
    monkeypatch.setattr(windows, "app_is_running", lambda: True)
    monkeypatch.setattr(helper.subprocess, "run", lambda *a, **k: calls.append(a))
    assert helper.install_pending(home, parent_pid=0) != 0
    assert calls == []

def test_tampered_installer_never_runs(monkeypatch, home, pending_update):
    pending_update.installer.write_bytes(b"altered")
    monkeypatch.setattr(helper.subprocess, "run", lambda *a, **k: pytest.fail("installer ran"))
    assert helper.install_pending(home, parent_pid=0) != 0
```

- [ ] Run the install tests and confirm failure before implementing these modules.
- [ ] Implement `registered_installation` to resolve the per-user Inno uninstall key `{03EDDFF2-4DC8-4E0E-84DE-FA73BFFCC5D8}_is1` in both registry views. Validate the registered install location and require `unins000.exe` plus both app EXEs. Read identity from bundled metadata when present; an older registered installer without metadata uses its validated registry `DisplayVersion` and build zero. This permits the CI old-to-new upgrade test without pretending the old app had an updater.
- [ ] Implement `detect_installation` for GUI automation with the additional requirements: frozen Windows x64 build, current EXE equals the registered `google-prep.exe`, and current bundled CI metadata has a nonzero build identity. The external helper uses `registered_installation` rather than requiring its own executable path to equal the GUI path. Do not infer installed mode merely from a familiar filename.
- [ ] Implement named mutex helpers with correct 64-bit HANDLE signatures and guaranteed `CloseHandle`. Use `GoogleSWEIIOS.Update` to serialize staging and `GoogleSWEIIOS.Installing` for installation. Open the existing `GoogleSWEIIOS.Running` mutex to detect running app instances; do not create that mutex in the helper. Launcher startup must defer while the installation mutex exists.
- [ ] Stage a copy of the small helper outside the installation directory. Start it with a subprocess argument list and Windows `CREATE_NO_WINDOW`, never a shell command. It derives the cache root from the trusted profile and registered install location; pending metadata cannot select a different target or executable. Validate resolved paths remain beneath that cache root and refuse links/reparse points for executable/cache files.
- [ ] The helper acquires the install mutex, waits a bounded 120 seconds for the parent and all app mutexes to disappear, rereads the shared enabled preference, rechecks pending metadata/build identity, installer size and digest, and rechecks the registered install directory. It defers if the app remains open or automation was disabled in another window. It does not invoke `taskkill` or force-close browsers, terminals or the app.
- [ ] Before installation, back up `prep.db` using SQLite's backup API to a unique `updates/backups/` file. Support WAL databases; do not use an ordinary file copy. If backup fails or the source database is unreadable, do not install.

```python
with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=10) as source:
    with sqlite3.connect(backup_path) as destination:
        source.backup(destination)
        if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Progress backup could not be verified")
```

- [ ] Run the installer with these argument-list entries; `install_dir`, `installer` and `log_path` are locally derived validated Paths:

```python
args = [str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/SP-", "/NORESTART",
        "/NOCLOSEAPPLICATIONS", "/NORESTARTAPPLICATIONS", "/RESTARTEXITCODE=3010",
        f"/DIR={install_dir}", f"/LOG={log_path}"]
result = subprocess.run(args, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
```

Exit zero records successful install; 3010 records `restart_required` without restarting Windows or repeatedly running the installer; timeout/other codes record failure and retain the pending installer for manual repair/retry. Never clear pending state based on process creation alone. Record installer outcome atomically. On next app launch, compare bundled identity to the pending candidate before clearing it. Failed installs do not cause an unattended retry loop.

- [ ] Set `UsePreviousAppDir=yes` and `UsePreviousTasks=yes` explicitly in Inno Setup. Preserve AppId and installation location. Do not pass `/TASKS` during OTA installation; test both existing desktop-shortcut/PATH opt-in and opt-out settings. User study files remain outside `[InstallDelete]` and `[Files]` destinations.
- [ ] Build the helper separately as a small stdlib-only PyInstaller one-file EXE before Inno compilation, then copy it into `dist/GoogleSWEPrep/` after the folder build. It must not be a process loading DLLs from the folder it replaces. Source and portable app builds stay non-installing.

```python
subprocess.run([sys.executable, "-m", "PyInstaller", "--onefile", "--noconsole",
                "--noconfirm", "--name=google-prep-update", "--paths=.",
                "--distpath=dist/update-helper", "google_prep/updates/helper.py"],
               cwd=ROOT, check=True)
shutil.copy2(ROOT / "dist/update-helper/google-prep-update.exe",
             ROOT / "dist/GoogleSWEPrep/google-prep-update.exe")
```

- [ ] Add tests for a helper that times out, a shortcut/PATH preference retained during upgrade, a WAL backup with committed rows, failed backup preventing launch, successful install awaiting next-launch confirmation, and pending state not being applied twice. Run all update unit tests and commit.

## Task 4: Launcher, API and simple Settings controls

**Consumes:** Task 2's manager and Task 3's installed-mode detection/helper functions.

**Produces:** `Api(now=clock.now, updater=None)` constructor injection; public RPCs `update_status() -> dict`, `update_check() -> dict`, `update_preferences(enabled: bool) -> dict`. GUI `state()` also reports current build number when present. Existing `Api()` calls remain offline and side-effect free.

Files: modify `gui/api.py`, `gui/launcher.py`, `gui/static/app.js`, `gui/static/training.css`, `tests/test_gui.py`, `tests/frontend-flows.test.cjs`, `README.md`; bump source version to 1.6.0 in `__init__.py` and `pyproject.toml`.

- [ ] Write API tests with an injected fake manager: status/check delegate correctly; `enabled` rejects `"true"`, 1 and null; check returns a status immediately rather than performing networking on the request thread. Source-mode status states why automatic installation is unavailable.

```python
@pytest.mark.parametrize("value", ["true", 1, None])
def test_update_preference_requires_a_boolean(value):
    with pytest.raises(ApiError):
        Api(updater=FakeManager()).update_preferences(enabled=value)
```

- [ ] Add launcher lifecycle tests with fake processes, clocks and manager. Assert no installation on `--selftest`, source runs, startup failure, `--no-browser`, keyboard interruption, heartbeat timeout or uncertain browser handoff. Normal owned-window closure may launch a pending helper, which still waits for the parent/all instances to exit.
- [ ] Confirm tests fail, then add API delegates with no global network worker. Attach the manager to the launcher-created API only for normal interactive desktop runs. Queue checks in the manager and return immediately.

```python
def update_preferences(self, enabled: bool) -> dict:
    if type(enabled) is not bool:
        raise ApiError("Automatic updates must be on or off.")
    return self.updater.set_enabled(enabled)
```

When `self.updater` is absent, status reports `unsupported` and check/preferences return a short explanatory ApiError instead of creating a background service. A status read never downloads or installs anything.

- [ ] On launch, inspect validated pending state before starting the server/browser. If an automatic update is ready and no install/app lock exists, start the helper with `reopen=True` and exit this launcher; the helper relaunches the installed app only after success, because the user just requested a launch. If install is already running, show a short finishing-update notice and do not open an old study window. Never install directly inside the running GUI process.
- [ ] During ordinary running, start the daemon manager. On shutdown, stop it; schedule installation only for the explicitly tracked `window_closed` exit reason. For heartbeat timeout or ambiguous handoff, leave the pending update for the next safe launch. Close all mutex handles correctly on startup failure and exit. A user closing the app does not cause it to reopen after updating.
- [ ] Add an update section inside existing Settings, with version/build, automatic-updates checkbox, **Check now**, status text and a validated release link for manual mode. Do not add a main navigation destination or repeated toasts. Updates must not refresh/restart the whiteboard, recall or STAR form.

```javascript
const automatic = h('input', {type:'checkbox', checked:status.enabled,
  disabled:!status.supported, 'data-test':'auto-updates'});
const message = h('p', {role:'status', 'aria-live':'polite', 'data-test':'update-status'}, status.message);
const check = btn('Check now', async () => {
  check.disabled = true;
  try {paint(await api('update_check'));} catch (error) {message.textContent = error.message;}
  finally {check.disabled = false;}
}, {cls:'ghost small', test:'update-check'});
```

`paint(status)` updates only this settings section. Poll every five seconds only while its modal is open and clear the interval on every close path. Keep the automatic-updates input out of the existing `FormData` payload (or explicitly remove it) so the old `settings_save` signature does not receive an unexpected argument. Saving that preference through `update_preferences` must show failures and restore the last confirmed checkbox state.

- [ ] Frontend tests exercise check-in-progress controls, error text, unsupported/manual mode, preference failure, and modal cleanup. API/launcher tests verify no network or helper launch during GUI selftest and no helper after heartbeat timeout. Run:

```text
python -m pytest -q tests/test_gui.py tests/test_updates.py tests/test_update_manager.py tests/test_update_install.py
node --check google_prep/gui/static/app.js
node --test tests/frontend-autosave.test.cjs tests/frontend-flows.test.cjs
```

- [ ] Document checks while running, installation on close/next launch, first manual upgrade, supported installed Windows mode, opt-out and manual repair. Commit the integration after tests pass.

## Task 5: CI metadata, release publishing and a real upgrade smoke test

**Consumes:** Task 1 manifest schema, Tasks 2–4's packaged updater, existing matrix tests/installer build.

**Produces:** `scripts/release_update.py` with subcommands `metadata`, `manifest`, `publish`; `scripts/ota_smoke.py` with `seed` and `verify`; an additional release job in `.github/workflows/build-exe.yml`.

- [ ] Write pure release tests: deterministic tag/digest/size, older build cannot supersede a newer one, development branches cannot publish, drafts stay invisible, already-published assets are not overwritten, and an incomplete draft from a failed upload can be discarded/recreated without changing a published release. Use a fake GitHub adapter; no test publishes to the real repo.

```python
def test_published_build_is_not_overwritten(fake_github, build, installer):
    fake_github.existing_published_tag = "google-prep-v1.6.0-b100"
    publish_update(fake_github, build, installer)
    assert fake_github.uploads == []

def test_late_older_build_is_not_promoted(fake_github, build, installer):
    fake_github.latest_build_number = build["build_number"] + 1
    publish_update(fake_github, build, installer)
    assert fake_github.published == []
```

- [ ] Run the new tests and confirm failure. Implement metadata generation before every CI build using `GITHUB_SHA` and `GITHUB_RUN_NUMBER`. Write `data/build_info.json` into the CI workspace only; do not commit generated build IDs. Metadata includes exact schema/app/platform/version/build/commit. The Windows installer and manifest must use the same metadata; publisher must read uploaded build metadata rather than rebuilding a manifest from a later checkout.

```python
build = {"schema": 1, "app_id": "google-swe-prep", "platform": "windows-x64",
         "version": __version__, "build_number": int(os.environ["GITHUB_RUN_NUMBER"]),
         "commit": os.environ["GITHUB_SHA"]}
tag = f"google-prep-v{build['version']}-b{build['build_number']}"
sha256 = hashlib.sha256(installer.read_bytes()).hexdigest()
```

For large installers hash incrementally as in Task 1. CI package uploads contain the installer, generated manifest and build metadata, retaining ordinary portable artifacts.

- [ ] Add the publish job with these gating rules and job-local permissions:

```yaml
publish-update:
  needs: build
  if: github.ref == 'refs/heads/main' && needs.build.result == 'success'
  runs-on: ubuntu-latest
  permissions:
    contents: write
    actions: read
  concurrency:
    group: google-prep-stable-publish
    cancel-in-progress: false
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    - uses: actions/download-artifact@v4
      with:
        name: GoogleSWEPrep-Setup-Windows
        path: update-release
    - name: Publish verified Windows update
      env:
        GH_TOKEN: ${{ github.token }}
      run: python google-swe-prep/scripts/release_update.py publish --directory update-release
```

- [ ] Publisher creates a draft tag targeting the exact built commit, uploads installer/manifest/build info, then publishes the completed release. It checks the latest existing product build immediately before publishing and never overwrites an existing published tag. GitHub token exists only in this CI job, never in the manifest or EXE. Use structured API requests or `gh` subprocess argument lists; no shell-built release body.
- [ ] Add a Windows upgrade smoke test before artifact upload. Compile an older baseline installer from pinned commit `8acc77492e5692f95c0a2b16603339cf0ed64742` in a separate directory, install it, seed a temporary test profile, and update to the new installer using the packaged helper. The baseline commit is the verified 1.5.0 release already merged in this chat.

```text
python scripts/ota_smoke.py seed --home <runner-test-profile>
<packaged-helper> --home <runner-test-profile> --parent-pid 0
<installed-google-prep.exe> --selftest
python scripts/ota_smoke.py verify --home <runner-test-profile>
```

The script must seed actual app data: one successful no-help Attempt, a STAR story, TrainingCheckin, saved design checklist/notes and session recovery metadata; record expected identifiers/text in a test fixture. Verify each record and file after upgrade, bundled build identity, successful installer result, SQLite backup integrity, and that invoking the helper a second time starts no installer. Cover existing desktop/PATH preferences and preserve the real user's profile. This test uses a dedicated `GOOGLE_PREP_HOME`, never the CI runner's default progress directory.
- [ ] Run all unit/frontend tests and the workflow on the branch. Review the full branch diff, address actionable review feedback, and create a PR with exact test results and supported-mode limits. Attach the PR to this chat. Publish only after merging this feature to `main`; user authorization for OTA publication is part of the approved design. Verify the actual published release has the matching installer, manifest and commit, and the anonymous public lookup sees it.

## Completion evidence and handoff

- [ ] Full Python suite passes, including updater/network/helper/CI contract tests.
- [ ] Node frontend tests and JavaScript syntax check pass.
- [ ] Windows CI performs a real old-to-new upgrade with data intact; all existing matrix builds stay green.
- [ ] Release publishing is absent on development branches and succeeds on the merged `main` build.
- [ ] One installed-app check stages the published installer and helper; installation completes after closing; the next launch confirms the new build and no repeated installation.
- [ ] Report separately: implemented code, published release, verified installed upgrade, and any unverified live/browser behavior. Do not claim OTA works end-to-end from unit tests alone.

## Self-review result

The five tasks cover all approved design sections. The manifest, installed identity and helper arguments have explicit contracts. Offline retry, interrupted download/publication, multiple instances, heartbeat shutdown, corrupt preference/pending data, Windows path handling and upgrade-data preservation each have an owning test. First manual upgrade, unsupported modes, no forced restart and no automatic rollback promise remain explicit in the documentation. Registered baseline identity and frozen GUI automation are separate checks, so the old-to-new smoke test can upgrade 1.5.0 without claiming that version already supports OTA.
