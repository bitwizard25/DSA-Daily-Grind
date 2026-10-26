"""Open the desktop window: the local server plus a chromeless Edge/Chrome app window.

The window uses a dedicated browser profile under ~/.google_prep/gui-profile, so it is its own
process: closing the window ends the app. Without Edge/Chrome the default browser opens the
same page and the app quits a few minutes after the page stops sending heartbeats."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

from google_prep import db
from google_prep.gui.api import Api
from google_prep.gui.server import serve

HEARTBEAT_TIMEOUT = 600  # seconds without a page heartbeat (sent every 30 s) before a browser-tab session ends;
# long, because a hidden or minimised window may send heartbeats only once a minute
SLEEP_GAP = 30  # a 2-second wait that took longer than this means the computer slept: not the page's fault
HANDOFF_SECONDS = 5  # a window process that exits faster than this handed off to an already-open browser
CLOSE_GRACE = 40  # after the window process ends, listen this long for the page (it pings every 30 s)
NATIVE_CLOSE_GRACE = 40  # when unload is lost, allow a heartbeat to prove a replacement window is alive
APP_MUTEX = "GoogleSWEIIOS.Running"  # must match AppMutex in installer/google-prep.iss


def browser_candidates() -> list[str]:
    if os.name == "nt":
        roots = [os.environ.get(k, "") for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        rel = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe",
               r"Chromium\Application\chrome.exe"]
        return [str(Path(r) / p) for p in rel for r in roots if r] + ["msedge", "chrome"]
    if sys.platform == "darwin":
        return ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    return ["microsoft-edge", "microsoft-edge-stable", "google-chrome", "google-chrome-stable", "chromium",
            "chromium-browser"]


def find_browser() -> str | None:
    override = os.environ.get("GOOGLE_PREP_BROWSER")
    if override:
        return override
    for candidate in browser_candidates():
        if os.path.isabs(candidate):
            if os.path.exists(candidate):
                return candidate
        elif shutil.which(candidate):
            return shutil.which(candidate)
    return None


def open_window(url: str) -> subprocess.Popen | None:
    browser = find_browser()
    if not browser:
        return None
    profile = db.home_dir() / "gui-profile"
    profile.mkdir(exist_ok=True)
    args = [browser, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1440,920",
            "--no-first-run", "--no-default-browser-check", "--disable-features=Translate",
            "--disable-sync", "--disable-extensions",
            "--disable-background-mode"]  # no hidden copy left running after the window closes
    try:
        return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def selftest() -> int:
    """Start the server, call a few actions over HTTP, check the page loads, exit. Used by CI.
    Never raises: the windowed build would turn an exception into a modal dialog that nobody can close."""
    try:
        server = serve(Api())
    except Exception:  # noqa: BLE001 - report every failure as an exit code plus a log
        return _selftest_failed()
    port = server.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
        assert "<title>Google SWE II OS</title>" in page, "index.html missing"
        for asset in ("app.js", "app.css", "icon.png", "engineering.js", "engineering.css"):
            res = urllib.request.urlopen(f"http://127.0.0.1:{port}/{asset}", timeout=10)
            assert res.status == 200 and res.read(), asset
        for name in ("state", "today", "roadmap", "problems", "star_list", "fasttrack", "vault_list", "engineering_state"):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/{name}", data=b"{}", method="POST",
                                         headers={"X-Prep-Token": server.token, "Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            assert "error" not in data, (name, data)
        _say(f"selftest ok: page + 5 assets + 8 actions on port {port}")
        return 0
    except Exception:  # noqa: BLE001
        return _selftest_failed()
    finally:
        server.shutdown()


def _selftest_failed() -> int:
    import traceback

    text = traceback.format_exc()
    try:
        (db.home_dir() / "selftest.log").write_text(text, encoding="utf-8")
    except OSError:
        pass
    _say(text)
    return 1


def _say(message: str) -> None:
    """Print when there is a console (the windowed Windows build has none)."""
    if sys.stdout is not None:
        print(message, flush=True)


def _windowed() -> bool:
    return sys.stdout is None  # PyInstaller --noconsole build


def _error_box(message: str, info: bool = False) -> None:
    """The windowed build has no console, so problems must be shown as a dialog."""
    _say(message)
    if os.name == "nt" and _windowed():
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "Google SWE II OS", 0x40 if info else 0x10)


def _release(handle) -> None:
    if handle and os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle(handle)
        except (AttributeError, OSError):
            pass


def _hold_app_mutex():
    """A named mutex the installer checks (AppMutex) so an update asks you to close the app first."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.restype = ctypes.c_void_p  # a HANDLE, not a 32-bit int
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        return kernel32.CreateMutexW(None, False, APP_MUTEX)
    except (AttributeError, OSError):
        return None


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # the windowed build has no stderr: show it
        _error_box(f"{message}\n\nTerminal commands are in google-prep-cli.exe, next to this program.")
        self.exit(2)

    def print_help(self, file=None):
        if _windowed():
            _error_box(self.format_help(), info=True)
        else:
            super().print_help(file)


# `google-prep.exe <command>` (old reminders and shortcuts) opens the window on the matching screen
ROUTES = {"today": "today", "ui": "today", "recall": "recall", "drills": "recall", "mock": "mock", "solve": "mock",
          "roadmap": "roadmap", "star": "star", "fasttrack": "fasttrack", "patterns": "vault", "vault": "vault",
          "role": "role"}


def _make_updater():
    from google_prep.updates.manager import UpdateManager
    from google_prep.updates.manifest import read_build_info
    from google_prep.updates.windows import detect_installation
    try:
        build = read_build_info(db.resource_dir())
        installation = detect_installation()
        if installation is not None and not build['build_number']:
            build = {**build, 'version':installation.version, 'build_number':installation.build_number}
        return UpdateManager(db.home_dir(), build, installation)
    except (OSError, ValueError, KeyError, TypeError):
        _log('updater unavailable; study features remain available')
        return None


def _close_updates(updater, reason):
    if updater is None: return
    updater.stop()
    if reason != 'window_closed' or updater.pending() is None: return
    from google_prep.updates.windows import launch_pending
    try:
        launch_pending(db.home_dir(), os.getpid())
    except (OSError, ValueError, KeyError, TypeError):
        _log('pending update deferred; reopen the app to retry')


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(prog="google-prep", description="Google SWE II OS desktop app")
    parser.add_argument("screen", nargs="?", default="today", help="open on: " + ", ".join(sorted(set(ROUTES))))
    parser.add_argument("--selftest", action="store_true", help="start, check, exit (for CI)")
    parser.add_argument("--no-browser", action="store_true", help="only serve; print the URL")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--version", action="store_true")
    parser.add_argument('--reminder', action='store_true', help='Open only when no planned break is active.')
    args, extra = parser.parse_known_args(argv)
    if args.version:
        from google_prep import __version__

        if _windowed():
            _error_box(f"Google SWE II OS {__version__}", info=True)
        else:
            _say(f"google-prep {__version__}")
        return 0
    if args.selftest:
        return selftest()
    if args.reminder:
        from google_prep.core import breaks, clock
        with db.get_session() as session:
            if breaks.is_break(session, clock.now().date()): return 0
    if extra or args.screen not in ROUTES:
        parser.error(f"unknown command: {' '.join([args.screen, *extra]).strip()}")
    if args.no_browser and _windowed():
        _error_box("--no-browser needs a terminal. Run: google-prep-cli ui --no-browser")
        return 2
    updater = None if args.no_browser else _make_updater()
    if updater is not None and updater.installation is not None:
        from google_prep.updates import windows
        try:
            if windows.installation_is_running():
                _error_box('An app update is finishing. Open Google SWE II OS again in a moment.', info=True)
                return 0
            if updater.pending() is not None and not windows.app_is_running():
                if windows.launch_pending(db.home_dir(), os.getpid(), reopen=True):
                    return 0
        except (OSError, ValueError, KeyError, TypeError):
            _log('startup update deferred; opening the app normally')
    mutex = _hold_app_mutex()  # held while this process shows the app
    try:
        server = serve(Api(updater=updater), args.port)
    except Exception as exc:  # e.g. a damaged database or a port already in use
        _error_box(f"Google SWE II OS could not start:\n{type(exc).__name__}: {exc}")
        _release(mutex)
        _close_updates(updater, 'startup_failure')
        return 1
    if updater is not None: updater.start()
    url = server.url + ("" if ROUTES[args.screen] == "today" else f"#/{ROUTES[args.screen]}")
    proc = None if args.no_browser else open_window(url)
    if proc is None and not args.no_browser:
        webbrowser.open(url)
    _say(f"Google SWE II OS is running at {url}  (Ctrl+C to quit)")
    exit_reason = 'browser_handoff'
    try:
        if proc is not None:
            started = time.monotonic()
            _log(f"started on {url.split('?')[0]} with its own window")
            if _wait_for_browser(proc, server):
                _log('all app pages closed: stopping the hidden server')
                exit_reason = 'window_closed'
                return 0
            if time.monotonic() - started >= HANDOFF_SECONDS:
                if not _page_still_open(server):
                    _log("window closed: stopping")
                    exit_reason = 'window_closed'
                    return 0
                # Edge can swap its process (an update, a relaunch) while the window stays open
                _log("window process ended but the page is still open: serving on heartbeats")
                exit_reason = _wait_for_page(server)
                return 0
            # The browser handed this page to an existing process. Keep this page's app lock
            # until it closes too: another server's window can close before this one.
        _log("window handed off to a running browser: serving while the page sends heartbeats")
        exit_reason = _wait_for_page(server, forever=args.no_browser)
    except KeyboardInterrupt:
        exit_reason = 'keyboard_interrupt'
    finally:
        server.shutdown()
        _release(mutex)
        _close_updates(updater, exit_reason)
    return 0


def _page_still_open(server) -> bool:
    """True if the page sends a heartbeat (or any request) after its window process ended."""
    ended = time.monotonic()
    while time.monotonic() - ended < CLOSE_GRACE:
        if _reported_closed(server):
            return False
        if server.last_seen > ended:
            return True
        time.sleep(1)
    return server.last_seen > ended


def _reported_closed(server) -> bool:
    return getattr(server, 'windows_closed', lambda:False)()


def _browser_has_window(pid) -> bool | None:
    """Check only this browser's top-level windows. Minimized/covered windows remain visible to Win32."""
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes
    try:
        library = ctypes.WinDLL('user32', use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        library.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        library.EnumWindows.restype = wintypes.BOOL
        library.IsWindowVisible.argtypes = [wintypes.HWND]
        library.IsWindowVisible.restype = wintypes.BOOL
        library.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        library.GetWindowThreadProcessId.restype = wintypes.DWORD
        found = False
        def visit(handle, _):
            nonlocal found
            owner = wintypes.DWORD()
            library.GetWindowThreadProcessId(handle, ctypes.byref(owner))
            if owner.value == pid and library.IsWindowVisible(handle):
                found = True
                return False
            return True
        callback = callback_type(visit)
        succeeded = library.EnumWindows(callback, 0)
        return True if found else False if succeeded else None
    except (OSError, AttributeError):
        return None  # unavailable is not evidence that a window was closed


def _wait_for_browser(proc, server) -> bool:
    """A browser process may outlive its last window; keep observing the window and the page."""
    seen_window, missing_since = False, None
    while proc.poll() is None:
        if _reported_closed(server):
            return True
        pid = getattr(proc, 'pid', None)
        visible = _browser_has_window(pid) if pid else None
        now = time.monotonic()
        if visible is True:
            seen_window, missing_since = True, None
        elif visible is False and seen_window:
            if missing_since is None or getattr(server, 'last_seen', 0) > missing_since:
                missing_since = now
            elif now - missing_since >= NATIVE_CLOSE_GRACE:
                return True
        else:
            missing_since = None
        time.sleep(.5)
    return False


def _wait_for_page(server, forever: bool = False) -> str:
    """Serve until the page has been silent for HEARTBEAT_TIMEOUT. Time the computer spends asleep
    does not count: after a nap the open window must still work."""
    last = time.monotonic()
    while True:
        if not forever and _reported_closed(server):
            _log('all app pages closed: stopping the hidden server')
            return 'window_closed'
        now = time.monotonic()
        if now - last > SLEEP_GAP:
            _log(f"woke up after {round(now - last)} s asleep: keeping the app running")
            server.last_seen = now
        last = now
        if not forever and now - server.last_seen >= HEARTBEAT_TIMEOUT:
            _log(f"no heartbeat for {round(now - server.last_seen)} s: the page is closed, stopping")
            return 'heartbeat_timeout'
        time.sleep(2)


def _log(message: str) -> None:
    """A short history of why the app started and stopped (~/.google_prep/gui.log), for support."""
    try:
        path = db.home_dir() / "gui.log"
        if path.exists() and path.stat().st_size > 200_000:
            path.write_text("", encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass
