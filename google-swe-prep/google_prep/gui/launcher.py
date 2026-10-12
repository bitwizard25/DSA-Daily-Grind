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

HEARTBEAT_TIMEOUT = 150  # seconds without a page heartbeat (sent every 30 s) before a browser-tab session ends
HANDOFF_SECONDS = 5  # a window process that exits faster than this handed off to an already-open browser
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
            "--disable-sync", "--disable-extensions"]
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
        for asset in ("app.js", "app.css", "icon.png"):
            res = urllib.request.urlopen(f"http://127.0.0.1:{port}/{asset}", timeout=10)
            assert res.status == 200 and res.read(), asset
        for name in ("state", "today", "roadmap", "problems", "star_list", "fasttrack", "vault_list"):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/{name}", data=b"{}", method="POST",
                                         headers={"X-Prep-Token": server.token, "Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            assert "error" not in data, (name, data)
        _say(f"selftest ok: page + 3 assets + 7 actions on port {port}")
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
          "roadmap": "roadmap", "star": "star", "fasttrack": "fasttrack", "patterns": "vault", "vault": "vault"}


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(prog="google-prep", description="Google SWE II OS desktop app")
    parser.add_argument("screen", nargs="?", default="today", help="open on: " + ", ".join(sorted(set(ROUTES))))
    parser.add_argument("--selftest", action="store_true", help="start, check, exit (for CI)")
    parser.add_argument("--no-browser", action="store_true", help="only serve; print the URL")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--version", action="store_true")
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
    if extra or args.screen not in ROUTES:
        parser.error(f"unknown command: {' '.join([args.screen, *extra]).strip()}")
    if args.no_browser and _windowed():
        _error_box("--no-browser needs a terminal. Run: google-prep-cli ui --no-browser")
        return 2
    mutex = _hold_app_mutex()  # held while this process shows the app
    try:
        server = serve(Api(), args.port)
    except Exception as exc:  # e.g. a damaged database or a port already in use
        _error_box(f"Google SWE II OS could not start:\n{type(exc).__name__}: {exc}")
        return 1
    url = server.url + ("" if ROUTES[args.screen] == "today" else f"#/{ROUTES[args.screen]}")
    proc = None if args.no_browser else open_window(url)
    if proc is None and not args.no_browser:
        webbrowser.open(url)
    _say(f"Google SWE II OS is running at {url}  (Ctrl+C to quit)")
    try:
        if proc is not None:
            started = time.monotonic()
            proc.wait()  # the window closed: we are done ...
            if time.monotonic() - started >= HANDOFF_SECONDS:
                return 0
            # ... unless the browser handed the page to a copy that was already running and exited.
            # That first copy marks the app as running; this window-less helper must not keep the
            # installer waiting after every window is closed.
            _release(mutex)
        while args.no_browser or time.monotonic() - server.last_seen < HEARTBEAT_TIMEOUT:
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0
