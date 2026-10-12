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
    """Start the server, call a few actions over HTTP, check the page loads, exit. Used by CI."""
    server = serve(Api())
    port = server.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
        assert "<title>Google SWE II OS</title>" in page, "index.html missing"
        for name in ("state", "today", "roadmap", "problems", "star_list", "fasttrack", "vault_list"):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/{name}", data=b"{}", method="POST",
                                         headers={"X-Prep-Token": server.token, "Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            assert "error" not in data, (name, data)
        print(f"selftest ok: page + 7 actions on port {port}")
        return 0
    finally:
        server.shutdown()


def _say(message: str) -> None:
    """Print when there is a console (the windowed Windows build has none)."""
    if sys.stdout is not None:
        print(message, flush=True)


def _error_box(message: str) -> None:
    """The windowed build has no console, so a start-up failure must be shown as a dialog."""
    _say(message)
    if os.name == "nt" and sys.stdout is None:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "Google SWE II OS", 0x10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="google-prep", description="Google SWE II OS desktop app")
    parser.add_argument("--selftest", action="store_true", help="start, check, exit (for CI)")
    parser.add_argument("--no-browser", action="store_true", help="only serve; print the URL")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        from google_prep import __version__

        _say(f"google-prep {__version__}")
        return 0
    if args.selftest:
        return selftest()
    try:
        server = serve(Api(), args.port)
    except Exception as exc:  # e.g. a damaged database or a port already in use
        _error_box(f"Google SWE II OS could not start:\n{type(exc).__name__}: {exc}")
        return 1
    url = server.url
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
            # ... unless the browser handed the page to a copy that was already running and exited
        while args.no_browser or time.monotonic() - server.last_seen < HEARTBEAT_TIMEOUT:
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0
