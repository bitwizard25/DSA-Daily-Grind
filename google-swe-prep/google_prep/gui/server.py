"""Local HTTP server for the desktop window. Binds to 127.0.0.1 only and requires a per-launch
secret token on every API call, so other local programs and web pages cannot reach your data."""
import json
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from google_prep.gui.api import Api, ApiError, call, dumps

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 2_000_000  # code + trace autosaves are small; anything bigger is a mistake
WINDOW_CLOSE_GRACE = 5  # allow a reload/new document to register before ending the server
# Fixed types for the bundled files. Not `mimetypes`: on Windows it reads the registry, where other
# software often maps .js to text/plain, and with nosniff the window would then refuse to run app.js.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


class PrepServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR would let a process of another account bind the same 127.0.0.1 port
    # and receive the window's requests; claim the port exclusively there instead.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        # Only for the automatic port (the normal launch): an exclusive port stays blocked while old
        # connections linger, which would make a quick restart with a fixed --port fail for minutes.
        if os.name == "nt" and self.server_address[1] == 0 and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, api: Api, port: int = 0, token: str | None = None):
        self.api = api
        self.token = token or secrets.token_urlsafe(24)
        self.lock = threading.Lock()  # one API call at a time: SQLite + simple state
        self.last_seen = time.monotonic()
        self._window_lock = threading.Lock()
        self._windows = {}
        self._windows_closed_at = None
        self._window_tracking_failed = False
        super().__init__(("127.0.0.1", port), Handler)

    def window_presence(self, payload):
        if not isinstance(payload, dict):
            raise ApiError('Invalid window presence')
        identity, is_open = payload.get('id'), payload.get('open')
        if not isinstance(identity, str) or not 1 <= len(identity) <= 100 or type(is_open) is not bool:
            raise ApiError('Invalid window presence')
        with self._window_lock:
            if identity not in self._windows and len(self._windows) >= 1024:
                self._window_tracking_failed = True  # fail safely: retain the server, never guess closed
                raise ApiError('Window tracking limit reached')
            # A delayed open message from an unloaded document must not resurrect it.
            if self._windows.get(identity) is False:
                return {'ok':True}
            self._windows[identity] = is_open
            if any(self._windows.values()):
                self._windows_closed_at = None
            elif self._windows_closed_at is None:
                self._windows_closed_at = time.monotonic()
        return {'ok':True}

    def windows_closed(self):
        with self._window_lock:
            return (not self._window_tracking_failed and self._windows_closed_at is not None
                    and time.monotonic() - self._windows_closed_at >= WINDOW_CLOSE_GRACE)

    def handle_error(self, request, client_address):
        import sys

        if isinstance(sys.exc_info()[1], ConnectionError):
            return  # the window reloaded or closed a kept-alive connection: not an error
        super().handle_error(request, client_address)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/?t={self.token}"


class Handler(BaseHTTPRequestHandler):
    server: PrepServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: this is a desktop app, not a web server
        pass

    def _host_ok(self) -> bool:
        # DNS-rebinding guard: only accept our own address as the Host header
        host = self.headers.get("Host", "").lower()
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:  # browsers leave the default port out of the Host header
            allowed |= {"127.0.0.1", "localhost"}
        return host in allowed

    def _send(self, code: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
                             "connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page reloaded or closed mid-request: nothing to answer

    def do_GET(self):  # noqa: N802 - http.server API
        if not self._host_ok():
            return self._send(403, b'{"error": "forbidden"}')
        path = self.path.split("?", 1)[0]
        name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        file = (STATIC / name).resolve()
        ctype = CONTENT_TYPES.get(file.suffix.lower())
        if file.parent != STATIC.resolve() or not ctype or not file.is_file():
            return self._send(404, b'{"error": "not found"}')
        self._send(200, file.read_bytes(), ctype)

    def _reject_post(self, code, body):
        # Closing a Windows socket with an unread POST body can reset it before the error arrives.
        # Drain only small, complete requests; never wait indefinitely for an unauthenticated peer.
        self.close_connection = True
        previous_timeout = self.connection.gettimeout()
        try:
            length = int(self.headers.get('Content-Length') or 0)
            if 0 < length <= 4096:
                self.connection.settimeout(1)
                self.rfile.read(length)
        except (OSError, ValueError):
            pass
        finally:
            self.connection.settimeout(previous_timeout)
        return self._send(code, body)

    def do_POST(self):  # noqa: N802
        if not self._host_ok() or not secrets.compare_digest(self.headers.get("X-Prep-Token", ""), self.server.token):
            return self._reject_post(403, b'{"error": "forbidden"}')
        if not self.path.startswith("/api/"):
            return self._reject_post(404, b'{"error": "not found"}')
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length < 0:
                raise ValueError
        except ValueError:
            self.close_connection = True
            return self._send(400, b'{"error": "bad content length"}')
        if length > MAX_BODY:
            self.close_connection = True
            return self._send(413, b'{"error": "too large"}')
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._send(400, b'{"error": "bad json"}')
        self.server.last_seen = time.monotonic()
        try:
            with self.server.lock:
                name = self.path[len("/api/"):]
                result = self.server.window_presence(payload) if name == 'window_presence' else call(self.server.api, name, payload)
            self._send(200, dumps(result))
        except ApiError as exc:
            self._send(400, dumps({"error": str(exc)}))
        except Exception as exc:  # keep the window alive and show the problem
            self._send(500, dumps({"error": f"{type(exc).__name__}: {exc}"}))


def serve(api: Api | None = None, port: int = 0) -> PrepServer:
    """Start the server on a background thread and return it."""
    server = PrepServer(api or Api(), port)
    threading.Thread(target=server.serve_forever, name="prep-gui", daemon=True).start()
    return server
