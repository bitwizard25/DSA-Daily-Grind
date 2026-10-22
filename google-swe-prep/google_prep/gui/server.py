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
        super().__init__(("127.0.0.1", port), Handler)

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

    def do_POST(self):  # noqa: N802
        if not self._host_ok() or not secrets.compare_digest(self.headers.get("X-Prep-Token", ""), self.server.token):
            self.close_connection = True  # unread POST data must not become the next HTTP method
            return self._send(403, b'{"error": "forbidden"}')
        if not self.path.startswith("/api/"):
            self.close_connection = True
            return self._send(404, b'{"error": "not found"}')
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
                result = call(self.server.api, self.path[len("/api/"):], payload)
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
