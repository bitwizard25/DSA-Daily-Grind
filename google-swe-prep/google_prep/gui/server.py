"""Local HTTP server for the desktop window. Binds to 127.0.0.1 only and requires a per-launch
secret token on every API call, so other local programs and web pages cannot reach your data."""
import json
import mimetypes
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from google_prep.gui.api import Api, ApiError, call, dumps

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 2_000_000  # code + trace autosaves are small; anything bigger is a mistake


class PrepServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, api: Api, port: int = 0, token: str | None = None):
        self.api = api
        self.token = token or secrets.token_urlsafe(24)
        self.lock = threading.Lock()  # one API call at a time: SQLite + simple state
        self.last_seen = time.monotonic()
        super().__init__(("127.0.0.1", port), Handler)

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
        host = self.headers.get("Host", "")
        return host in (f"127.0.0.1:{self.server.server_address[1]}", f"localhost:{self.server.server_address[1]}")

    def _send(self, code: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
                             "connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        if not self._host_ok():
            return self._send(403, b'{"error": "forbidden"}')
        path = self.path.split("?", 1)[0]
        name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        file = (STATIC / name).resolve()
        if file.parent != STATIC or not file.is_file():
            return self._send(404, b'{"error": "not found"}')
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, file.read_bytes(), ctype)

    def do_POST(self):  # noqa: N802
        if not self._host_ok() or not secrets.compare_digest(self.headers.get("X-Prep-Token", ""), self.server.token):
            return self._send(403, b'{"error": "forbidden"}')
        if not self.path.startswith("/api/"):
            return self._send(404, b'{"error": "not found"}')
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
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
