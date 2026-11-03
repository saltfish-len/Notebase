"""Long-running HTTP server: the two verbs over HTTP, plus the web UI.

The app (and with it the embedding model) is built once at start-up, so a find
takes milliseconds instead of reloading the model on every CLI call.

    GET  /        the web UI (one input for write and find, graph view)
    POST /write   {"text": str, "source"?: str, "files"?: [{"name": str, "data": base64}]}
                  -> {"capture": ..., "notes": [...]}
    POST /find    {"text": str, "k"?: int}
                  -> {"seeds", "notes", "activation", "edges", "clusters"}  (= notebase --json find)

Standard library only (no web framework to install). There is no
authentication: it listens on 127.0.0.1 unless told otherwise.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import logging
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from notebase.app import Config, Notebase
from notebase.jsonio import find_dict, write_dict
from notebase.models import Attachment
from notebase.storage.base import EmbeddingMismatch

try:
    from pymongo.errors import ConnectionFailure
except ImportError:  # pragma: no cover - pymongo is a dependency
    ConnectionFailure = OSError

log = logging.getLogger("notebase.server")

WEB_DIR = Path(__file__).parent / "web"
MAX_BODY = 64 * 1024 * 1024  # attachments arrive base64-encoded in the JSON body


class BadRequest(ValueError):
    pass


class NotebaseServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app: Notebase, files_dir: Path) -> None:
        super().__init__(address, Handler)
        self.app = app
        self.files_dir = files_dir
        # One request at a time through the pipelines: the embedding model and the
        # LLM boxes were written for a single caller, and this is a one-user app.
        self.lock = threading.Lock()

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0', '::') else host}:{port}/"


class Handler(BaseHTTPRequestHandler):
    server: NotebaseServer
    server_version = "Notebase"

    # ---------------------------------------------------------------- routes
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, (WEB_DIR / "index.html").read_bytes(), "text/html")
        elif path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "text/plain")
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        routes = {"/write": self._write, "/find": self._find}
        if path not in routes:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            body = self._read_json()
            with self.server.lock:
                result = routes[path](body)
            self._json(HTTPStatus.OK, result)
        except BadRequest as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))
        except ConnectionFailure:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "cannot reach MongoDB. Start it from the Notebase folder with: docker compose up -d",
            )
        except Exception as e:  # keep serving; the traceback goes to the log
            log.exception("%s failed", path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    def _write(self, body: dict) -> dict:
        text = body.get("text") or ""
        if not isinstance(text, str):
            raise BadRequest("text must be a string")
        source = body.get("source") or "typed"
        attachments = [self._save_file(f) for f in body.get("files") or []]
        if not text.strip() and not attachments:
            raise BadRequest("nothing to write")
        return write_dict(self.server.app.write(text, source=str(source), attachments=attachments))

    def _find(self, body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise BadRequest("text must be a non-empty string")
        try:
            k = int(body.get("k", 5))
        except (TypeError, ValueError):
            raise BadRequest("k must be an integer") from None
        return find_dict(self.server.app.find(text, k=max(1, min(k, 50))))

    # --------------------------------------------------------------- helpers
    def _save_file(self, f: dict) -> Attachment:
        """Keep an uploaded file on disk, so the capture can point at it."""
        if not isinstance(f, dict) or not f.get("name") or not isinstance(f.get("data"), str):
            raise BadRequest("each file needs a name and base64 data")
        try:
            data = base64.b64decode(f["data"], validate=True)
        except (binascii.Error, ValueError):
            raise BadRequest(f"file {f['name']!r} is not valid base64") from None
        name = Path(str(f["name"]).replace("\\", "/")).name or "file"
        digest = hashlib.sha256(data).hexdigest()
        target = self.server.files_dir / digest / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return Attachment.from_path(target)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise BadRequest("bad Content-Length") from None
        if length > MAX_BODY:
            raise BadRequest(f"request too large (max {MAX_BODY // 2**20} MB)")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise BadRequest("body must be JSON") from None
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        return body

    def _json(self, status: HTTPStatus, data) -> None:
        payload = json.dumps(data, default=str, ensure_ascii=False).encode()
        self._send(status, payload, "application/json")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _send(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s %s", self.address_string(), fmt % args)


def make_server(
    app: Notebase, host: str = "127.0.0.1", port: int = 8765, files_dir: str | Path | None = None
) -> NotebaseServer:
    files = Path(files_dir or Config().files_dir).expanduser()
    return NotebaseServer((host, port), app, files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="notebase-web", description="Serve write and find over HTTP, with a web UI."
    )
    parser.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 (this machine only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    from notebase.cli import _quiet_model_loading

    # requests to this server at INFO; other libraries (httpx under the Anthropic and
    # Hugging Face clients) only when something is wrong
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    _quiet_model_loading()
    config = Config()
    print("loading the embedding model and connecting to MongoDB…", file=sys.stderr)
    try:
        app = Notebase.from_config(config)
    except EmbeddingMismatch as e:
        print(f"notebase: {e}", file=sys.stderr)
        return 2
    except ConnectionFailure:
        print(
            f"notebase: cannot reach MongoDB at {config.mongo_uri}\n"
            "Is it running? Start it from the Notebase folder with: docker compose up -d",
            file=sys.stderr,
        )
        return 2
    try:
        server = make_server(app, args.host, args.port, config.files_dir)
    except OSError as e:
        print(f"notebase: cannot listen on {args.host}:{args.port}: {e.strerror}", file=sys.stderr)
        return 2
    print(f"Notebase is running at {server.url}  (Ctrl+C to stop)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
