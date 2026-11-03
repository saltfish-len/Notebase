"""The HTTP server: POST /write and /find, the web UI at /. In-memory store, stubs."""

import base64
import json
import threading
import urllib.error
import urllib.request

import pytest

from notebase import Notebase
from notebase.server import make_server
from notebase.storage import InMemoryNoteStore


@pytest.fixture
def server(tmp_path):
    app = Notebase.with_stubs(InMemoryNoteStore(), embedding_dim=128)
    srv = make_server(app, "127.0.0.1", 0, files_dir=tmp_path / "files")
    thread = threading.Thread(target=srv.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(srv, path, body=None):
    host, port = srv.server_address[:2]
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_ui_is_served(server):
    status, body = call(server, "/")
    assert status == 200 and b"<title>Notebase</title>" in body


def test_write_then_find(server):
    status, body = call(server, "/write", {"text": "Coffee late in the afternoon hurts deep sleep."})
    assert status == 200
    written = json.loads(body)
    note_id = written["notes"][0]["id"]
    assert "embedding" not in written["notes"][0]
    assert written["capture"]["text"].startswith("Coffee")

    call(server, "/write", {"text": "Deep sleep consolidates memory in the hippocampus."})
    status, body = call(server, "/find", {"text": "coffee"})
    assert status == 200
    found = json.loads(body)
    # same shape as `notebase --json find`
    assert set(found) == {"seeds", "notes", "activation", "edges", "clusters"}
    assert found["seeds"][0] == note_id
    assert note_id in found["notes"] and "embedding" not in found["notes"][note_id]


def test_write_with_file_keeps_a_copy(server, tmp_path):
    data = base64.b64encode(b"\x89PNG fake bytes").decode()
    status, body = call(
        server, "/write", {"text": "", "files": [{"name": "../whiteboard_plan.png", "data": data}]}
    )
    assert status == 200
    attachment = json.loads(body)["capture"]["attachments"][0]
    assert attachment["name"] == "whiteboard_plan.png"  # no path tricks
    saved = tmp_path / "files" / attachment["sha256"] / "whiteboard_plan.png"
    assert saved.read_bytes() == b"\x89PNG fake bytes"
    assert attachment["path"] == str(saved.resolve())


@pytest.mark.parametrize(
    "path, body",
    [
        ("/write", {"text": "   "}),
        ("/find", {"text": ""}),
        ("/find", {"text": "x", "k": "many"}),
        ("/write", {"files": [{"name": "a.png", "data": "not base64!"}]}),
    ],
)
def test_bad_requests(server, path, body):
    status, raw = call(server, path, body)
    assert status == 400 and "error" in json.loads(raw)


def test_unknown_route(server):
    assert call(server, "/delete", {})[0] == 404
