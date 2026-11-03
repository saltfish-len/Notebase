"""MCP server: lets Claude Code (or any MCP client) write to and find in Notebase.

This is another front end for the same two verbs, like the CLI. The tools are:

- ``write``: one thought, optionally with files attached
- ``write_batch``: many thoughts in one call (bulk import of text the client already has)
- ``import_files``: bulk import of files and folders on this machine (Markdown and Obsidian
  vaults, text, images; see ``notebase.importing``), read here so the file contents never pass
  through the client's context
- ``find``: recall by association

``write_batch`` and ``import_files`` are batch forms of *write*, not new verbs. Each item is
its own capture, one failing item does not stop the others, and an item that was already
written (same file contents, same batch item) is skipped.

Run it with ``notebase-mcp`` (stdio). The tool logic lives in ``NotebaseTools`` so it can be
tested without the ``mcp`` package; ``build_server`` only registers it.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from notebase.importing import Vault, plan_import, prepare
from notebase.models import Attachment, FindResult, WriteResult
from notebase.storage.base import EmbeddingMismatch

try:
    from pymongo.errors import ConnectionFailure
except ImportError:  # pragma: no cover - pymongo is a dependency
    ConnectionFailure = OSError

log = logging.getLogger(__name__)

MAX_BATCH = 50  # items per call, so one call stays well inside client timeouts

INSTRUCTIONS = """\
Notebase is the user's associative memory. It has two verbs: write and find.

- write: store a thought. Pass the user's words as they are; Notebase splits the text into \
atomic notes, tags and links them itself. Do not add titles, folders or tags.
- write_batch: store several separate thoughts at once (each item becomes its own capture). \
An item identical to one written by an earlier batch is skipped, so a batch can be retried.
- import_files: import Markdown/text files or images from paths or folders on this machine, \
including Obsidian vaults (links become the text Obsidian shows, embedded images are \
attached to their note). Run it with dry_run=true first to show the user what would be \
imported. It imports at most `limit` files per call; call again with the returned \
next_offset until it is null. Files imported before (same contents) are skipped, so an \
interrupted import can simply be run again.
- find: recall by association. Give a cue in natural language (any language); "#tag" filters \
by tag, and a cue of tags only returns every note with those tags. "#all" returns the whole \
graph (all notes and links), "#all #tag" every note with that tag; use it only when the user \
asks for everything, since the result can be large. \
Results come back as clusters of related notes; seed notes matched the cue directly, \
the rest were reached through links.
"""


class Unavailable(RuntimeError):
    """Notebase itself cannot work (database down, wrong embedding model). Unlike a bad
    item, this stops a batch: every other item would fail the same way."""


@dataclass
class NotebaseTools:
    """The MCP tools as plain methods. ``factory`` builds the app on first use, so the
    server starts at once and the embedding model loads only when a tool needs it."""

    factory: Callable[[], Any]
    _app: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def app(self):
        with self._lock:
            if self._app is None:
                with _unavailable_on_infra_errors():
                    self._app = self.factory()
            return self._app

    # ------------------------------------------------------------------ write
    def write(
        self, text: str = "", attachments: list[str] | None = None, source: str = "mcp"
    ) -> dict:
        """Store one thought (text, files, or both). Returns the atomic notes it became."""
        return self._write(text, [Attachment.from_path(p) for p in attachments or []], source)

    def _write(
        self, text: str, files: list[Attachment], source: str, fingerprint: str | None = None
    ) -> dict:
        if not text.strip() and not files:
            raise ValueError("nothing to write: give text, attachments, or both")
        app = self.app
        with _unavailable_on_infra_errors():
            result = app.write(text, source=source, attachments=files, fingerprint=fingerprint)
        return _write_json(result)

    def write_batch(
        self, items: list[dict], source: str = "mcp", skip_duplicates: bool = True
    ) -> dict:
        """Store many thoughts; each item is ``{"text": ..., "attachments": [...], "source": ...}``
        (only ``text`` or ``attachments`` is required). Failures are reported per item. An item
        with the same text and files as one written by an earlier batch is skipped (reported as
        ``duplicate``) unless ``skip_duplicates`` is false."""
        if len(items) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} items per call; split the batch")
        results = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                item = {"text": item}
            try:
                text = item.get("text", "")
                files = [Attachment.from_path(p) for p in item.get("attachments") or []]
                fp = _batch_fingerprint(text, files) if skip_duplicates else None
                r = self._write(text, files, item.get("source", source), fp)
                results.append({"index": i, "ok": True, **r})
            except Unavailable:
                raise
            except Exception as e:  # one bad item must not stop the batch
                log.warning("write_batch item %d failed: %s", i, e)
                results.append({"index": i, "ok": False, "error": _err(e)})
        return _batch_summary(results)

    def import_files(
        self,
        paths: list[str],
        recursive: bool = True,
        dry_run: bool = False,
        offset: int = 0,
        limit: int = MAX_BATCH,
    ) -> dict:
        """Import files and folders: each text file (.md, .txt, ...) becomes one capture of its
        contents, each image one capture with the image attached. Markdown is imported as the
        text Obsidian shows: [[Note|alias]] reads "alias", [[Note#Heading]] "Note > Heading",
        front matter and %%comments%% are dropped, code is kept as written, and embedded images
        (![[photo.png]], ![](photo.png)) are attached to their note instead of imported alone.
        Hidden files and folders (e.g. .obsidian, .git) are skipped. A file whose contents were
        imported before is skipped (reported as ``duplicate``). Files are taken in sorted order,
        ``limit`` per call. ``dry_run`` lists the files without checking for duplicates."""
        limit = max(1, min(limit, MAX_BATCH))
        found, skipped = plan_import(paths, recursive)
        page = found[offset : offset + limit]
        next_offset = offset + limit if offset + limit < len(found) else None
        head = {"total": len(found), "offset": offset, "next_offset": next_offset}
        if dry_run:
            return {**head, "dry_run": True, "files": [str(p) for p in page], "skipped": skipped}

        vault = Vault.for_paths(paths)
        results = []
        for p in page:
            try:
                item = prepare(p, vault)
                files = [Attachment.from_path(a) for a in item.attachments]
                r = self._write(item.text, files, "import", item.fingerprint)
                results.append({"path": str(p), "ok": True, **r})
            except Unavailable:
                raise
            except Exception as e:
                log.warning("import of %s failed: %s", p, e)
                results.append({"path": str(p), "ok": False, "error": _err(e)})
        return {**head, **_batch_summary(results), "skipped": skipped}

    # ------------------------------------------------------------------- find
    def find(self, cue: str, k: int = 5) -> dict:
        """Recall by association: seed notes for the cue plus notes reached over links,
        grouped into labelled clusters. "#tag" filters by tag; "#all" returns the whole graph
        (every note and link), "#all #tag" every note with that tag; ``k`` is ignored then."""
        if not cue.strip():
            raise ValueError("empty cue")
        app = self.app
        with _unavailable_on_infra_errors():
            return _find_json(app.find(cue, k=max(1, min(k, 50))))


class _unavailable_on_infra_errors:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and issubclass(exc_type, ConnectionFailure):
            raise Unavailable(
                "cannot reach MongoDB. Is it running? Start it from the Notebase folder "
                "with: docker compose up -d"
            ) from exc
        if exc_type is not None and issubclass(exc_type, EmbeddingMismatch):
            raise Unavailable(str(exc)) from exc
        return False


# ----------------------------------------------------------------------------- json


def _write_json(result: WriteResult) -> dict:
    if result.duplicate:
        return {"capture_id": result.capture.id, "duplicate": True, "notes": []}
    return {
        "capture_id": result.capture.id,
        "notes": [
            {"id": n.id, "content": n.content, "tags": n.tags, "links": len(n.links)}
            for n in result.notes
        ],
    }


def _find_json(result: FindResult) -> dict:
    seeds = set(result.seeds)

    def note(i: str) -> dict:
        n = result.notes[i]
        return {
            "id": n.id,
            "content": n.content,
            "tags": n.tags,
            "activation": round(result.activation.get(i, 0.0), 3),
            "seed": i in seeds,
            "created_at": n.created_at.isoformat(),
        }

    clusters = [
        {
            "label": c.label,
            "notes": [
                note(i)
                for i in sorted(c.note_ids, key=lambda i: result.activation.get(i, 0), reverse=True)
            ],
        }
        for c in result.clusters
    ]
    return {
        "clusters": clusters,
        "edges": [
            {"source": e.source, "target": e.target, "weight": round(e.weight, 3)}
            for e in result.edges
        ],
        "total_notes": len(result.notes),
    }


def _batch_summary(results: list[dict]) -> dict:
    ok = [r for r in results if r["ok"] and not r.get("duplicate")]
    duplicates = sum(1 for r in results if r.get("duplicate"))
    return {
        "written": len(ok),
        "duplicates": duplicates,
        "failed": sum(1 for r in results if not r["ok"]),
        "notes": sum(len(r["notes"]) for r in ok),
        "results": results,
    }


def _batch_fingerprint(text: str, files: list[Attachment]) -> str:
    h = hashlib.sha256(text.strip().encode())
    for f in files:
        h.update(b"\0" + f.sha256.encode())
    return "item:" + h.hexdigest()


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- server


def build_server(tools: NotebaseTools):
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server
        from mcp.server.fastmcp.exceptions import ToolError

    def reported(fn):
        # The SDK hides the text of unexpected exceptions from the client; the client is the
        # user's own assistant on the user's machine, so pass the reason on for it to act on.
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                log.warning("%s failed: %s", fn.__name__, e)
                raise ToolError(str(e) if isinstance(e, Unavailable) else _err(e)) from e

        return wrapper

    server = Server("notebase", instructions=INSTRUCTIONS)
    for fn in (tools.write, tools.write_batch, tools.import_files, tools.find):
        server.tool(name=fn.__name__, description=fn.__doc__)(reported(fn))
    return server


def main() -> None:
    # stdout is the MCP channel: keep every log and progress bar on stderr
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from notebase.app import Notebase
    from notebase.cli import _quiet_model_loading

    _quiet_model_loading()
    build_server(NotebaseTools(factory=Notebase.from_config)).run("stdio")


if __name__ == "__main__":
    main()
