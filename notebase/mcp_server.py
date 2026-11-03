"""MCP server: lets Claude Code (or any MCP client) write to and find in Notebase.

This is another front end for the same two verbs, like the CLI. The tools are:

- ``write``: one thought, optionally with files attached
- ``write_batch``: many thoughts in one call (bulk import of text the client already has)
- ``import_files``: bulk import of files and folders on this machine (Markdown, text, images),
  read here so the file contents never pass through the client's context
- ``find``: recall by association

``write_batch`` and ``import_files`` are batch forms of *write*, not new verbs. Each item is
its own capture, and one failing item does not stop the others.

Run it with ``notebase-mcp`` (stdio). The tool logic lives in ``NotebaseTools`` so it can be
tested without the ``mcp`` package; ``build_server`` only registers it.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from notebase.models import Attachment, FindResult, WriteResult
from notebase.storage.base import EmbeddingMismatch

try:
    from pymongo.errors import ConnectionFailure
except ImportError:  # pragma: no cover - pymongo is a dependency
    ConnectionFailure = OSError

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".org", ".rst"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif"}
MAX_TEXT_BYTES = 200_000  # a larger text file is almost certainly not a note
MAX_BATCH = 50  # items per call, so one call stays well inside client timeouts

INSTRUCTIONS = """\
Notebase is the user's associative memory. It has two verbs: write and find.

- write: store a thought. Pass the user's words as they are; Notebase splits the text into \
atomic notes, tags and links them itself. Do not add titles, folders or tags.
- write_batch: store several separate thoughts at once (each item becomes its own capture).
- import_files: import Markdown/text files or images from paths or folders on this machine. \
Run it with dry_run=true first to show the user what would be imported. It imports at most \
`limit` files per call; call again with the returned next_offset until it is null. \
Importing the same files twice creates duplicates.
- find: recall by association. Give a cue in natural language (any language); "#tag" filters \
by tag. Results come back as clusters of related notes; seed notes matched the cue directly, \
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
        files = [Attachment.from_path(p) for p in attachments or []]
        if not text.strip() and not files:
            raise ValueError("nothing to write: give text, attachments, or both")
        app = self.app
        with _unavailable_on_infra_errors():
            return _write_json(app.write(text, source=source, attachments=files))

    def write_batch(self, items: list[dict], source: str = "mcp") -> dict:
        """Store many thoughts; each item is ``{"text": ..., "attachments": [...], "source": ...}``
        (only ``text`` or ``attachments`` is required). Failures are reported per item."""
        if len(items) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} items per call; split the batch")
        results = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                item = {"text": item}
            try:
                r = self.write(
                    text=item.get("text", ""),
                    attachments=item.get("attachments"),
                    source=item.get("source", source),
                )
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
        contents, each image one capture with the image attached. Hidden files and folders
        (e.g. .obsidian, .git) are skipped. Files are taken in sorted order, ``limit`` per call."""
        limit = max(1, min(limit, MAX_BATCH))
        found, skipped = collect_files(paths, recursive)
        page = found[offset : offset + limit]
        next_offset = offset + limit if offset + limit < len(found) else None
        head = {"total": len(found), "offset": offset, "next_offset": next_offset}
        if dry_run:
            return {**head, "dry_run": True, "files": [str(p) for p in page], "skipped": skipped}

        results = []
        for p in page:
            try:
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    r = self.write("", attachments=[str(p)], source="import")
                else:
                    r = self.write(_read_text(p), source="import")
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
        grouped into labelled clusters."""
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


# ---------------------------------------------------------------------------- files


def collect_files(paths: list[str], recursive: bool = True) -> tuple[list[Path], list[dict]]:
    """Expand paths and folders into importable files (sorted, deduplicated)."""
    found: dict[Path, None] = {}
    skipped: list[dict] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            walker = p.rglob("*") if recursive else p.glob("*")
            for f in sorted(walker):
                if f.is_file() and not _hidden(f.relative_to(p)) and _importable(f):
                    found.setdefault(f)
        elif p.is_file():
            if _importable(p):
                found.setdefault(p)
            else:
                skipped.append({"path": str(p), "reason": "unsupported file type"})
        else:
            skipped.append({"path": str(p), "reason": "not found"})

    kept = []
    for f in found:
        if f.suffix.lower() in TEXT_SUFFIXES and f.stat().st_size > MAX_TEXT_BYTES:
            skipped.append({"path": str(f), "reason": f"larger than {MAX_TEXT_BYTES} bytes"})
        else:
            kept.append(f)
    return kept, skipped


def _importable(p: Path) -> bool:
    return p.suffix.lower() in TEXT_SUFFIXES | IMAGE_SUFFIXES


def _hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def _read_text(p: Path) -> str:
    """File contents, with the file name as a heading when the text has none, so the
    Atomizer knows what an untitled list or fragment is about."""
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("empty file")
    if not text.startswith("#"):
        text = f"# {p.stem}\n\n{text}"
    return text


# ----------------------------------------------------------------------------- json


def _write_json(result: WriteResult) -> dict:
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
    ok = [r for r in results if r["ok"]]
    return {
        "written": len(ok),
        "failed": len(results) - len(ok),
        "notes": sum(len(r["notes"]) for r in ok),
        "results": results,
    }


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
