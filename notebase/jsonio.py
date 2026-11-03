"""JSON shapes for write and find results, shared by the CLI and the HTTP server."""

from __future__ import annotations

from dataclasses import asdict

from notebase.models import FindResult, Note, WriteResult


def note_dict(note: Note) -> dict:
    """A note without its embedding (large, and useless to a reader)."""
    d = asdict(note)
    d.pop("embedding", None)
    return d


def write_dict(result: WriteResult) -> dict:
    return {"capture": asdict(result.capture), "notes": [note_dict(n) for n in result.notes]}


def find_dict(result: FindResult) -> dict:
    """Same shape as ``notebase --json find``: everything a graph view needs."""
    return {
        "seeds": result.seeds,
        "notes": {i: note_dict(n) for i, n in result.notes.items()},
        "activation": result.activation,
        "edges": [asdict(e) for e in result.edges],
        "clusters": [asdict(c) for c in result.clusters],
    }
