"""Command line: ``notebase write`` and ``notebase find``. Nothing else."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

from notebase.app import Config, Notebase
from notebase.models import Attachment, FindResult, WriteResult
from notebase.storage.base import EmbeddingMismatch

try:
    from pymongo.errors import ConnectionFailure
except ImportError:  # pragma: no cover - pymongo is a dependency
    ConnectionFailure = OSError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="notebase", description="Write and find. That's it.")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="verb", required=True)

    w = sub.add_parser("write", help="put a thought in")
    w.add_argument("text", nargs="?", help="the thought (reads stdin if omitted and no files)")
    w.add_argument(
        "-a",
        "--attach",
        action="append",
        default=[],
        metavar="FILE",
        help="attach a file (image, audio, PDF, ...); repeat for several",
    )
    w.add_argument("--source", default="typed")

    f = sub.add_parser("find", help="recall by association")
    f.add_argument("cue", help="anything that might bring it back; #tag filters by tag, #all returns the whole graph")
    f.add_argument("-k", type=int, default=5, help="number of seed notes")

    args = parser.parse_args(argv)
    attachments: list[Attachment] = []
    if args.verb == "write":
        try:
            attachments = [Attachment.from_path(p) for p in args.attach]
        except OSError as e:
            parser.error(f"cannot attach {e.filename}: {e.strerror}")
    _quiet_model_loading()
    config = Config()
    try:
        return _run(parser, args, config, attachments)
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


def _run(parser, args, config: Config, attachments: list[Attachment]) -> int:
    app = Notebase.from_config(config)

    if args.verb == "write":
        text = args.text
        if text is None:
            text = "" if attachments else sys.stdin.read()
        if not text.strip() and not attachments:
            parser.error("nothing to write")
        result = app.write(text, source=args.source, attachments=attachments)
        _print_write(result, args.json)
    else:
        result = app.find(args.cue, k=args.k)
        _print_find(result, args.json)
    return 0


def _quiet_model_loading() -> None:
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass


def _print_write(result: WriteResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps([_note_json(n) for n in result.notes], default=str, ensure_ascii=False))
        return
    files = len(result.capture.attachments)
    with_files = f" with {files} file(s)" if files else ""
    print(f"captured {result.capture.id}{with_files} -> {len(result.notes)} atomic note(s)")
    for n in result.notes:
        print(f"  • {n.id}  tags={n.tags}  links={len(n.links)}")
        print(f"    {_short(n.content)}")


def _print_find(result: FindResult, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "seeds": result.seeds,
                    "notes": {i: _note_json(n) for i, n in result.notes.items()},
                    "activation": result.activation,
                    "edges": [asdict(e) for e in result.edges],
                    "clusters": [asdict(c) for c in result.clusters],
                },
                default=str,
                ensure_ascii=False,
            )
        )
        return
    if not result.notes:
        print("nothing comes to mind.")
        return
    seeds = set(result.seeds)
    for c in result.clusters:
        print(f"◆ {c.label}")
        for i in sorted(c.note_ids, key=lambda i: result.activation.get(i, 0), reverse=True):
            mark = "★" if i in seeds else "·"
            print(f"  {mark} [{result.activation.get(i, 0):.2f}] {_short(result.notes[i].content)}")
    print(f"\n{len(result.notes)} notes, {len(result.edges)} edges, {len(result.clusters)} clusters")


def _note_json(n) -> dict:
    d = asdict(n)
    d.pop("embedding", None)
    return d


def _short(text: str, width: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())
