"""Command line: ``notebase write`` and ``notebase find``, plus ``notebase import``
(write in a loop, for an Obsidian vault or a folder of Markdown)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from notebase.app import Config, Notebase
from notebase.importer import import_vault, parse_note, vault_files
from notebase.jsonio import find_dict, note_dict
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
    f.add_argument("cue", help="anything that might bring it back; #tag filters by tag")
    f.add_argument("-k", type=int, default=5, help="number of seed notes")

    i = sub.add_parser("import", help="write every Markdown file in a folder (Obsidian vault)")
    i.add_argument("folder", help="the vault or folder to import")
    i.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="skip files matching this glob (file name or path in the vault); repeat for several",
    )
    i.add_argument("--dry-run", action="store_true", help="list the files, write nothing")

    args = parser.parse_args(argv)
    if args.verb == "import":
        if not Path(args.folder).expanduser().is_dir():
            parser.error(f"not a folder: {args.folder}")
        if args.dry_run:
            return _dry_run(args)
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
    elif args.verb == "import":
        _run_import(app, args)
    else:
        result = app.find(args.cue, k=args.k)
        _print_find(result, args.json)
    return 0


def _dry_run(args) -> int:
    root = Path(args.folder).expanduser().resolve()
    files = vault_files(root, args.exclude)
    for p in files:
        note = parse_note(p.read_text(encoding="utf-8", errors="replace"), p.stem)
        extra = f"  +{len(note.embeds)} file(s)" if note.embeds else ""
        print(f"{p.relative_to(root)}  ({len(note.text)} chars, {len(note.links)} links){extra}")
    print(f"\n{len(files)} file(s) would be imported (nothing written)")
    return 0


def _run_import(app: Notebase, args) -> None:
    marks = {"written": "+", "unchanged": "="}

    def on_file(rel: str, status: str, notes: int) -> None:
        if not args.json:
            print(f"{marks[status]} {rel}" + (f"  -> {notes} note(s)" if notes else ""))

    report = import_vault(app, args.folder, args.exclude, on_file=on_file)
    if args.json:
        summary = {
            "written": report.written,
            "unchanged": report.unchanged,
            "notes": report.notes,
            "wikilinks": report.wikilinks,
            "unresolved": sorted(report.unresolved),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return
    print(
        f"\n{len(report.written)} file(s) written -> {report.notes} note(s); "
        f"{len(report.unchanged)} unchanged; "
        f"{report.wikilinks} wikilink edge(s)"
    )
    if report.unresolved:
        print(f"links to notes that don't exist: {', '.join(sorted(report.unresolved))}")


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
        print(json.dumps([note_dict(n) for n in result.notes], default=str, ensure_ascii=False))
        return
    files = len(result.capture.attachments)
    with_files = f" with {files} file(s)" if files else ""
    print(f"captured {result.capture.id}{with_files} -> {len(result.notes)} atomic note(s)")
    for n in result.notes:
        print(f"  • {n.id}  tags={n.tags}  links={len(n.links)}")
        print(f"    {_short(n.content)}")


def _print_find(result: FindResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(find_dict(result), default=str, ensure_ascii=False))
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


def _short(text: str, width: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())
