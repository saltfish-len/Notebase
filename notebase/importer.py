"""Bulk import: an Obsidian vault (or any folder of Markdown) as many writes.

Import is *write* in a loop, not a third verb. Each ``.md`` file becomes one
capture with ``source="import"``, and the write pipeline splits it into
atomic notes as usual. On top of that, the importer keeps what the vault
already knows:

* **Frontmatter** is dropped from the text; its ``tags`` are added to every
  note from the file, and ``created``/``date`` (else the file's creation
  time) becomes the notes' ``created_at``.
* **Embeds** (``![[diagram.png]]``) become attachments, so images get alt
  text like any other write.
* **Wikilinks** (``[[Other note|alias]]``) become plain text, and after all
  files are written each one becomes a ``wikilink`` edge between the notes of
  the two files: you already said these thoughts belong together.
* **Re-running is safe.** A file whose text hasn't changed since its last
  import is skipped. A changed file is written again as a new capture; notes
  from the old version stay (memory is append-only).
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from notebase.models import Attachment, Link, Note, Origin

WIKILINK_WEIGHT = 0.9  # an explicit link is as strong as a near-duplicate

_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
_EMBED = re.compile(r"!\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_WIKILINK = re.compile(r"\[\[([^\]|#]*)(?:#([^\]|]*))?(?:\|([^\]]*))?\]\]")


@dataclass
class ParsedNote:
    """One Markdown file, ready to write."""

    text: str  # title + body, frontmatter removed, wikilinks as plain text
    tags: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)  # (target, text shown)
    embeds: list[str] = field(default_factory=list)  # file names, e.g. "diagram.png"
    created: datetime | None = None  # from frontmatter, if it says


def parse_note(raw: str, title: str) -> ParsedNote:
    """Turn Obsidian Markdown into text for the write pipeline."""
    meta: dict = {}
    match = _FRONTMATTER.match(raw)
    if match:
        try:
            loaded = yaml.safe_load(match.group(1))
            meta = loaded if isinstance(loaded, dict) else {}
        except yaml.YAMLError:
            pass
        raw = raw[match.end() :]

    embeds = [m.strip() for m in _EMBED.findall(raw)]
    body = _EMBED.sub("", raw)

    links: list[tuple[str, str]] = []

    def plain(m: re.Match) -> str:
        target, heading, alias = (g.strip() if g else "" for g in m.groups())
        shown = alias or target or heading
        if target:
            links.append((target, shown))
        return shown

    body = _WIKILINK.sub(plain, body).strip()

    # the file name is often the idea itself ("Dropout is denoising Bernoulli noises");
    # keep it on the first line, attached to the first paragraph
    first_line = body.split("\n", 1)[0].lstrip("# ").strip()
    if body and first_line.lower() != title.lower():
        text = f"{title}\n{body}"
    else:
        text = body or title

    return ParsedNote(
        text=text,
        tags=_tags(meta.get("tags") or meta.get("tag")),
        links=links,
        embeds=embeds,
        created=_date(meta.get("created") or meta.get("date")),
    )


def _tags(value) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value)
    if not isinstance(value, list):
        return []
    tags = [str(t).strip().lstrip("#").lower() for t in value]
    return list(dict.fromkeys(t for t in tags if t))


def _date(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return _date(datetime.fromisoformat(value.strip()))
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------- the vault


@dataclass
class ImportReport:
    written: list[str] = field(default_factory=list)  # relative paths
    unchanged: list[str] = field(default_factory=list)
    notes: int = 0
    wikilinks: int = 0
    unresolved: set[str] = field(default_factory=set)  # link targets with no file


def vault_files(root: Path, exclude: list[str] = ()) -> list[Path]:
    """Markdown files under ``root``, skipping hidden folders (.obsidian, .git,
    .trash, ...) and anything matching an ``exclude`` glob (relative path)."""
    files = []
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if any(fnmatch.fnmatch(str(rel), pat) or fnmatch.fnmatch(p.name, pat) for pat in exclude):
            continue
        files.append(p)
    return files


def import_vault(
    app,
    root: str | Path,
    exclude: list[str] = (),
    on_file: Callable[[str, str, int], None] | None = None,
) -> ImportReport:
    """Write every Markdown file under ``root``, then add its wikilinks as edges.

    ``on_file(rel_path, status, notes)`` is called per file, status being
    ``written`` or ``unchanged``. A file with no body is still a note: its
    name is often the whole idea.
    """
    root = Path(root).expanduser().resolve()
    store = app.store
    report = ImportReport()
    files = vault_files(root, list(exclude))
    by_name = _index_attachments(root)

    parsed: dict[str, ParsedNote] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        raw = path.read_text(encoding="utf-8", errors="replace")
        note = parse_note(raw, path.stem)
        parsed[rel] = note
        digest = hashlib.sha256(raw.encode()).hexdigest()

        if any(c.origin.sha256 == digest for c in store.captures_from(_uri(rel))):
            report.unchanged.append(rel)
            status, count = "unchanged", 0
        else:
            attachments = [
                Attachment.from_path(by_name[name.lower()])
                for name in note.embeds
                if name.lower() in by_name
            ]
            result = app.write(
                note.text,
                source="import",
                attachments=attachments,
                origin=Origin(uri=_uri(rel), sha256=digest),
                created_at=note.created or _file_created(path),
            )
            if note.tags:
                # re-read: later siblings mirrored links onto the earlier notes
                for n in store.get_notes([n.id for n in result.notes]).values():
                    n.tags = list(dict.fromkeys(n.tags + note.tags))
                    store.save_note(n)
            report.written.append(rel)
            status, count = "written", len(result.notes)
            report.notes += count
        if on_file:
            on_file(rel, status, count)

    # wikilinks, over every file in the vault (also unchanged ones: add_links is
    # idempotent, and a target may only have arrived in this run)
    resolve = _resolver(parsed)
    for rel, note in parsed.items():
        sources = _latest_notes(store, rel)
        for target, shown in note.links:
            target_rel = resolve(target, rel)
            if target_rel is None:
                if not Path(target).suffix or Path(target).suffix == ".md":
                    report.unresolved.add(target)
                continue
            if target_rel == rel:
                continue
            targets = _latest_notes(store, target_rel)
            for src, dst in _wikilink_pairs(sources, targets, shown):
                store.add_links(src.id, [Link(dst.id, WIKILINK_WEIGHT, "wikilink")])
                report.wikilinks += 1
    return report


def _uri(rel: str) -> str:
    return f"obsidian:{rel}"


def _latest_notes(store, rel: str) -> list[Note]:
    captures = store.captures_from(_uri(rel))
    return store.notes_of_capture(captures[-1].id) if captures else []


def _file_created(path: Path) -> datetime:
    st = path.stat()
    ts = getattr(st, "st_birthtime", None) or st.st_mtime  # birthtime on macOS
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _index_attachments(root: Path) -> dict[str, Path]:
    """Obsidian finds an embed by file name anywhere in the vault."""
    index: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and p.suffix != ".md" and not any(s.startswith(".") for s in rel.parts):
            index.setdefault(p.name.lower(), p)
    return index


def _resolver(parsed: dict[str, ParsedNote]) -> Callable[[str, str], str | None]:
    """Resolve ``[[target]]`` the way Obsidian does: a path relative to the
    vault, else a file name anywhere (the one closest to the linking file)."""
    by_path = {rel.lower().removesuffix(".md"): rel for rel in parsed}
    by_stem: dict[str, list[str]] = {}
    for rel in parsed:
        by_stem.setdefault(Path(rel).stem.lower(), []).append(rel)

    def resolve(target: str, from_rel: str) -> str | None:
        key = target.strip().lower().removesuffix(".md")
        if key in by_path:
            return by_path[key]
        candidates = by_stem.get(Path(key).name, [])
        if not candidates:
            return None
        here = Path(from_rel).parent.parts
        return max(candidates, key=lambda c: _shared_prefix(Path(c).parent.parts, here))

    return resolve


def _shared_prefix(a: tuple, b: tuple) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _wikilink_pairs(sources: list[Note], targets: list[Note], shown: str):
    """Which note of the linking file goes to which note of the linked file.

    A file can be split into several notes, so the edge goes from the notes
    that mention the link text (all of them if the atomizer rewrote it away)
    to the target note closest in meaning to each.
    """
    if not sources or not targets:
        return []
    mentioning = [s for s in sources if shown.lower() in s.content.lower()] or sources
    return [(s, max(targets, key=lambda t: _cosine(s.embedding, t.embedding))) for s in mentioning]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
