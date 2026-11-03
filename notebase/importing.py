"""Bulk import of files and folders (Markdown/Obsidian vaults, text files, images).

Import is a batch form of *write*: each file becomes one capture that goes through the
normal write pipeline. What this module adds is turning a file into what its reader sees:

- Markdown, including Obsidian's syntax, becomes the text Obsidian shows when it renders the
  note: ``[[Note|alias]]`` reads ``alias``, ``[[Note#Heading]]`` reads ``Note > Heading``,
  front matter and ``%% comments %%`` disappear. Code blocks are left exactly as written.
- Embedded files (``![[photo.png|300]]``, ``![alt](img/photo.png)``) become attachments of the
  note they are embedded in, and stay in the text as ``![[photo.png]]``, where the Normalizer
  puts the file's description, so it stays next to the words around it. Such a file is not
  imported a second time on its own.
- Every file gets a fingerprint (hash of its bytes), so importing it again is skipped.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from notebase.blackbox.stubs import embed_placeholder

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".org", ".rst"}
MARKDOWN_SUFFIXES = {".md", ".markdown"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif"}
MAX_TEXT_BYTES = 200_000  # a larger text file is almost certainly not a note


@dataclass
class ImportItem:
    """One file, ready for ``Notebase.write``."""

    path: Path
    text: str
    attachments: list[Path] = field(default_factory=list)
    fingerprint: str = ""


# --------------------------------------------------------------------------- planning


def plan_import(paths: list[str], recursive: bool = True) -> tuple[list[Path], list[dict]]:
    """Expand paths and folders into the files to import, in a stable (sorted) order.

    Returns ``(files, skipped)``. Hidden paths inside a folder (``.obsidian``, ``.git``) and
    unsupported types are left out; images embedded in an imported Markdown file are left out
    too, because they come in as that note's attachments.
    """
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

    files: list[Path] = []
    for f in found:
        if f.suffix.lower() in TEXT_SUFFIXES and f.stat().st_size > MAX_TEXT_BYTES:
            skipped.append({"path": str(f), "reason": f"larger than {MAX_TEXT_BYTES} bytes"})
        else:
            files.append(f)

    vault = Vault.for_paths(paths)
    embedded: set[Path] = set()
    for f in files:
        if f.suffix.lower() in MARKDOWN_SUFFIXES:
            embedded.update(render_markdown(_read(f), f, vault)[1])
    return [f for f in files if f not in embedded], skipped


def prepare(path: Path, vault: "Vault") -> ImportItem:
    """Read one file into an ``ImportItem``. Raises ``ValueError`` for an empty file."""
    data = path.read_bytes()
    fingerprint = "file:" + hashlib.sha256(data).hexdigest()
    if path.suffix.lower() not in TEXT_SUFFIXES:  # an image on its own
        return ImportItem(path, "", [path], fingerprint)

    text = data.decode("utf-8", errors="replace")
    attachments: list[Path] = []
    if path.suffix.lower() in MARKDOWN_SUFFIXES:
        text, attachments = render_markdown(text, path, vault)
    text = text.strip()
    if not text and not attachments:
        raise ValueError("empty file")
    if text and not _HEADING.match(text):
        # give an untitled list or fragment its file name, so the Atomizer knows the topic
        text = f"# {path.stem}\n\n{text}"
    return ImportItem(path, text, attachments, fingerprint)


class Vault:
    """Finds embedded files the way Obsidian does: next to the note, from the vault root,
    then anywhere in the vault by file name."""

    def __init__(self, roots: list[Path]) -> None:
        self.roots = list(dict.fromkeys(roots))
        self._by_name: dict[str, Path] | None = None

    @classmethod
    def for_paths(cls, paths: list[str]) -> "Vault":
        """The vault an import of ``paths`` resolves embeds in: the folders given, and the
        folder of each file given on its own."""
        roots = [Path(p).expanduser().resolve() for p in paths]
        return cls([r if r.is_dir() else r.parent for r in roots if r.exists()])

    def resolve(self, target: str, note: Path) -> Path | None:
        target = target.strip()
        if not target:
            return None
        for base in [note.parent, *self.roots]:
            candidate = (base / target).resolve()
            if candidate.is_file():
                return candidate
        return self._names().get(Path(target).name.lower())

    def _names(self) -> dict[str, Path]:
        if self._by_name is None:
            self._by_name = {}
            for root in self.roots:
                for f in sorted(root.rglob("*")):
                    if f.is_file() and not _hidden(f.relative_to(root)):
                        self._by_name.setdefault(f.name.lower(), f)
        return self._by_name


# ------------------------------------------------------------------- markdown → text

_HEADING = re.compile(r"#{1,6}\s")
_FRONT_MATTER = re.compile(r"\A﻿?---[ \t]*\n.*?\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.S)
# fenced blocks and inline code are copied through untouched
_CODE = re.compile(r"(^[ \t]*```.*?^[ \t]*```[^\n]*$|^[ \t]*~~~.*?^[ \t]*~~~[^\n]*$|`[^`\n]+`)", re.M | re.S)
_COMMENT = re.compile(r"%%.*?%%", re.S)
_WIKILINK = re.compile(r"(!?)\[\[([^\[\]\n]+?)\]\]")
_MDLINK = re.compile(r"(!?)\[([^\]\n]*)\]\(\s*(<[^>\n]+>|[^)\s]+)(?:\s+[\"'][^\"'\n]*[\"'])?\s*\)")
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def render_markdown(text: str, note: Path, vault: Vault) -> tuple[str, list[Path]]:
    """Markdown/Obsidian source → (the text a reader sees, files embedded in it)."""
    embeds: list[Path] = []

    def embed(target: str) -> str | None:
        """Attach an embedded file if it exists and isn't a note; its placeholder if so."""
        path = vault.resolve(target, note)
        if path is None or path.suffix.lower() in TEXT_SUFFIXES:
            return None
        if path not in embeds:
            embeds.append(path)
        return embed_placeholder(path.name)

    def wikilink(m: re.Match) -> str:
        bang, inner = m.groups()
        target, _, alias = inner.replace("\\|", "|").partition("|")
        if bang:
            if placeholder := embed(target.partition("#")[0]):
                return placeholder  # the Normalizer puts the file's description here
            if _is_file_name(target) and target.rsplit(".", 1)[-1].lower() != "md":
                return Path(target).name  # a missing file: keep its name
        return alias.strip() if alias.strip() else _display(target)

    def mdlink(m: re.Match) -> str:
        bang, label, target = m.groups()
        target = unquote(target.strip("<>"))
        if _SCHEME.match(target):
            return m.group(0)  # external link: keep the URL, it is information
        if bang:
            if placeholder := embed(target.partition("#")[0]):
                return placeholder
            return label or Path(target).name
        return label

    def prose(part: str) -> str:
        part = _COMMENT.sub("", part)
        part = _WIKILINK.sub(wikilink, part)
        return _MDLINK.sub(mdlink, part)

    text = _FRONT_MATTER.sub("", text, count=1)
    parts = _CODE.split(text)
    rendered = "".join(p if i % 2 else prose(p) for i, p in enumerate(parts))
    rendered = re.sub(r"[ \t]+\n", "\n", rendered)  # a removed comment leaves trailing spaces
    return re.sub(r"\n{3,}", "\n\n", rendered), embeds


def _display(target: str) -> str:
    """How Obsidian shows a link without an alias: ``Note#A#B`` → ``Note > A > B``."""
    name, *sections = target.strip().split("#")
    shown = [s.strip() for s in ([name] if name.strip() else []) + sections if s.strip()]
    return " > ".join(shown)


def _is_file_name(target: str) -> bool:
    return "." in Path(target.partition("#")[0]).name


def _importable(p: Path) -> bool:
    return p.suffix.lower() in TEXT_SUFFIXES | IMAGE_SUFFIXES


def _hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")
