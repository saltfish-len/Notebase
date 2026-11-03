"""Domain models shared by every layer of Notebase.

The atomic ``Note`` is the only unit of knowledge. Everything else is either
what goes in (``Capture``), what the black boxes produce (``Enrichment``,
``Link``, ``Query``) or what comes out of *find* (``FindResult``).
"""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- write


@dataclass
class Attachment:
    """A file that came with a capture (image, audio, PDF, ...).

    Only a reference is kept: where the file was, what it is, and its hash, so
    a later Normalizer can find it again and tell whether it changed. The bytes
    themselves are not copied into the store yet.
    """

    name: str  # file name as the user gave it, e.g. "whiteboard.jpg"
    media_type: str  # MIME type, e.g. "image/jpeg"; "application/octet-stream" if unknown
    path: str = ""  # absolute path at write time
    size: int = 0
    sha256: str = ""
    id: str = field(default_factory=lambda: new_id("a"))

    @property
    def kind(self) -> str:
        """Coarse kind: image | audio | video | document | file."""
        major, _, minor = self.media_type.partition("/")
        if major in ("image", "audio", "video"):
            return major
        if minor in ("pdf", "msword") or minor.startswith(
            ("vnd.openxmlformats-officedocument", "vnd.oasis.opendocument")
        ):
            return "document"
        return "file"

    @classmethod
    def from_path(cls, path: str | Path) -> "Attachment":
        p = Path(path).expanduser().resolve()
        data = p.read_bytes()
        media_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return cls(
            name=p.name,
            media_type=media_type,
            path=str(p),
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )


@dataclass
class Origin:
    """Where an imported capture came from, so a re-import can skip it.

    ``uri`` names the file, e.g. ``obsidian:Deep Learning/Linear probing.md``;
    ``sha256`` is the hash of its text at import time.
    """

    uri: str
    sha256: str
    imported_at: datetime = field(default_factory=utcnow)


@dataclass
class Capture:
    """Raw user input, stored verbatim for provenance.

    ``text`` is what the user typed and ``attachments`` are the files that came
    with it, both kept as given. ``normalized`` is the text-only version the
    Normalizer made from them; the boxes after it read ``content``.
    """

    text: str
    source: str = "typed"  # typed | pasted | voice | web-clip | import
    attachments: list[Attachment] = field(default_factory=list)
    normalized: str | None = None
    origin: Origin | None = None  # set for imports only
    id: str = field(default_factory=lambda: new_id("c"))
    created_at: datetime = field(default_factory=utcnow)

    @property
    def content(self) -> str:
        """The text the write boxes work on: normalized if available, else typed."""
        return self.normalized if self.normalized is not None else self.text


@dataclass
class Entity:
    type: str  # person | place | project | concept | url | ...
    name: str


@dataclass
class Enrichment:
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)


@dataclass
class Link:
    """A weighted, associative edge from one note to another."""

    to: str
    weight: float
    reason: str = "semantic"  # semantic | shared-entity | supersedes | ...


@dataclass
class Note:
    """One atomic, self-contained idea."""

    content: str
    capture_id: str
    source: str = "typed"
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("n"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class WriteResult:
    capture: Capture
    notes: list[Note]


# ---------------------------------------------------------------------------- find


@dataclass
class Query:
    text: str
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # optional metadata filter


@dataclass
class ScoredNote:
    note: Note
    score: float


@dataclass
class Edge:
    source: str
    target: str
    weight: float
    reason: str


@dataclass
class Cluster:
    label: str
    note_ids: list[str]


@dataclass
class FindResult:
    """A zoomed-in region of the knowledge graph.

    ``seeds`` are the notes the query landed on directly; ``notes`` holds the
    seeds plus everything reached by spreading over the graph; ``edges`` are the
    links between those notes; ``clusters`` group them into labelled regions.
    """

    query: Query
    seeds: list[str]
    notes: dict[str, Note]
    activation: dict[str, float]
    edges: list[Edge]
    clusters: list[Cluster]
