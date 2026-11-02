"""Domain models shared by every layer of Notebase.

The atomic ``Note`` is the only unit of knowledge. Everything else is either
what goes in (``Capture``), what the black boxes produce (``Enrichment``,
``Link``, ``Query``) or what comes out of *find* (``FindResult``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- write


@dataclass
class Capture:
    """Raw user input, stored verbatim for provenance."""

    text: str
    source: str = "typed"  # typed | pasted | voice | web-clip | import
    id: str = field(default_factory=lambda: new_id("c"))
    created_at: datetime = field(default_factory=utcnow)


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
