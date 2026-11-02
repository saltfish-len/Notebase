"""The black boxes.

Every core algorithm of Notebase sits behind one of these interfaces. The
pipelines in ``notebase.write`` / ``notebase.find`` only orchestrate them, so
each box can be swapped (stub -> heuristic -> LLM / ML model) independently.

Write:  Atomizer -> Contextualizer -> Enricher -> Embedder -> Linker
Find:   QueryInterpreter -> Retriever -> Spreader -> Clusterer -> Labeler
"""

from __future__ import annotations

from typing import Protocol

from notebase.models import Capture, Enrichment, Link, Note, Query, ScoredNote
from notebase.storage.base import NoteStore

# ---------------------------------------------------------------------- write


class Atomizer(Protocol):
    """Split a raw capture into fragments that each hold one idea."""

    def atomize(self, capture: Capture) -> list[str]: ...


class Contextualizer(Protocol):
    """Rewrite a fragment so it stands on its own (resolve "it", "that meeting")."""

    def contextualize(self, fragment: str, capture: Capture) -> str: ...


class Enricher(Protocol):
    """Produce keywords, tags and entities for an atomic note."""

    def enrich(self, content: str) -> Enrichment: ...


class Embedder(Protocol):
    """Map text into the vector space used for semantic search.

    Notes and queries are embedded separately because retrieval models are
    often asymmetric (a short cue vs. a full note).

    ``similarity_floor`` is the cosine similarity an *unrelated* pair typically
    gets with this model. Each model has its own noise level (hashing ~0,
    Qwen3 ~0.3, e5 ~0.75). Downstream boxes rescale with
    ``calibrate(cos, floor)`` so their thresholds work for any model.

    ``name`` identifies the vector space. Vectors from different names must
    never be mixed in one store.
    """

    name: str
    dim: int
    similarity_floor: float

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def calibrate(cosine: float, floor: float) -> float:
    """Map raw cosine to a model-independent association strength in [0, 1]."""
    return max(0.0, min(1.0, (cosine - floor) / (1.0 - floor)))


class Linker(Protocol):
    """Decide which existing notes a new note is associated with.

    ``siblings`` are notes already produced from the same capture. They are
    passed explicitly because the store's vector index may not have them yet.
    """

    def link(self, note: Note, store: NoteStore, siblings: list[Note]) -> list[Link]: ...


# ----------------------------------------------------------------------- find


class QueryInterpreter(Protocol):
    """Turn a user's cue into a structured query (keywords, vector, filters)."""

    def interpret(self, text: str) -> Query: ...


class Retriever(Protocol):
    """Hybrid search: find the seed notes the query lands on."""

    def retrieve(self, query: Query, store: NoteStore, k: int) -> list[ScoredNote]: ...


class Spreader(Protocol):
    """Spread activation from the seeds over the association graph.

    Returns ``note_id -> activation`` for every reached note, seeds included.
    """

    def spread(self, seeds: list[ScoredNote], store: NoteStore) -> dict[str, float]: ...


class Clusterer(Protocol):
    """Group the reached notes into coherent regions."""

    def cluster(self, notes: dict[str, Note]) -> list[list[str]]: ...


class Labeler(Protocol):
    """Give a cluster a short human-readable name."""

    def label(self, notes: list[Note]) -> str: ...
