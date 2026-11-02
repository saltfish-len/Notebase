"""Storage contract. The pipelines only ever talk to this interface."""

from __future__ import annotations

from typing import Iterable, Protocol

from notebase.models import Capture, Link, Note, ScoredNote


class EmbeddingMismatch(RuntimeError):
    """The store holds vectors from a different embedding model."""


class NoteStore(Protocol):
    def bind_embedding(self, name: str, dim: int) -> None:
        """Pin the store to one embedding space; raise EmbeddingMismatch if it
        already holds vectors from another model (they are not comparable)."""
        ...

    # -- write side ------------------------------------------------------------
    def save_capture(self, capture: Capture) -> None: ...

    def save_note(self, note: Note) -> None: ...

    def add_links(self, note_id: str, links: Iterable[Link]) -> None:
        """Store links from ``note_id`` and the mirrored links back to it.

        Association is symmetric: if A reminds you of B, B reminds you of A.
        """
        ...

    # -- read side -------------------------------------------------------------
    def get_notes(self, ids: Iterable[str]) -> dict[str, Note]: ...

    def vector_search(
        self,
        embedding: list[float],
        k: int,
        exclude_ids: Iterable[str] = (),
        tags: Iterable[str] = (),
    ) -> list[ScoredNote]:
        """Nearest notes by embedding; ``score`` is cosine similarity in [-1, 1].

        May be eventually consistent: very recent writes might not show up yet.
        """
        ...

    def keyword_search(
        self, terms: Iterable[str], k: int, tags: Iterable[str] = ()
    ) -> list[ScoredNote]: ...

    def count(self) -> int: ...

    def top_tags(self, limit: int = 200) -> list[str]:
        """Most used tags, most frequent first (the tag vocabulary)."""
        ...
