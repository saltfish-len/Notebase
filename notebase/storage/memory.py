"""In-process store for tests and offline demos. Same contract as MongoDB."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from notebase.models import Capture, Link, Note, ScoredNote
from notebase.storage.base import EmbeddingMismatch
from notebase.text import tokenize


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryNoteStore:
    def __init__(self) -> None:
        self.captures: dict[str, Capture] = {}
        self.notes: dict[str, Note] = {}
        self.embedding: tuple[str, int] | None = None

    def bind_embedding(self, name: str, dim: int) -> None:
        if self.embedding not in (None, (name, dim)):
            raise EmbeddingMismatch(f"store uses {self.embedding[0]}, not {name}")
        self.embedding = (name, dim)

    def save_capture(self, capture: Capture) -> None:
        self.captures[capture.id] = capture

    def save_note(self, note: Note) -> None:
        self.notes[note.id] = note

    def add_links(self, note_id: str, links: Iterable[Link]) -> None:
        src = self.notes[note_id]
        for link in links:
            if link.to not in self.notes or link.to == note_id:
                continue
            _upsert_link(src, link)
            _upsert_link(self.notes[link.to], Link(note_id, link.weight, link.reason))

    def get_notes(self, ids: Iterable[str]) -> dict[str, Note]:
        return {i: self.notes[i] for i in ids if i in self.notes}

    def vector_search(self, embedding, k, exclude_ids=(), tags=()):
        excluded, wanted = set(exclude_ids), set(tags)
        scored = [
            ScoredNote(n, _cosine(embedding, n.embedding))
            for n in self.notes.values()
            if n.id not in excluded and (not wanted or wanted & set(n.tags))
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    def keyword_search(self, terms, k, tags=()):
        terms = {t.lower() for t in terms}
        wanted = set(tags)
        scored = []
        for n in self.notes.values():
            if wanted and not wanted & set(n.tags):
                continue
            haystack = set(tokenize(n.content)) | {w.lower() for w in n.keywords}
            hits = len(terms & haystack)
            if hits:
                scored.append(ScoredNote(n, float(hits)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self.notes)

    def top_tags(self, limit: int = 200) -> list[str]:
        counts = Counter(t for n in self.notes.values() for t in n.tags)
        return [t for t, _ in counts.most_common(limit)]


def _upsert_link(note: Note, link: Link) -> None:
    for existing in note.links:
        if existing.to == link.to:
            existing.weight = max(existing.weight, link.weight)
            return
    note.links.append(link)
