"""Placeholder implementations of the black boxes.

These are intentionally naive. They exist so the whole system runs end to end
while the real algorithms (LLM atomizing/enriching, a real embedding model,
smarter spreading and clustering) are developed. Replace one box at a time.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

from notebase.blackbox.interfaces import calibrate
from notebase.models import Capture, Enrichment, Entity, Link, Note, Query, ScoredNote
from notebase.storage.base import NoteStore
from notebase.text import tokenize

# ---------------------------------------------------------------------- write


class ParagraphAtomizer:
    """STUB: one paragraph (blank-line separated) = one atomic note."""

    def atomize(self, capture: Capture) -> list[str]:
        parts = re.split(r"\n\s*\n", capture.text.strip())
        return [p.strip() for p in parts if p.strip()]


class IdentityContextualizer:
    """STUB: leaves the fragment untouched."""

    def contextualize(self, fragment: str, capture: Capture) -> str:
        return fragment


_URL = re.compile(r"https?://\S+")
_HASHTAG = re.compile(r"#([\w\-]+)")


class FrequencyEnricher:
    """STUB: keywords = most frequent tokens; tags = #hashtags or top keywords."""

    def __init__(self, max_keywords: int = 8, max_tags: int = 2) -> None:
        self.max_keywords = max_keywords
        self.max_tags = max_tags

    def enrich(self, content: str) -> Enrichment:
        counts = Counter(tokenize(_URL.sub(" ", content)))
        keywords = [w for w, _ in counts.most_common(self.max_keywords)]
        hashtags = [h.lower() for h in _HASHTAG.findall(content)]
        tags = hashtags or keywords[: self.max_tags]
        entities = [Entity("url", u) for u in _URL.findall(content)]
        return Enrichment(keywords=keywords, tags=tags, entities=entities)


class HashingEmbedder:
    """STUB: bag-of-words hashing trick. Captures word overlap, not meaning.

    Kept for tests and offline use; the real one is in ``embedders.py``.
    """

    similarity_floor = 0.0

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-v1:{dim}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class NearestNeighborLinker:
    """STUB: link to the top-k most similar notes above a threshold.

    Link weight is the *calibrated* similarity (see ``calibrate``), so
    ``min_weight`` means the same thing whatever embedding model is in use.
    """

    def __init__(self, k: int = 5, min_weight: float = 0.2, similarity_floor: float = 0.0) -> None:
        self.k = k
        self.min_weight = min_weight
        self.similarity_floor = similarity_floor

    def link(self, note: Note, store: NoteStore, siblings: list[Note]) -> list[Link]:
        sibling_ids = [s.id for s in siblings]
        scores = {
            h.note.id: h.score
            for h in store.vector_search(note.embedding, self.k, exclude_ids=[note.id, *sibling_ids])
        }
        for s in siblings:
            scores[s.id] = _cosine(note.embedding, s.embedding)
        scores = {i: calibrate(c, self.similarity_floor) for i, c in scores.items()}
        best = sorted(scores, key=scores.get, reverse=True)[: self.k]
        return [
            Link(to=i, weight=round(scores[i], 4), reason="semantic")
            for i in best
            if scores[i] >= self.min_weight
        ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ----------------------------------------------------------------------- find


class SimpleQueryInterpreter:
    """STUB: keywords = tokens of the query; no expansion; #tags become filters."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def interpret(self, text: str) -> Query:
        tags = [h.lower() for h in _HASHTAG.findall(text)]
        clean = _HASHTAG.sub(" ", text)
        return Query(
            text=text,
            keywords=tokenize(clean),
            embedding=self.embedder.embed_query(clean),
            tags=tags,
        )


class RRFRetriever:
    """STUB-ish: reciprocal-rank fusion of vector search and keyword search."""

    def __init__(
        self, rrf_k: int = 60, min_vector_score: float = 0.05, similarity_floor: float = 0.0
    ) -> None:
        self.rrf_k = rrf_k
        self.min_vector_score = min_vector_score  # on the calibrated scale
        self.similarity_floor = similarity_floor

    def retrieve(self, query: Query, store: NoteStore, k: int) -> list[ScoredNote]:
        pool = k * 3
        vector = [
            s
            for s in store.vector_search(query.embedding, pool, tags=query.tags)
            if calibrate(s.score, self.similarity_floor) >= self.min_vector_score
        ]
        keyword = store.keyword_search(query.keywords, pool, tags=query.tags)

        fused: dict[str, float] = defaultdict(float)
        notes: dict[str, Note] = {}
        for ranking in (vector, keyword):
            for rank, hit in enumerate(ranking):
                fused[hit.note.id] += 1.0 / (self.rrf_k + rank + 1)
                notes[hit.note.id] = hit.note
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [ScoredNote(notes[i], fused[i]) for i in best]


class DecaySpreader:
    """STUB: breadth-first spreading activation with per-hop decay."""

    def __init__(
        self, hops: int = 2, decay: float = 0.6, min_activation: float = 0.05, max_notes: int = 50
    ) -> None:
        self.hops = hops
        self.decay = decay
        self.min_activation = min_activation
        self.max_notes = max_notes

    def spread(self, seeds: list[ScoredNote], store: NoteStore) -> dict[str, float]:
        if not seeds:
            return {}
        top = max(s.score for s in seeds) or 1.0
        activation = {s.note.id: s.score / top for s in seeds}
        known: dict[str, Note] = {s.note.id: s.note for s in seeds}
        frontier = list(activation)

        for _ in range(self.hops):
            reached: dict[str, float] = {}
            for nid in frontier:
                for link in known[nid].links:
                    a = activation[nid] * link.weight * self.decay
                    if a >= self.min_activation and a > activation.get(link.to, 0.0):
                        reached[link.to] = max(a, reached.get(link.to, 0.0))
            if not reached:
                break
            known.update(store.get_notes(i for i in reached if i not in known))
            reached = {i: a for i, a in reached.items() if i in known}
            activation.update(reached)
            frontier = list(reached)

        strongest = sorted(activation, key=activation.get, reverse=True)[: self.max_notes]
        return {i: activation[i] for i in strongest}


class ComponentClusterer:
    """STUB: connected components over links above a weight threshold."""

    def __init__(self, min_weight: float = 0.3) -> None:
        self.min_weight = min_weight

    def cluster(self, notes: dict[str, Note]) -> list[list[str]]:
        parent = {i: i for i in notes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for note in notes.values():
            for link in note.links:
                if link.to in notes and link.weight >= self.min_weight:
                    parent[find(note.id)] = find(link.to)

        groups: dict[str, list[str]] = defaultdict(list)
        for i in notes:
            groups[find(i)].append(i)
        return sorted(groups.values(), key=len, reverse=True)


class TopTagLabeler:
    """STUB: label = the most common tags in the cluster."""

    def label(self, notes: list[Note]) -> str:
        counts = Counter(t for n in notes for t in n.tags)
        top = [t for t, _ in counts.most_common(2)]
        return " / ".join(top) if top else "untitled"
