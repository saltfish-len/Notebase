"""The *find* pipeline: a cue -> a zoomed-in, clustered region of the graph."""

from __future__ import annotations

from dataclasses import dataclass

from notebase.blackbox import Clusterer, Labeler, QueryInterpreter, Retriever, Spreader
from notebase.models import Cluster, Edge, FindResult
from notebase.storage.base import NoteStore


@dataclass
class FindPipeline:
    store: NoteStore
    interpreter: QueryInterpreter
    retriever: Retriever
    spreader: Spreader
    clusterer: Clusterer
    labeler: Labeler

    def find(self, text: str, k: int = 5) -> FindResult:
        # 1. interpret the cue
        query = self.interpreter.interpret(text)
        # 2. retrieve seed notes (hybrid search)
        seeds = self.retriever.retrieve(query, self.store, k)
        # 3. spread activation over the association graph
        activation = self.spreader.spread(seeds, self.store)
        notes = self.store.get_notes(activation)
        # 4. cluster the reached region
        groups = self.clusterer.cluster(notes)
        # 5. label clusters
        clusters = [Cluster(self.labeler.label([notes[i] for i in g]), g) for g in groups]

        return FindResult(
            query=query,
            seeds=[s.note.id for s in seeds],
            notes=notes,
            activation=activation,
            edges=_subgraph_edges(notes),
            clusters=clusters,
        )


def _subgraph_edges(notes) -> list[Edge]:
    """Edges whose both ends are inside the returned region (deduplicated)."""
    seen: set[tuple[str, str]] = set()
    edges: list[Edge] = []
    for note in notes.values():
        for link in note.links:
            key = tuple(sorted((note.id, link.to)))
            if link.to in notes and key not in seen:
                seen.add(key)
                edges.append(Edge(note.id, link.to, link.weight, link.reason))
    return edges
