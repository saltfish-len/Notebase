"""Composition root: wires storage + black boxes into the two verbs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from notebase.blackbox import stubs
from notebase.find import FindPipeline
from notebase.models import FindResult, WriteResult
from notebase.storage.base import NoteStore
from notebase.write import WritePipeline


@dataclass
class Config:
    mongo_uri: str = field(
        default_factory=lambda: os.environ.get(
            "NOTEBASE_MONGO_URI", "mongodb://localhost:27017/?directConnection=true"
        )
    )
    database: str = field(default_factory=lambda: os.environ.get("NOTEBASE_DB", "notebase"))
    embedding_dim: int = field(
        default_factory=lambda: int(os.environ.get("NOTEBASE_EMBED_DIM", "256"))
    )


class Notebase:
    """The whole app: exactly two public operations."""

    def __init__(self, write_pipeline: WritePipeline, find_pipeline: FindPipeline) -> None:
        self._write = write_pipeline
        self._find = find_pipeline

    def write(self, text: str, source: str = "typed") -> WriteResult:
        return self._write.write(text, source)

    def find(self, text: str, k: int = 5) -> FindResult:
        return self._find.find(text, k)

    # ------------------------------------------------------------------ build
    @classmethod
    def with_stubs(cls, store: NoteStore, embedding_dim: int = 256) -> "Notebase":
        """Build the app with the placeholder black boxes."""
        embedder = stubs.HashingEmbedder(dim=embedding_dim)
        return cls(
            WritePipeline(
                store=store,
                atomizer=stubs.ParagraphAtomizer(),
                contextualizer=stubs.IdentityContextualizer(),
                enricher=stubs.FrequencyEnricher(),
                embedder=embedder,
                linker=stubs.NearestNeighborLinker(),
            ),
            FindPipeline(
                store=store,
                interpreter=stubs.SimpleQueryInterpreter(embedder),
                retriever=stubs.RRFRetriever(),
                spreader=stubs.DecaySpreader(),
                clusterer=stubs.ComponentClusterer(),
                labeler=stubs.TopTagLabeler(),
            ),
        )

    @classmethod
    def from_config(cls, config: Config | None = None) -> "Notebase":
        from notebase.storage.mongo import MongoNoteStore

        config = config or Config()
        store = MongoNoteStore(config.mongo_uri, config.database, config.embedding_dim)
        store.ensure_indexes()
        return cls.with_stubs(store, config.embedding_dim)
