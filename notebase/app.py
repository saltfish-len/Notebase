"""Composition root: wires storage + black boxes into the two verbs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from notebase.blackbox import Embedder, stubs
from notebase.find import FindPipeline
from notebase.models import FindResult, WriteResult
from notebase.storage.base import NoteStore
from notebase.write import WritePipeline


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


@dataclass
class Config:
    mongo_uri: str = field(
        default_factory=lambda: os.environ.get(
            "NOTEBASE_MONGO_URI", "mongodb://localhost:27017/?directConnection=true"
        )
    )
    database: str = field(default_factory=lambda: os.environ.get("NOTEBASE_DB", "notebase"))
    # preset name (qwen3-0.6b | bge-m3 | minilm-multilingual), "hashing", or a HF model id
    embedder: str | None = field(default_factory=lambda: os.environ.get("NOTEBASE_EMBEDDER"))
    # optional: truncate Matryoshka models (e.g. Qwen3 1024 -> 256) / size of hashing stub
    embedding_dim: int | None = field(default_factory=lambda: _env_int("NOTEBASE_EMBED_DIM"))


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
    def create(cls, store: NoteStore, embedder: Embedder) -> "Notebase":
        """Real embedder + placeholder black boxes for everything else."""
        store.bind_embedding(embedder.name, embedder.dim)
        floor = embedder.similarity_floor
        return cls(
            WritePipeline(
                store=store,
                atomizer=stubs.ParagraphAtomizer(),
                contextualizer=stubs.IdentityContextualizer(),
                enricher=stubs.FrequencyEnricher(),
                embedder=embedder,
                linker=stubs.NearestNeighborLinker(similarity_floor=floor),
            ),
            FindPipeline(
                store=store,
                interpreter=stubs.SimpleQueryInterpreter(embedder),
                retriever=stubs.RRFRetriever(similarity_floor=floor),
                spreader=stubs.DecaySpreader(),
                clusterer=stubs.ComponentClusterer(),
                labeler=stubs.TopTagLabeler(),
            ),
        )

    @classmethod
    def with_stubs(cls, store: NoteStore, embedding_dim: int = 256) -> "Notebase":
        """Everything stubbed, including the embedder. Fast; for tests."""
        return cls.create(store, stubs.HashingEmbedder(dim=embedding_dim))

    @classmethod
    def from_config(cls, config: Config | None = None) -> "Notebase":
        from notebase.blackbox.embedders import build_embedder
        from notebase.storage.mongo import MongoNoteStore

        config = config or Config()
        embedder = build_embedder(config.embedder, config.embedding_dim)
        store = MongoNoteStore(config.mongo_uri, config.database, embedder.dim)
        store.bind_embedding(embedder.name, embedder.dim)
        store.ensure_indexes()
        return cls.create(store, embedder)
