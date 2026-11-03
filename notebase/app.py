"""Composition root: wires storage + black boxes into the two verbs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from notebase.blackbox import Atomizer, Embedder, Enricher, Normalizer, stubs
from notebase.find import FindPipeline
from notebase.models import Attachment, FindResult, WriteResult
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
    # "anthropic" (default): Normalizer (image alt text), Atomizer and Enricher on
    # Claude; "none": stubs only, nothing leaves the machine
    llm: str = field(default_factory=lambda: os.environ.get("NOTEBASE_LLM", "anthropic"))
    llm_model: str = field(
        default_factory=lambda: os.environ.get("NOTEBASE_LLM_MODEL", "claude-haiku-5-5")
    )
    # per-box overrides, e.g. atomize with a stronger model, enrich with Haiku
    atomizer_model: str | None = field(
        default_factory=lambda: os.environ.get("NOTEBASE_ATOMIZER_MODEL")
    )
    enricher_model: str | None = field(
        default_factory=lambda: os.environ.get("NOTEBASE_ENRICHER_MODEL")
    )
    normalizer_model: str | None = field(
        default_factory=lambda: os.environ.get("NOTEBASE_NORMALIZER_MODEL")
    )
    # where the web server keeps files uploaded with a write (the capture points here)
    files_dir: str = field(
        default_factory=lambda: os.environ.get("NOTEBASE_FILES_DIR", "~/.notebase/files")
    )


class Notebase:
    """The whole app: exactly two public operations."""

    def __init__(self, write_pipeline: WritePipeline, find_pipeline: FindPipeline) -> None:
        self._write = write_pipeline
        self._find = find_pipeline

    def write(
        self,
        text: str = "",
        source: str = "typed",
        attachments: list[Attachment] | None = None,
        fingerprint: str | None = None,
    ) -> WriteResult:
        """Put a thought in: typed text, files (images, audio, PDFs, ...), or both.

        ``fingerprint`` identifies the source (an imported file, a batch item); a write whose
        fingerprint was already written does nothing and returns ``duplicate=True``."""
        return self._write.write(text, source, attachments, fingerprint)

    def find(self, text: str, k: int = 5) -> FindResult:
        return self._find.find(text, k)

    # ------------------------------------------------------------------ build
    @classmethod
    def create(
        cls,
        store: NoteStore,
        embedder: Embedder,
        atomizer: Atomizer | None = None,
        enricher: Enricher | None = None,
        normalizer: Normalizer | None = None,
    ) -> "Notebase":
        """Wire the pipelines; any box not given falls back to its stub."""
        store.bind_embedding(embedder.name, embedder.dim)
        floor = embedder.similarity_floor
        return cls(
            WritePipeline(
                store=store,
                normalizer=normalizer or stubs.FilenameAltTextNormalizer(),
                atomizer=atomizer or stubs.ParagraphAtomizer(),
                contextualizer=stubs.IdentityContextualizer(),
                enricher=enricher or stubs.FrequencyEnricher(),
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

        atomizer = enricher = normalizer = None
        if config.llm == "anthropic":
            from notebase.blackbox.llm import (
                LLMAtomizer,
                LLMEnricher,
                VisionNormalizer,
                build_client,
            )

            client = build_client()
            normalizer = VisionNormalizer(client, model=config.normalizer_model or config.llm_model)
            atomizer = LLMAtomizer(client, model=config.atomizer_model or config.llm_model)
            enricher = LLMEnricher(
                client,
                model=config.enricher_model or config.llm_model,
                tag_vocabulary=store.top_tags,
            )
        elif config.llm != "none":
            raise ValueError(f"NOTEBASE_LLM must be 'anthropic' or 'none', not {config.llm!r}")
        return cls.create(
            store, embedder, atomizer=atomizer, enricher=enricher, normalizer=normalizer
        )
