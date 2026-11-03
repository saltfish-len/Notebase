"""Real embedding models for the Embedder black box.

The default is Qwen3-Embedding-0.6B, run locally through sentence-transformers.
It was picked by measuring how well each candidate separates associated notes
from unrelated ones in English, Chinese and across the two languages. See
docs/embedding.md for the numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

NOTES_INSTRUCTION = (
    "Instruct: Given a memory cue, retrieve personal notes that are related to "
    "or associated with it\nQuery:"
)


@dataclass(frozen=True)
class ModelPreset:
    model: str
    similarity_floor: float
    query_prompt: str = ""
    document_prompt: str = ""
    kwargs: dict = field(default_factory=dict)


PRESETS: dict[str, ModelPreset] = {
    # 1024-d, ~1.2 GB, zh/en/cross-lingual; best separation in our measurements
    "qwen3-0.6b": ModelPreset(
        "Qwen/Qwen3-Embedding-0.6B", similarity_floor=0.30, query_prompt=NOTES_INSTRUCTION
    ),
    # 1024-d, ~2.2 GB, strong multilingual, slower
    "bge-m3": ModelPreset("BAAI/bge-m3", similarity_floor=0.35),
    # 384-d, ~0.5 GB, tiny and fast, but weak at loose associations
    "minilm-multilingual": ModelPreset(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", similarity_floor=0.15
    ),
}

DEFAULT_PRESET = "qwen3-0.6b"


class SentenceTransformerEmbedder:
    """Any sentence-transformers model, running locally (CPU, CUDA or MPS)."""

    def __init__(
        self,
        model: str,
        similarity_floor: float,
        query_prompt: str = "",
        document_prompt: str = "",
        truncate_dim: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
        **model_kwargs,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError(
                "The local embedder needs sentence-transformers: pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(
            model, device=device, truncate_dim=truncate_dim, **model_kwargs
        )
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension  # sentence-transformers < 6
        )
        self.dim = get_dim()
        self.similarity_floor = similarity_floor
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.batch_size = batch_size
        self.name = f"st:{model}:{self.dim}"

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "SentenceTransformerEmbedder":
        p = PRESETS[preset]
        return cls(
            p.model,
            similarity_floor=p.similarity_floor,
            query_prompt=p.query_prompt,
            document_prompt=p.document_prompt,
            **{**p.kwargs, **overrides},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self.document_prompt) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.query_prompt)[0]

    def _encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        import numpy as np

        vecs = self._model.encode(
            texts, prompt=prompt or None, batch_size=self.batch_size, convert_to_numpy=True
        ).astype(np.float64)
        # Normalize in float64: low-precision (bf16) weights leave the model's own
        # normalization ~0.2% off, which skews cosine scores.
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs.tolist()


def build_embedder(spec: str | None = None, dim: int | None = None):
    """Build an embedder from a preset name, ``hashing``, or any HF model id.

    ``spec`` defaults to $NOTEBASE_EMBEDDER, then to the Qwen3 preset.
    ``dim`` truncates Matryoshka-capable models (e.g. Qwen3 -> 256), or sets
    the size of the hashing stub.
    """
    spec = spec or os.environ.get("NOTEBASE_EMBEDDER") or DEFAULT_PRESET
    if spec == "hashing":
        from notebase.blackbox.stubs import HashingEmbedder

        return HashingEmbedder(dim=dim or 256)
    if spec in PRESETS:
        return SentenceTransformerEmbedder.from_preset(spec, truncate_dim=dim)
    floor = float(os.environ.get("NOTEBASE_SIMILARITY_FLOOR", "0.3"))
    return SentenceTransformerEmbedder(spec, similarity_floor=floor, truncate_dim=dim)
