# Embedder

The Embedder is the first black box to get a real implementation:
**[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)**,
run locally through `sentence-transformers` (`notebase/blackbox/embedders.py`).

## What Notebase needs from an embedding model

Notebase is not a search engine over documents. It links *thoughts* by
*association* and expects to work across languages. So a model has to:

1. score **associated** notes clearly above **unrelated** ones, even when they
   share no words ("I keep forgetting names" ↔ "spaced repetition"),
2. work in **Chinese and English, and across them**,
3. leave a **wide gap** between related and unrelated scores, because the Linker,
   Spreader and Clusterer all decide with thresholds,
4. run **locally**, since notes are private.

## Measurements

Cosine similarity of note pairs (normalized embeddings, no prompts):

| model | dim | related en | related zh | en↔zh | **association** | unrelated en | unrelated zh | unrelated en↔zh |
|---|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.48 | 0.73 | 0.80 | **0.11** ✗ | 0.10 | 0.17 | 0.00 |
| multilingual-e5-small | 384 | 0.87 | 0.91 | 0.81 | 0.80 | **0.76** ✗ | **0.86** ✗ | 0.73 |
| bge-m3 | 1024 | 0.61 | 0.59 | 0.72 | 0.53 | 0.32 | 0.35 | 0.42 |
| **Qwen3-Embedding-0.6B** | 1024 | 0.56 | 0.60 | 0.80 | **0.59** | 0.28 | 0.15 | 0.33 |

* **MiniLM** scores association the same as noise (0.11 vs 0.10), which fails
  requirement 1.
* **e5** packs everything into 0.73–0.91, and an unrelated Chinese pair (0.86)
  outscores a related English pair (0.87). No threshold can work.
* **bge-m3** is usable, but the gap is smaller, it is 2.2 GB, and it is the slowest.
* **Qwen3** has the clearest separation and is about 1.2 GB. It is the default.

All three alternatives are still available as presets through `NOTEBASE_EMBEDDER`.

EmbeddingGemma 2 (2026-11, text + image + audio in one space) has not been measured here yet; on
MTEB it is behind Qwen3 on text. See [papers/embeddinggemma2.md](papers/embeddinggemma2.md).

### Query instruction

Qwen3 is instruction-aware on the query side. Short *find* cues were tested
with three settings: no prompt, the model's default web-search instruction,
and a notes-specific instruction:

```
Instruct: Given a memory cue, retrieve personal notes that are related to or associated with it
Query:
```

Without an instruction, "meeting schedule" ranked "I keep forgetting names"
above the budget-meeting note, and "that thing about cooking" barely separated
pasta from noise. The notes-specific instruction ranks both correctly, and
"cooking" also pulls in the Chinese pasta note. Notes are embedded with no
prompt, and queries get this instruction.

## Calibration: `similarity_floor`

Every model has its own noise level, meaning the cosine an unrelated pair
typically gets: about 0.3 for Qwen3, about 0.75 for e5, and 0 for the hashing
stub. A threshold like "link if similarity ≥ 0.2" would mean something
different for each model.

So each embedder declares a `similarity_floor`, and downstream boxes rescale:

```
strength = clip((cosine − floor) / (1 − floor), 0, 1)
```

Link weights, the Retriever's cut-off, the Spreader's activation and the
Clusterer's threshold all work on this calibrated scale, so swapping the model
does not require retuning them. With Qwen3 (floor 0.30) and the Linker's
`min_weight = 0.2`, notes link when cosine is at least 0.44.

## One database, one vector space

Vectors from different models (or different truncation sizes) can't be
compared. The store records the embedder's name and dimension in
`meta.embedding` and raises `EmbeddingMismatch` if you start Notebase against
it with a different model. To switch models, point `NOTEBASE_DB` at a new
database. (A re-embed command can be added when needed: the raw text is kept in
`captures` and `notes`.)

## Practical notes

* The first run downloads about 1.2 GB from Hugging Face. Set `HF_TOKEN` to
  avoid anonymous rate limits.
* On CPU, the model loads in a few seconds and embeds a note in tens of
  milliseconds. The CLI loads the model on every call. A long-running server
  would load it once.
* The weights load in bfloat16, which was faster than float32 on the test CPU.
  Vectors are re-normalized in float64 afterwards, because bf16 leaves them
  about 0.2% off unit length.
* `NOTEBASE_EMBED_DIM=256` (or any of 32–1024) truncates Qwen3's Matryoshka
  embeddings to save storage, at some loss of quality.
* The hashing stub is still available (`NOTEBASE_EMBEDDER=hashing`) and is used
  by the fast unit tests.
