# Dev log

Decisions and the work behind them, newest at the bottom. Open work is tracked
in GitHub issues.

## Decisions

| # | Decision | Why | Status |
|---|---|---|---|
| D1 | **Python** | ML and embedding ecosystem; the user didn't name a language | ✅ |
| D2 | **MongoDB as the only store**: notes, vectors (Atlas Vector Search `$vectorSearch`) and graph edges (`links` embedded on each note, mirrored) | User's choice of vector DB. One database keeps documents, vectors and the graph consistent. Edges can move to their own collection with `$graphLookup` later without changing `NoteStore`. | ✅ |
| D3 | **Every algorithm is a black box** (a Protocol) **with a naive stub**; pipelines only orchestrate | The whole system runs end to end from day one, and real algorithms replace stubs one at a time | ✅ |
| D4 | **Embedder: Qwen3-Embedding-0.6B, local**, with a notes-specific query instruction | Measured against MiniLM, e5-small and bge-m3: best separation of associated from unrelated notes in zh, en and zh↔en. MiniLM can't detect associations, and e5 compresses all scores into 0.73–0.91. See `docs/embedding.md`. | ✅ measured |
| D5 | **Calibrated similarity**: each embedder declares `similarity_floor`, and downstream thresholds use `(cos − floor)/(1 − floor)` | Each model has a different noise level, so raw-cosine thresholds break whenever the model changes | ✅ |
| D6 | **One database = one embedding space** (`meta.embedding`, `EmbeddingMismatch`) | Vectors from different models can't be compared, and mixing them would silently return garbage | ✅ |
| D7 | **Atomizer and Enricher on Claude Haiku 5.5**; Enricher batched per capture; existing tag vocabulary passed in; zh↔en keywords | Bounded extraction tasks, structured outputs, about 20× cheaper than Sonnet 5.5 | ⏳ reasoned, **not measured**: [#1](https://github.com/saltfish-len/Notebase/issues/1) |
| D8 | **Writes never fail because of a model**: the raw capture is saved first, and any LLM error, refusal or bad output falls back to the stub | A memory app must not lose what the user wrote | ✅ |
| D9 | **Contextualizer stays the identity stub** | The LLM Atomizer already makes notes self-contained, so a separate call would cost more for little gain | ✅ (revisit if [#1](https://github.com/saltfish-len/Notebase/issues/1) shows unresolved references) |

## Sessions

### 2026-11-02: from idea to working skeleton

1. **README**: the write/find concept, atomic notes, non-goals.
2. **Skeleton**: write and find pipelines; 10 black boxes with stubs; `NoteStore` (MongoDB + in-memory); CLI; docker-compose with `mongodb-atlas-local`.
   - Bug found in the demo: Atlas reports cosine as `(1 + cos)/2`, so unrelated notes scored about 0.5 and linked to everything. The store now converts back to raw cosine.
   - Bug found in the demo: index lag meant notes from the same capture never linked. They are now passed to the Linker as `siblings`.
3. **Real Embedder** (D4–D6). bf16 weights were faster than fp32 on CPU, but vectors came out about 0.2% off unit length, so they are re-normalized in float64.
4. **LLM Atomizer and Enricher** (D7–D9). Verified through the real SDK with a mocked transport (request shape, structured-output schema, response that begins with a `thinking` block). Not run against the live API: no key was available in this session. `scripts/eval_llm.py` is ready for [#1](https://github.com/saltfish-len/Notebase/issues/1).
5. **Project records**: this log, `CLAUDE.md`, CI for the fast tests, backlog issues [#1](https://github.com/saltfish-len/Notebase/issues/1)–[#6](https://github.com/saltfish-len/Notebase/issues/6).

State at the end: 27 tests pass, including the MongoDB integration tests and the real-model tests. A CLI write with no API key succeeds through the fallback.
