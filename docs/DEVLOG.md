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
| D7 | **Atomizer and Enricher on Claude Haiku 5.5** (Atomizer effort `medium`, Enricher `low`); Enricher batched per capture; existing tag vocabulary passed in; zh↔en keywords | Measured against Sonnet 5.5: after four prompt fixes Haiku passes 10/10, 9/10 and 9/10 eval cases with 0 fallbacks, at $0.36 vs $7.14 per 1000 writes and 1.5 s vs 2.2 s p50 per call. Raising the Atomizer to `high` did not help. See [docs/llm.md](llm.md) | ✅ measured: [#1](https://github.com/saltfish-len/Notebase/issues/1) |
| D8 | **Writes never fail because of a model**: the raw capture is saved first, and any LLM error, refusal or bad output falls back to the stub | A memory app must not lose what the user wrote | ✅ |
| D9 | **Contextualizer stays the identity stub** | The LLM Atomizer already makes notes self-contained, so a separate call would cost more for little gain | ✅ [#1](https://github.com/saltfish-len/Notebase/issues/1): pronouns and "it says" attributions were resolved on Haiku once the prompt asked for attribution explicitly |
| D10 | **Files become text in a Normalizer box before the Atomizer**, not a multimodal embedding | Keeps one text embedding space (D6) and every later box unchanged; a real Normalizer (Claude vision, local speech-to-text, PDF and web extraction) can replace the stub behind the same Protocol | ✅ v0 stub |
| D11 | **Captures store file references (path, size, SHA-256), not bytes**, for now | Enough for the file-name stub, which never reads the file. A real Normalizer that must re-read files later will need the bytes kept (e.g. GridFS) | ⚠️ revisit with the real Normalizer |
| D12 | **Image alt text on Claude Haiku 5.5** (effort `low`, one call per image, typed text as context); other files keep the file-name marker | Measured on 7 images (text-heavy synthetic + real photos, one HEIC): Haiku 20/21 and 19/21 over 2 × 3 runs at $0.25 per 1000 images, 1.5 s p50; Sonnet 7/7 at $5.11. See `docs/llm.md` | ✅ measured |

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

### 2026-11-02: is Haiku enough? ([#1](https://github.com/saltfish-len/Notebase/issues/1))

1. **Eval against the live API**, Haiku 5.5 vs Sonnet 5.5 (`scripts/eval_llm.py`). With the original prompt, Haiku passed 1/8 cases and Sonnet 8/8. The Atomizer outputs were nearly identical. The gap was the Enricher: Haiku never added Chinese keywords to English notes, and it stretched existing tags (flight booking tagged `memory`).
2. **The checks were partly wrong.** `refs_resolved` failed Sonnet's correct output and passed Haiku's dropped attribution. `url` expected 1 note where splitting off the task is correct. `fallbacks` missed validation fallbacks. All three were fixed in the eval, not by reinterpreting results, and `tags_vary` and `no_catchall_tags` were added.
3. **Four prompt fixes**: keep attributions, keep each part's language in mixed captures, keep cross-lingual keywords, and fit tags without catch-alls. With them, Haiku passes 10/10, 9/10 and 9/10 over 3 runs, Sonnet 10/10, and there are 0 fallbacks anywhere. Haiku costs $0.36 per 1000 writes vs Sonnet's $7.14, with p50 1.5 s vs 2.2 s per call.
4. **End to end** with MongoDB and the real embedder: mixed zh/en brain dumps produced 0 fallbacks, and cross-lingual *find* worked in both directions. The first end-to-end run found a failure the eval had no case for: Haiku translated an English sentence in a mixed capture into Chinese, 3 out of 3 times. A new eval case plus a prompt line brought that down to 1 in 9 runs. Atomizer effort `high` did not help (1 in 3, +40% cost, +0.6 s), so the default stays at `medium`.
5. **Decision**: Haiku for both boxes, defaults unchanged. Residual weaknesses are documented in `docs/llm.md`: occasional translation, tag drift between neighbouring topics, and inferred entities. `NOTEBASE_ATOMIZER_MODEL=claude-sonnet-5-5` is the documented escape hatch.

Setup note: in this container, `pytest` on `PATH` belongs to a separate uv tool environment and cannot import `notebase`. Use `python -m pytest`.


### 2026-11-03: Normalizer v0, files as input to *write*

1. **New black box: Normalizer** (`Capture -> str`, before the Atomizer). Multimodal input goes through *write*; there is no new verb. Every box after it still works on text (D10).
2. **v0 stub, `FilenameAltTextNormalizer`**: each attachment becomes `[kind: words from the file name (file name)]`, placed on its own line right after the typed text, so the paragraph stub keeps a caption and its files in one note. With no typed text, each file becomes its own note. The file is never opened.
3. **Data model**: `Attachment` (name, MIME type, path, size, SHA-256) on `Capture.attachments`. `Capture.text` stays what the user typed, `Capture.normalized` holds the Normalizer's output, and the boxes read `Capture.content`. The capture is saved before normalizing and again after. Only file **references** are stored, not bytes (D11).
4. **CLI**: `notebase write [text] -a FILE [-a FILE ...]`. A missing file is a usage error before anything is written.
5. **Atomizer prompt**: one rule for file markers. The first wording split a caption that already described its image into caption + bare marker, 3 of 3 runs. The rewrite ("a file is part of the idea it was attached to... a marker is a note of its own only when the capture has no text") passed a 5-case marker check 15/15, and the eval, now with 4 file cases and a `markers_once` check, passed 14/14, 14/14 and 13/14 on Haiku with 0 fallbacks (the one failure: too many keywords on `brain-dump-zh`, a known Enricher quirk). See `docs/llm.md`.

State at the end: 29 fast tests pass; the MongoDB tests (3, including a capture round trip with attachments) pass. End to end through the CLI (MongoDB, live Haiku, hashing embedder): a captioned image and a file-only audio write each gave one note, and `find` hit them by words from the file name, in English and Chinese. Not run: the real-embedder tests (unchanged code) and anything on macOS.

### 2026-11-03: images get real alt text (Normalizer, Claude vision)

1. **`VisionNormalizer`** (`notebase/blackbox/llm.py`): each attached image goes to Claude Haiku 5.5 with the typed text as context and comes back as 1–3 sentences describing the content, quoting important text in the image. It becomes `[image: <description> (file name)]`. Non-image files, and any image that fails (unreadable, unsupported, API error, refusal, empty answer), get the stub's file-name marker, one image at a time, so a write never fails because of this box. `NOTEBASE_NORMALIZER_MODEL` overrides the model; `NOTEBASE_LLM=none` keeps images local.
2. **Image preparation** (`notebase/blackbox/images.py`): API formats under 1568 px and 5 MB go as they are; HEIC (iPhone), TIFF, BMP and larger photos are rotated from EXIF, shrunk and re-encoded. `Pillow` and `pillow-heif` are now dependencies.
3. **Eval** (`scripts/eval_vision.py`, D12). Reading the outputs found two problems the first checks missed: the alt text repeated the typed text, and after the first fix a Chinese capture got an English description. Both are fixed in the prompt; see `docs/llm.md` for the steps and the one accepted quirk.
4. **End to end** (MongoDB, real Qwen3 embedder, live Haiku, CLI): 6 images (one HEIC) and 2 text notes. Six of seven cues found the right image as the top seed, including cross-lingual ones: "那张猫的照片" → the cat photo described in English, "咖啡花了多少钱" → the receipt, "端口被占用怎么办" → the error screenshot, "日本料理" → the sushi photo, "法国旅行" → the Eiffel Tower, "中文分词什么时候做" → the whiteboard. "铁塔" found nothing: the two-character cue scored under the retriever's threshold, a find-side limit ([#4](https://github.com/saltfish-len/Notebase/issues/4)), not an alt-text one.

State at the end: 41 fast tests pass (12 new for images and the VisionNormalizer, against a fake client), plus the MongoDB tests. Not run: anything on macOS (the HEIC path was tested with an image converted by `pillow-heif`, not a real iPhone file).
