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
| D13 | **MCP server as a second front end** (`notebase-mcp`, stdio), with `write_batch` and `import_files` as batch forms of *write* | Lets Claude Code or any MCP client write, find and bulk-import for the user without a new verb. `import_files` reads files on the server, so a folder import costs the assistant no tokens for file contents. A batch keeps going past a bad item but stops at once when MongoDB is down or the embedding model doesn't match, since every item would fail the same way | ✅ |
| D14 | **De-duplicate by source fingerprint**: a write may carry a fingerprint (hash of an imported file's bytes, or of a batch item's text and files); a write whose fingerprint is already stored does nothing. The fingerprint is stored only once the write has finished | Makes import and `write_batch` safe to re-run after an interruption or a client retry. Content hash rather than path, so a moved or renamed file is still a duplicate. Set at the end so a write that crashed halfway is retried, not skipped. A single `write` has no fingerprint: writing the same thought twice is allowed. An edited file counts as new; replacing its old notes waits for superseding | ✅ |
| D15 | **Import Markdown as Obsidian renders it**, and embedded files as attachments placed where they were embedded (`![[name]]` placeholder, replaced by the Normalizer) | The user's rule: what you see is what you meant. Link syntax, front matter and comments are noise to the Atomizer and the embedder. Appending image descriptions at the end put a whiteboard photo into the wrong note in the end-to-end run; in place, it stayed with its caption | ✅ |

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

### 2026-11-03: is it easy to deploy?

A clean-room run of the README in a fresh container (new venv, `docker compose up -d`, first write and find, then failure cases). Install took about 1.5 minutes plus downloads (venv 1.5 GB, MongoDB image 2 GB, model 1.2 GB). What it found and what changed:

1. **Bug: MongoDB never started again after its container was recreated.** `mongodb-atlas-local` names its replica set after the container's hostname and stores that in the data volume. Compose gave the container a random hostname, so any recreation (changing the compose file, updating the image, `down` + `up`) left the node `REMOVED`, stuck in `health: starting`, with all notes unreachable. Fixed with `hostname: mongodb` in `docker-compose.yml`; checked that the database comes back healthy with its notes after `up --force-recreate`, `restart` and `down` + `up`. Databases created before this change can't be recovered this way (the README says how to start over).
2. **MongoDB down → 30 s hang and a pymongo traceback.** Now a 5 s server-selection timeout and one line: "cannot reach MongoDB at … Start it … with: docker compose up -d".
3. **`restart: unless-stopped`**, so MongoDB comes back after a reboot once Docker Desktop is running.
4. **`NOTEBASE_DEVICE`** (`cpu` / `cuda` / `mps`) as an escape hatch for the embedding model, because MPS on Apple Silicon is untested.
5. **README quickstart** split by platform: on macOS PyPI's PyTorch is right; on Linux without a GPU the CPU index avoids several GB of CUDA. It now states Python 3.10+ (macOS's system Python is too old), the disk space, the API key and what happens without one, and the 7 s per command from reloading the model ([#2](https://github.com/saltfish-len/Notebase/issues/2)).

Seen but left alone: without an API key each write prints two fallback warnings (the README says to set `NOTEBASE_LLM=none` for quiet offline use), and Hugging Face prints an "unauthenticated requests" warning on every command unless `HF_TOKEN` or `HF_HUB_OFFLINE=1` is set.

Still not tested on macOS: Docker Desktop, the arm64 MongoDB image (the manifest lists arm64), MPS, and a real iPhone HEIC file.

### 2026-11-03: MCP server for Claude Code and other LLM clients

1. **`notebase/mcp_server.py`, `notebase-mcp`** (D13): four tools, `write`, `write_batch`, `import_files`, `find`. The logic is in `NotebaseTools`, plain methods testable without the `mcp` package; `build_server` registers them. `mcp` is an optional extra (`pip install -e ".[mcp]"`); the server works with the SDK's 2.x `MCPServer` and 1.x `FastMCP`.
2. **Lazy app**: the app (embedding model, MongoDB connection) is built on the first tool call, so the client sees the server start at once. After that the model stays loaded, unlike the CLI's ~7 s per command.
3. **`import_files`**: files and folders → one capture per text file (`.md`, `.txt`, `.org`, `.rst`, ...) and one per image, `source="import"`. Hidden paths are skipped (`.obsidian`, `.git`), text over 200 KB is skipped, an untitled file gets its file name as a heading so the Atomizer knows what a bare list is about. Sorted order, 50 per call with `offset` / `next_offset`, so one call stays inside client timeouts; `dry_run` lists the files.
4. **Errors reach the assistant**: the SDK hides the text of unexpected exceptions from the client, so tool errors are re-raised as `ToolError` with the reason. "Cannot reach MongoDB … docker compose up -d" stops a batch on the first item instead of timing out 5 s per item.

State at the end: 52 fast tests pass (10 new, including tool registration through the real SDK; the new ones also pass on mcp 1.30), and with MongoDB 55 pass. End to end over real stdio with the SDK's client, on mcp 2.3 and 1.30 (MongoDB, hashing embedder, `NOTEBASE_LLM=none`): list tools, `write_batch` of 2, `find` hit the right note as the seed; with MongoDB stopped, both tools returned the "cannot reach MongoDB" message. Not run: the real Qwen3 embedder or live Claude through the server (same pipeline code as the CLI), and a client other than the SDK's (Claude Code itself was not driven in this session). Left for later: de-duplication on re-import, Obsidian front matter and wikilinks.

### 2026-11-03: import de-duplication and Obsidian syntax

1. **Fingerprints** (D14): `Capture.fingerprint`, `NoteStore.find_capture(fingerprint)` (MongoDB: sparse index on `captures.fingerprint`), and `Notebase.write(..., fingerprint=)`, which returns `WriteResult(duplicate=True)` with the earlier capture and writes nothing. `import_files` uses `file:<sha256 of the bytes>`, `write_batch` `item:<sha256 of text + attachment hashes>` (`skip_duplicates=false` turns it off). Results report `duplicates` next to `written` and `failed`.
2. **`notebase/importing.py`** (D15): planning, file reading and the Markdown renderer moved out of the MCP server. Rendered as Obsidian shows it: wikilinks with alias, heading and block parts, escaped `\|` in tables, relative Markdown links, front matter, `%%` comments; code is left alone. Embedded files are resolved like Obsidian (next to the note, vault root, then by name anywhere in the vault), attached, and not imported again on their own.
3. **Normalizer placement**: `place_markers` now replaces `![[file name]]` with the file's marker, and appends only files without a placeholder (both the stub and `VisionNormalizer`). Found in the end-to-end run: with the image appended at the end, the Atomizer put the whiteboard description into the note about cutting a feature by Friday instead of the "sprint plan is on the whiteboard" note.

State at the end: 76 fast tests pass (24 new across importing and the MCP tools), and with MongoDB 80 pass (one new: capture lookup by fingerprint, attachments round-tripped). End to end over stdio (MongoDB, real Qwen3 embedder, live Haiku) on a small vault with front matter, aliased and heading links, a comment, a `.obsidian` folder and a daily note embedding a whiteboard image from `attachments/`: the dry run listed the 3 notes and not the embedded image; the import gave 6 notes with links rendered as text and the image described in Chinese inside the "冲刺计划拍在白板上了" note; a second import reported 3 duplicates and wrote nothing; "白板上的冲刺计划" found the whiteboard note as the seed. Not tested: a real, large Obsidian vault (speed and cost of a big import are not measured). Obsidian's rendering of rarer forms (`[[#^block]]`, headings with special characters) follows its documented behaviour and was not checked in the app.

### 2026-11-03: two memory papers read against the design

Notes and feedback in `docs/papers/`. No code changed. All numbers there are the papers'; none are measured here.

1. **[LightMem](papers/lightmem.md)** (ICLR 2026, arXiv 2510.18866): agent dialogue memory with token pre-compression, topic segmentation, batched summaries, and long-term updates done offline ("sleep-time"). It supports atomizing (topic segmentation is worth 5–6 points) and append-only writes: in its case study an online LLM update deleted related information it took for a conflict. Proposed: offline consolidation as a maintenance tool (update queue from Linker neighbours, typed `supersedes` / `contradicts` / `elaborates` / `duplicate` links, tag merging, no deletions), which would also unblock superseding when an edited file is re-imported. Not adopted: pre-compression and cross-write batching. Found while reading the code: a long imported file (up to 200 KB) goes to the Atomizer in one call capped at `max_tokens=16000`, so it may fall back to the paragraph stub. Unmeasured; segmenting by heading, then by embedding similarity, is the proposed fix.
2. **[EM²Mem](papers/em2mem.md)** (arXiv 2609.00551): event-centric multimodal memory for long video. It binds evidence at construction time, and structured fields beat flattened captions. Gaps it points at in Notebase: entities are extracted but never used (the Linker is similarity only), and the Spreader cannot follow `capture_id` or time. Proposed: `shared-entity`, then `same-capture` and `temporal` links, kept only if a find evaluation supports them.
3. **Common lesson**: in both papers the simplest baseline is close (NaiveRAG ties LightMem on GLM-4.6; EM²Mem's local-records-only variant is 2 points behind, and any single ablation falls below it). Before the find-side boxes in [#4](https://github.com/saltfish-len/Notebase/issues/4), build a find evaluation (evidence recall@k for seeds and for the region) with a Retriever-only baseline.

### 2026-11-19: EmbeddingGemma 2 read against the Embedder

Notes and feedback in [`docs/papers/embeddinggemma2.md`](papers/embeddinggemma2.md) (arXiv 2610.37364). No code changed. All numbers there are the paper's; none are measured here.

EmbeddingGemma 2 (Google, 744M, 768-d with Matryoshka down to 128) embeds text, images, audio and video in one space; text-only it is 271M. By the paper's own tables it is behind Qwen3-Embedding-0.6B on text (MTEB Multilingual 61.4 vs 64.3, retrieval 60.8 vs 64.6), with no Chinese breakdown, so there is no case for switching the text Embedder on benchmarks. Worth measuring with our pair test anyway: it is trained for calibrated scores (unrelated pairs pushed toward cosine 0), which would mean a low `similarity_floor`, and it is half the size. Where it could matter: native image vectors would make images findable without the API, and native audio beat ASR + text by 12 MRR points on MSEB, which bears on the planned voice-memo Normalizer. Both need the `Embedder` Protocol to accept media, a whole-database switch (one vector space per database), and file bytes at embed time (D11).

### 2026-11-02: tag-only cues and the whole graph (`#all`)

1. **Bug: a cue of tags only (`#active-matter`) found nothing.** With the tags stripped, the cue was empty: no keywords, and the embedding of a blank string scored below the vector threshold. Now `SimpleQueryInterpreter` skips the embedder for such a cue, and `RRFRetriever` seeds every note with those tags (up to `max_tagged=50`), so they come back with their neighbours.
2. **`#all`** (`models.WHOLE_GRAPH_TAG`, reserved): `Query.whole_graph`. `FindPipeline` skips the Retriever and Spreader and returns every note from `NoteStore.list_notes(tags, limit)`, newest first, with all links between them, clustered and labelled as usual, capped at `max_graph_notes=2000`. `#all #tag` gives a tag's part of the graph without the cap of 50. It is a cue for *find*, not a new verb. `list_notes` is new in the store contract (memory, MongoDB; MongoDB leaves the embedding out).

State at the end: 80 fast tests pass (4 new), and the MongoDB tests pass (1 new: `list_notes` by tag, newest first). On the real database (94 notes, 369 links, Qwen3 embedder, imported Obsidian vault), driven through the MCP server: `#all` returned all 94 notes and all 369 edges in 9 clusters (107 KB of JSON); `#all #h-jepa` exactly the 18 tagged notes; `#active-matter` its 7 tagged notes as seeds plus neighbours (50); `#nope` nothing. Not handled: there is no index on `tags`/`created_at` for `list_notes` (fine at this size, not measured at 10k notes), a truncated `#all` does not say so, and a large `#all` result over MCP can be too big for the client's context.

### 2026-11-02: the web UI on main

PR #10 (`notebase-web` and `notebase import`) was merged into `claude/nice-brahmagupta-4z8rkw` instead of `main`, so the web UI never reached main. Merging the whole branch conflicts across the stores, the write pipeline and the docs, because it and PR #9 both built an Obsidian importer with its own de-duplication (`Capture.origin` vs. `Capture.fingerprint`). Only the web UI was brought over: `server.py`, `jsonio.py`, `web/index.html`, `tests/test_server.py` unchanged, plus the `notebase-web` script, the `web/*` package data and `Config.files_dir`. `notebase import` (the CLI importer) stays on that branch; the MCP `import_files` covers vault imports on main.

State at the end: 88 fast tests pass (8 server tests). Run against the real database (94 notes): `#all` drew all 94 notes in 9 clusters in 64 ms.

### 2026-11-03: records caught up with the web UI

Docs only. The web UI covers [#3](https://github.com/saltfish-len/Notebase/issues/3) (closed) and most of [#2](https://github.com/saltfish-len/Notebase/issues/2): `notebase-web` is the long-running server with `POST /write` and `POST /find`. What is left of #2 is the CLI using a running server instead of loading the model itself, so #2 stays open for that and the README roadmap now lists it as its own item. `.claude/launch.json` (the `notebase-web` preview config) is committed; `.claude/settings.local.json` is ignored as per-machine.

uv support, alongside pip: `uv.lock` is committed and `uv sync --all-extras` installs everything. `torch` is now a direct dependency (it already came in through sentence-transformers) so that `[tool.uv.sources]` can take it from the CPU index on Linux, matching the pip instructions; macOS gets PyPI's build. pip ignores the `[tool.uv]` tables. Checked on macOS: `uv lock`, `uv sync --all-extras`, 88 fast tests pass. Not run: `uv sync` on Linux (the lock has `+cpu` torch wheels for Linux).
