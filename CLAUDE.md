# Notebase: notes for agent sessions

A note-taking app with exactly two verbs, **write** and **find**. Notes are
atomic, have no folders, and are found by association (semantic search plus
spreading over a link graph). Read `README.md` for the idea and
`docs/architecture.md` for the structure.

## Where things are

- `notebase/write.py`, `notebase/find.py`: the pipelines. They only orchestrate.
- `notebase/blackbox/interfaces.py`: the 10 black-box Protocols. All algorithmic decisions live behind these.
  - `stubs.py`: naive placeholder for every box (also the fallback when a real box fails).
  - `embedders.py`: real Embedder (Qwen3-Embedding-0.6B, local). See `docs/embedding.md`.
  - `llm.py`: real Atomizer and Enricher (Claude, Haiku 5.5 by default). See `docs/llm.md`.
- `notebase/storage/`: `NoteStore` contract, MongoDB implementation, in-memory implementation for tests.
- `notebase/app.py`: composition root and `Config` (all `NOTEBASE_*` env vars).
- `docs/DEVLOG.md`: decisions and session history. **Add an entry when you finish a piece of work.**
- Backlog: GitHub issues on `saltfish-len/Notebase`.

## Setting up a fresh cloud container

```bash
dockerd > /tmp/dockerd.log 2>&1 &          # the Docker daemon is not running by default
docker compose up -d                       # MongoDB with Vector Search; wait until healthy:
until [ "$(docker inspect -f '{{.State.Health.Status}}' notebase-mongodb-1)" = healthy ]; do sleep 2; done
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

The first real-model run downloads Qwen3-Embedding-0.6B (about 1.2 GB) from Hugging Face.
The LLM boxes read `ANTHROPIC_API_KEY`, or `CLAUDE_API_KEY` as an alias (the user's cloud
environment sets `CLAUDE_API_KEY`). Without a key, writes still work through the stub fallback.
If the Docker daemon dies mid-session, MongoDB goes with it: restart both.

## Tests

```bash
python -m pytest   # `pytest` on PATH may be a separate tool env that cannot import notebase
```

```bash
NOTEBASE_TEST_MONGO_URI="mongodb://localhost:27017/?directConnection=true" NOTEBASE_TEST_MODEL=1 python -m pytest
```

The first command is fast (about 0.1 s), runs offline, and is what CI runs. The second adds the
MongoDB integration tests and the real embedding model tests.

## Rules of the codebase

- Keep the app to two verbs. Maintenance tools are fine, but new user-facing verbs are not.
- Replace one black box at a time, behind its Protocol. Keep its stub as the fallback.
- **A write must never fail because of a model.** Store the raw capture first, and fall back to the stub on any error.
- Similarity thresholds are on the **calibrated** scale (`calibrate(cos, embedder.similarity_floor)`), never raw cosine.
- Never mix vectors from different embedding models in one database (`EmbeddingMismatch` guards this).
- Claude API: current model IDs only (`claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`), structured outputs through `messages.parse`, depth set with `effort`, and no `temperature` or prefill.
- Back claims about model quality with measurements (see `docs/embedding.md` for the pattern). Mark anything not measured as such.
- Keep the docs current: `README.md` (config table, roadmap), `docs/architecture.md` (black-box table), `docs/DEVLOG.md`.
