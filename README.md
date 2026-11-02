# Notebase

> A note-taking app with two verbs: **Write** and **Find**.

Notebase treats notes as an extension of your memory. People don't keep their thoughts in folders, and they don't recall them by walking a path like `Work/2025/Projects/Q3/ideas.md`. One thought brings up another that is related to it. Notebase works the same way: you put thoughts in, you get them back by association, and there is no hierarchy to look after.

---

## Why

Most note apps ask you to organize at write time. You have to choose a folder, a file name, maybe a few tags. That costs effort when the thought is fresh, and the structure decays as your interests drift. Months later the note is in a folder you no longer think in.

Notebase moves all organization to the system:

| Traditional notes | Notebase |
|---|---|
| You decide where a note lives | Notes have no location |
| You name and tag notes | The system extracts keywords and metadata |
| One file can hold many ideas | Each note holds one atomic idea |
| You search by path or exact words | You search by meaning and association |
| Results come back as a flat list | Results come back as a cluster of related notes |

---

## The two verbs

### ✍️ Write

You type, paste, or dictate anything: a half-formed idea, a meeting dump, a quote, a link with a comment. You don't pick a title, folder, or tag.

The system then processes the input:

1. **Capture.** The raw input is stored unchanged, so nothing is ever lost.
2. **Atomize.** The input is split into *atomic notes*, each holding one self-contained idea. A long brain-dump might become five notes. A short thought stays as one.
3. **Contextualize.** Each atomic note is rewritten just enough to be understood on its own, for example by resolving "it" and "that meeting" from the surrounding text, while keeping your voice.
4. **Enrich.** Each note gets:
   - **Keywords**: terms you might use later to look for it, including synonyms you didn't write
   - **Tags**: broad topic labels, generated and normalized against existing tags
   - **Entities**: people, places, projects, books, URLs
   - **Metadata**: creation time, source (typed, pasted, voice, web clip), and a link back to the raw capture
5. **Embed.** Each note is turned into a vector for semantic search.
6. **Link.** The new note is compared with existing notes, and weighted associative edges are created to its nearest neighbours. These edges are what make up the graph.
7. **Store.** Notes, vectors, and edges are saved to a store that supports semantic search.

```
raw input ──► capture ──► atomize ──► contextualize ──► enrich ──► embed ──► link ──► store
                │                                                                       │
                └────────────────────── raw text kept for provenance ───────────────────┘
```

### 🔎 Find

You type a cue: a word, a question, a vague feeling ("that thing about sleep and memory someone told me"). The system returns the *region of your memory* the cue lands in, rather than a ranked list of documents.

1. **Interpret.** The query is expanded into keywords and embedded.
2. **Retrieve.** Hybrid search combines semantic similarity, keyword/BM25 matches, and metadata filters (time, source, entity) to find **seed notes**.
3. **Spread.** Starting from the seeds, the search walks the associative graph a few hops out, the way one memory brings up the next. Edge weights and decay limit how far it goes.
4. **Cluster.** The collected notes are grouped into coherent sub-clusters, and each cluster gets a short generated label.
5. **Present.** The result is a zoomed-in area of the knowledge graph, similar to focusing on one neighbourhood of Obsidian's graph view:
   - seed notes are highlighted at the centre
   - related notes surround them, placed according to their associations
   - clusters are visually grouped and labelled
   - clicking a note re-centres the view on it, so you can keep following associations

```
        query
          │
    ┌─────▼─────┐      ┌──────────────┐      ┌───────────┐      ┌──────────────┐
    │ interpret │ ───► │ hybrid search│ ───► │  spread   │ ───► │ cluster +    │ ───► graph view
    └───────────┘      │   (seeds)    │      │ over graph│      │ label        │
                       └──────────────┘      └───────────┘      └──────────────┘
```

---

## The atomic note

The atomic note is the only unit in the system.

```jsonc
{
  "id": "n_01HZX...",
  "content": "Sleep consolidates memories by replaying the day's experiences in the hippocampus.",
  "keywords": ["sleep", "memory consolidation", "hippocampus", "replay", "learning"],
  "tags": ["neuroscience", "memory"],
  "entities": [{ "type": "concept", "name": "hippocampus" }],
  "created_at": "2026-11-02T21:40:00Z",
  "source": { "kind": "typed", "capture_id": "c_01HZX..." },
  "embedding": [/* float vector */],
  "links": [
    { "to": "n_01HZW...", "weight": 0.82, "reason": "semantic" },
    { "to": "n_01HZV...", "weight": 0.64, "reason": "shared-entity" }
  ]
}
```

Principles:

- **One idea per note.** Small notes recombine well.
- **No location.** Notes have no folder, path, or parent.
- **Mostly append-only.** To change your mind, you write a new thought. The system links it to the old one and can mark the old one as superseded, the way memory layers new understanding on old.
- **Provenance is kept.** Every atomic note links back to the raw input it came from.

---

## Non-goals

These are left out on purpose:

- Folders, notebooks, or any hierarchy
- Manual tagging or manual linking (both may be allowed as hints, but they are never required)
- Rich-text documents, long-form writing, or page layout
- Task management, calendars, or databases

If a feature isn't *write* or *find*, it probably doesn't belong.

---

## Architecture

Python + **MongoDB** (notes, vectors via Atlas Vector Search `$vectorSearch`, and graph edges in one database).
Every core algorithm is a **black box**: an interface that currently has a naive placeholder stub, so the
whole system runs end to end while the real algorithms are built one box at a time.

```
                write                                              find
                  │                                                  │
   ┌──────────────▼──────────────┐                   ┌───────────────▼──────────────┐
   │ WritePipeline               │                   │ FindPipeline                 │
   │  [Atomizer]                 │                   │  [QueryInterpreter]          │
   │  [Contextualizer]           │                   │  [Retriever]   ◄── $vectorSearch + $text
   │  [Enricher]                 │                   │  [Spreader]    ◄── notes.links
   │  [Embedder]                 │                   │  [Clusterer]                 │
   │  [Linker]  ◄── $vectorSearch│                   │  [Labeler]                   │
   └──────────────┬──────────────┘                   └───────────────┬──────────────┘
                  │                                                  │
                  ▼                                                  ▼
        ┌──────────────────── NoteStore ────────────────────┐   FindResult
        │ MongoDB: captures · notes{embedding, links}       │   (seeds, notes, edges,
        │ (InMemoryNoteStore for tests)                     │    clusters → graph view)
        └───────────────────────────────────────────────────┘
```

See **[docs/architecture.md](docs/architecture.md)** for the diagrams, each black box's contract and
stub, and the MongoDB data model.

```
notebase/
├── app.py              # composition root: Notebase.write / Notebase.find
├── write.py            # WritePipeline (orchestration only)
├── find.py             # FindPipeline  (orchestration only)
├── models.py           # Capture, Note, Link, Query, FindResult, ...
├── blackbox/
│   ├── interfaces.py   # the 10 black-box Protocols
│   ├── embedders.py    # real Embedder (Qwen3 via sentence-transformers)
│   ├── llm.py          # real Atomizer + Enricher (Claude)
│   └── stubs.py        # placeholder implementations
├── storage/
│   ├── base.py         # NoteStore contract
│   ├── mongo.py        # MongoDB + Vector Search
│   └── memory.py       # in-memory, for tests
└── cli.py              # `notebase write` / `notebase find`
```

## Quickstart

```bash
docker compose up -d          # MongoDB with Vector Search (mongodb-atlas-local)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch (skip if you have a GPU)
pip install -e ".[dev]"
```

```bash
notebase write "Deep sleep consolidates memory in the hippocampus."
notebase write "Coffee in the afternoon hurts deep sleep."
notebase write "Spaced repetition helps long-term retention."
notebase find "怎么记得更牢"
```

```
◆ ...
  ★ [1.00] Spaced repetition helps long-term retention.
  ★ [0.97] Deep sleep consolidates memory in the hippocampus.
  · [0.21] Coffee in the afternoon hurts deep sleep.
```

★ marks a seed note the query hit directly. · marks a note reached by association. `--json` gives the full
`FindResult` (nodes, edges, clusters) for a graph UI.

The first run downloads the embedding model, [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
(about 1.2 GB). It runs locally on CPU, CUDA or Apple Silicon, so notes never leave your machine. It works across
Chinese and English, so a cue in one language finds notes written in the other. See
[docs/embedding.md](docs/embedding.md) for why this model was chosen.

### Configuration

| Variable | Default | |
|---|---|---|
| `NOTEBASE_MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | |
| `NOTEBASE_DB` | `notebase` | one database = one embedding model |
| `NOTEBASE_EMBEDDER` | `qwen3-0.6b` | preset (`qwen3-0.6b`, `bge-m3`, `minilm-multilingual`), `hashing` (offline stub), or any sentence-transformers model id |
| `NOTEBASE_EMBED_DIM` | model native | truncate Matryoshka models, e.g. `256` for Qwen3 |
| `NOTEBASE_SIMILARITY_FLOOR` | `0.3` | only for custom model ids: the cosine an unrelated pair typically gets |
| `NOTEBASE_LLM` | `anthropic` | `anthropic`: LLM Atomizer + Enricher on Claude; `none`: stubs only, fully offline |
| `NOTEBASE_LLM_MODEL` | `claude-haiku-5-5` | model for both LLM boxes |
| `NOTEBASE_ATOMIZER_MODEL` / `NOTEBASE_ENRICHER_MODEL` | (unset) | per-box override, e.g. atomize with Sonnet, enrich with Haiku |
| `ANTHROPIC_API_KEY` (or `CLAUDE_API_KEY`) | (unset) | Claude API key from the [Claude Console](https://platform.claude.com/) |

Without a working API key, writes still succeed: the LLM boxes log a warning and fall back to their stubs.

A database remembers which model wrote its vectors and refuses to run with a different one, because vectors from
different models can't be compared.

### Tests

```bash
pytest
```

```bash
NOTEBASE_TEST_MONGO_URI="mongodb://localhost:27017/?directConnection=true" NOTEBASE_TEST_MODEL=1 pytest
```

The first command is fast: it uses the in-memory store and the hashing stub. The second also runs the MongoDB
integration tests and the real-model tests.

---

## Roadmap

Open work is tracked in [GitHub issues](https://github.com/saltfish-len/Notebase/issues). Decisions and history are in [docs/DEVLOG.md](docs/DEVLOG.md).

- [x] **v0: Skeleton**: full write/find pipelines on MongoDB, with every algorithm as a stubbed black box
- [ ] **v0.x: Fill the black boxes**
  - [x] Real embedding model (Qwen3-Embedding-0.6B, local)
  - [x] LLM Atomizer + Enricher (Claude Haiku 5.5; see [docs/llm.md](docs/llm.md))
  - [ ] Measure Haiku vs Sonnet for the LLM boxes ([#1](https://github.com/saltfish-len/Notebase/issues/1))
  - [ ] Find-side boxes: Labeler, Clusterer, Spreader, QueryInterpreter ([#4](https://github.com/saltfish-len/Notebase/issues/4))
  - [ ] Chinese keyword search, vector index lag ([#6](https://github.com/saltfish-len/Notebase/issues/6))
  - [ ] Re-embed when switching embedding models ([#5](https://github.com/saltfish-len/Notebase/issues/5))
- [ ] **v1: Graph view**
  - [ ] Long-running server with a write/find HTTP API ([#2](https://github.com/saltfish-len/Notebase/issues/2))
  - [ ] Web UI: one input for write and find, zoomed-in subgraph, click to re-centre ([#3](https://github.com/saltfish-len/Notebase/issues/3))
- [ ] **v2: Better memory**
  - [ ] Superseding and versioning of thoughts
  - [ ] Recency and frequency weighting (often-recalled notes get stronger)
  - [ ] Import from Markdown / Obsidian vaults (flattened into atomic notes)
  - [ ] Voice and web-clip capture

---

## Open questions

- How aggressively should input be atomized, and should the user be able to review the split?
- Should tags come from a fixed vocabulary or emerge freely and get merged over time?
- How many hops should spreading activation take, and how should the graph be pruned?
- Should *find* also be able to synthesize an answer from the cluster, or only show the notes?

---

## License

TBD
