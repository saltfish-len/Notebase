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

## Architecture (proposed)

```
┌──────────────┐        ┌───────────────────────────────┐        ┌──────────────────────┐
│   Client     │        │           Core                │        │       Storage        │
│              │ write  │  Write pipeline               │        │                      │
│  input box ──┼───────►│  (LLM: atomize / enrich)      ├───────►│  notes  (SQL)        │
│              │        │  embedding model              │        │  vectors (ANN index) │
│  graph view ◄┼────────┤  Find pipeline                │◄───────┤  edges  (graph)      │
│              │  find  │  (hybrid search / spread /    │        │  raw captures        │
└──────────────┘        │   cluster / label)            │        └──────────────────────┘
                        └───────────────────────────────┘
```

- **LLM layer**: atomizing, contextualizing, keyword and tag extraction, query expansion, cluster labelling. The provider is pluggable, and local models are supported.
- **Embedding layer**: a pluggable embedding model.
- **Storage**: local-first by default. A single SQLite file with a vector extension can hold notes, vectors, and edges, so your memory stays on your machine.
- **Client**: a minimal UI with one input box and one graph canvas.

The tech stack is not final. See the roadmap.

---

## Roadmap

- [ ] **v0: Core loop (CLI)**
  - [ ] `notebase write "..."` runs the full write pipeline
  - [ ] `notebase find "..."` returns seed notes and their clustered neighbourhood as text
  - [ ] Local SQLite + vector storage
- [ ] **v1: Graph view**
  - [ ] Web UI with a single input that switches between write and find
  - [ ] Interactive zoomed-in subgraph for find results
  - [ ] Click-to-recentre navigation
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
