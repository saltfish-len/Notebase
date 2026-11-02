# Architecture

Notebase has two verbs, `write` and `find`. Each one is a fixed pipeline of
**black boxes**: an interface whose implementation can be swapped without
touching anything else. Right now every box has a placeholder stub
(`notebase/blackbox/stubs.py`), so the whole system runs end to end on MongoDB.
Boxes get replaced by real implementations one at a time. So far the
**Embedder** is real (`notebase/blackbox/embedders.py`, see
[embedding.md](embedding.md)).

## Layers

```mermaid
flowchart LR
    subgraph Client
        CLI["CLI<br/>notebase write / find"]
    end

    subgraph App["Notebase (app.py)"]
        WP["WritePipeline<br/>write.py"]
        FP["FindPipeline<br/>find.py"]
    end

    subgraph BB["Black boxes (blackbox/interfaces.py)"]
        direction TB
        A[Atomizer] --- C[Contextualizer] --- E[Enricher] --- EM[Embedder] --- L[Linker]
        QI[QueryInterpreter] --- R[Retriever] --- S[Spreader] --- CL[Clusterer] --- LB[Labeler]
    end

    subgraph Storage["NoteStore (storage/base.py)"]
        M[("MongoDB<br/>captures · notes<br/>$vectorSearch · $text")]
        MEM[("InMemory<br/>tests")]
    end

    CLI --> WP & FP
    WP --> A
    FP --> QI
    WP & FP --> Storage
    L & R & S -.-> Storage
```

* **Pipelines** only orchestrate. They contain no algorithmic decisions.
* **Black boxes** hold all the algorithmic decisions. Each is a `Protocol`.
* **NoteStore** is the only way to persist or query data. MongoDB is the
  real backend, and an in-memory store with the same contract is used for tests.

## Write

```mermaid
flowchart LR
    IN([raw text]) --> CAP[capture<br/><i>save verbatim</i>]
    CAP --> AT[[Atomizer]]
    AT -- fragments --> CTX[[Contextualizer]]
    CTX -- self-contained text --> EN[[Enricher]]
    EN -- keywords · tags · entities --> EMB[[Embedder]]
    EMB -- vector --> LK[[Linker]]
    LK -- weighted links --> ST[(store note +<br/>mirror links)]
    CAP --> DB1[(captures)]
    ST --> DB2[(notes)]
```

| Black box | Contract | Current stub | Intended real version |
|---|---|---|---|
| Atomizer | `Capture -> list[str]` | split on blank lines | LLM splits input into one idea per note |
| Contextualizer | `(fragment, Capture) -> str` | identity | LLM resolves pronouns and references so the note stands alone |
| Enricher | `str -> Enrichment` | top-frequency words, `#hashtags` as tags, URLs as entities | LLM keywords + synonyms, tags normalized against existing vocabulary, NER |
| Embedder | `embed_documents(list[str])`, `embed_query(str)` | ✅ **real**: Qwen3-Embedding-0.6B (local); hashing stub kept for tests | n/a |
| Linker | `(Note, store, siblings) -> list[Link]` | top-k calibrated similarity above a threshold | similarity + shared entities + temporal proximity, LLM-judged relations |

## Find

```mermaid
flowchart LR
    Q([cue]) --> QI[[QueryInterpreter]]
    QI -- keywords · vector · tag filter --> R[[Retriever]]
    R -- seed notes --> S[[Spreader]]
    S -- activation map --> CL[[Clusterer]]
    CL -- groups --> LB[[Labeler]]
    LB --> OUT([FindResult<br/>seeds · notes · edges · clusters])
    R <-.-> VS[($vectorSearch)]
    R <-.-> TS[($text)]
    S <-.-> G[(notes.links)]
```

| Black box | Contract | Current stub | Intended real version |
|---|---|---|---|
| QueryInterpreter | `str -> Query` | tokenize + embed; `#tag` becomes a filter | LLM query expansion, time and entity filters ("last summer", "from Alice") |
| Retriever | `(Query, store, k) -> list[ScoredNote]` | reciprocal-rank fusion of vector and keyword hits | tuned hybrid search, reranker |
| Spreader | `(seeds, store) -> {id: activation}` | BFS, `activation × weight × decay` per hop | spreading activation with recency and frequency weighting |
| Clusterer | `notes -> list[list[id]]` | connected components over strong links | community detection (Louvain/Leiden) or embedding clustering |
| Labeler | `list[Note] -> str` | most common tags | LLM summary label |

`FindResult` is everything a graph view needs to render the zoomed-in region:
`seeds` (highlighted centre), `notes` (nodes), `activation` (node size or
distance), `edges` (only those inside the region) and `clusters` (groups and
their labels).

## MongoDB data model

**`captures`**: raw input, never modified.

```jsonc
{ "_id": "c_…", "text": "…", "source": "typed", "created_at": ISODate }
```

**`notes`**: one document per atomic note. The vector and the graph edges live
on the note itself.

```jsonc
{
  "_id": "n_…",
  "content": "…",
  "capture_id": "c_…",
  "source": "typed",
  "keywords": ["…"],
  "tags": ["…"],
  "entities": [{ "type": "url", "name": "…" }],
  "embedding": [0.01, …],                         // vector
  "links": [{ "to": "n_…", "weight": 0.62, "reason": "semantic" }],  // edges
  "created_at": ISODate
}
```

| Index | Type | Used by |
|---|---|---|
| `note_embedding` on `embedding` (cosine) + filter `tags` | Vector Search | `vector_search` → Linker, Retriever |
| `note_text` on `content`, `keywords` | text | `keyword_search` → Retriever |
| `links.to` | B-tree | reverse link lookups |
| `capture_id` | B-tree | provenance |

Links are **mirrored**: when A links to B, B also gets a link to A, because
association runs both ways. Edges are stored as an embedded array, so one
`find({_id: {$in: frontier}})` per hop is enough to spread. If the graph gets
large, the edges can move to their own collection and use `$graphLookup`
without changing the `NoteStore` contract.

### Caveats

* `$vectorSearch` needs Atlas, the `mongodb/mongodb-atlas-local` image (see
  `docker-compose.yml`) or Community 8.2+ with `mongot`.
* The vector index is **eventually consistent**: a note becomes searchable
  shortly after it is written. Notes from the *same* capture are handed to the
  Linker as `siblings` so they always link to each other. Notes written by
  separate captures less than about a second apart may miss each other.
* Atlas reports cosine scores as `(1 + cos) / 2`. `MongoNoteStore` converts
  them back, so the store contract always returns raw cosine.
* MongoDB's `$text` index does not segment Chinese. The stub enricher stores
  CJK bigrams in `keywords` as a workaround. A real setup would use Atlas
  Search with a CJK analyzer.
