# Paper notes: EM²Mem

**EM²Mem: Event-Centric Multimodal Memory for Large Language Models.** Chen, Zheng et al., Zhejiang
University, South China University of Technology and Lenovo. [arXiv:2609.00551v1](https://arxiv.org/abs/2609.00551)
(2026-09-01). It comes from the same group as [LightMem](lightmem.md), and its code is to be merged into
[zjunlp/LightMem](https://github.com/zjunlp/LightMem).

Read on 2026-11-03 against Notebase `main` at `f82ad8b`. Numbers below are the paper's, not ours.

## What the paper does

The task is long-video question answering: hours to a week of egocentric video, where the evidence for a
question is sparse and spread across frames, speech, OCR and recurring behaviour. The paper argues that
existing multimodal memories are **fragment-centric**. They store captions, keyframes, transcripts and graph
triples as separate units, so retrieval returns loose fragments, and the LLM must re-align them across
modalities and time while answering. The authors call this *retrieve-then-align*. EM²Mem does
**align-then-retrieve**: the binding happens when memory is built.

### Memory schema

- **Event anchor** `e_i = (i, start, end)`. The video is cut into 30-second segments. An anchor is a
  temporal *address*, not content: it is the shared key that all evidence about that moment is attached to.
- **Event memory cell** `μ_i = (e_i, R_i, C_i)`:
  - `R_i`, the **local record**: keyframe caption, transcript, representative keyframes, and **structured
    fields** `z_i = {actions, objects, topics, scenes, entities}`.
  - `C_i`, the **temporal context views**: summaries at several scales (3 min, 10 min, 1 h), each attached
    to every event it covers.
- **Episodic graph** `G_E`: events connected through shared typed entities (people, objects, places,
  scenes, topics) and through temporal transitions between adjacent events.
- **Semantic graph** `G_S`: long-term regularities (habits, preferences, routines, stable relations)
  summarized from episodic evidence, **each linked back to the events that support it**.

### Retrieval

1. Score event cells using their local records, context views and graph-linked evidence.
2. Expand along temporal transitions, shared entities, objects, places, topics and relevant semantic facts.
3. An LLM selector keeps only the cells needed for the question.
4. Build a compact evidence view (captions, transcripts, fields, summaries, facts, a few keyframes) and
   answer.

## Results

Three multiple-choice benchmarks: EgoLifeQA, Ego-R1 Bench and Video-MME (Long).

| | Strongest memory baseline (WorldMM) | EM²Mem |
|---|---|---|
| EgoLifeQA accuracy | 64.0 (reproduced) | **66.0** |
| Ego-R1 Bench accuracy | 65.3 (as published) | **67.7** |
| Video-MME (L) accuracy | 73.1 (reproduced) | **76.8** |
| Latency per query (EgoLifeQA) | 459 s | **98 s** (4.67×) |
| Inference tokens (EgoLifeQA) | 42.0 M | **15.3 M** (−64%) |
| Strict 30 s evidence recall@5 | 23.8 (after 5 retrieval rounds) | **30.8** (one pass) |

Building the memory costs more up front. Counting only wall-clock time, that cost is paid back after about
23–24 queries; token cost needs more queries to break even.

### Ablations (EgoLifeQA, Table 4)

| Variant | Accuracy | Δ |
|---|---|---|
| Full EM²Mem | 66.0 | – |
| without temporal context views | 60.4 | −5.6 |
| without semantic memory | 61.4 | −4.6 |
| without episodic graph | 61.6 | −4.4 |
| local 30 s event records only | 64.0 | −2.0 |

### Representation × when evidence is unified (first 250 EgoLifeQA questions, Table 5)

| Visual evidence stored as | Fused at retrieval | Unified at construction |
|---|---|---|
| Raw frames | 66.8 | 68.0 |
| Flattened captions | 65.2 | 65.6 |
| **Structured event fields** | 68.0 | **71.2** |

Adding keyframes to the final evidence view raises accuracy from 63.2 (no keyframes) to 66.0 (three).

## How far to trust it

- **The simplest variant is close.** Retrieving local 30-second records only scores 64.0, against 66.0
  for the full system. This is the same pattern as NaiveRAG vs LightMem.
- **Removing any single component scores below that simple baseline** (60.4–61.6 < 64.0). The parts help
  only in combination. Taken alone, graph expansion or semantic facts add noise. That matters for our
  Spreader.
- **Small samples.** The Table 5 analysis uses 250 questions. At about 66% accuracy, the standard error is
  about 3 points, so 2–3 point differences are within noise. No variance across runs is reported.
- **Mixed baselines.** On Ego-R1 Bench, the WorldMM number is the published one, not a reproduction under
  the same setup.
- **Retrieval is still weak in absolute terms**: 30.8% recall@5 at the event level.
- **Their own limitations:** turning pixels into text fields loses fine visual detail (small objects,
  colours, layout), errors in captioning and extraction spread into the cells and graphs, and the design
  suits data that is processed once and queried many times.

## Feedback for Notebase

### What it supports in our current design

- **Do the organizing at write time.** "Unify at construction, not at retrieval" (71.2 vs 68.0) is the
  README's idea of moving all organization into the system at write time. The Enricher already produces
  keywords, tags and entities during *write*.
- **Placing attachments where they belong (D15).** Putting an image's description where it was embedded,
  not at the end of the capture, is construction-time binding in miniature. Our own end-to-end run found
  the same failure the paper describes: appended at the end, the whiteboard description went into the
  wrong note.
- **Multimodal input as text in one embedding space (D10).** EM²Mem also turns visual evidence into text
  fields for retrieval, and keeps the pixels only for later checking.

### Gap 1: entities are extracted but never used

The paper's strongest representation result is that **structured fields beat the same information
flattened into prose** (71.2 vs 65.6), and removing the entity- and time-based episodic graph costs 4.4
points. In our code:

- the Enricher extracts `entities` (and the stub extracts URLs), and they are stored on each note;
- `NearestNeighborLinker` (`notebase/blackbox/stubs.py`) uses **only** embedding similarity, so the README's
  example `reason: "shared-entity"` link is never created;
- neither the Retriever nor the Spreader reads entities.

**Proposal:** a Linker that adds `shared-entity` links (same normalized entity name and type), at a weight
below strong semantic links. It needs no extra model call. Imported Obsidian wikilinks, which are on the
roadmap as "wikilinks as graph links", are explicit structural links of the same kind.

### Gap 2: the capture and time are not used for association

Temporal context views are the most valuable component in the ablation (−5.6). For us, the natural
"event" is the **capture**: one write session, split by the Atomizer into several notes.

- Today, notes from the same capture are linked only if their similarity clears the threshold (they are
  passed as `siblings`, but scored like any other note). The Spreader cannot follow `capture_id`.
  Atomizing splits a train of thought apart, and nothing puts it back together at *find* time.
- **Proposal:** a low-weight `same-capture` link (or a Spreader step that expands to sibling notes), plus
  an optional `temporal` link for notes written close together. `docs/architecture.md` already plans
  "temporal proximity" for the Linker; the paper is evidence for it.

### Gap 3: the image description is one flat sentence

`VisionNormalizer` returns 1–3 sentences of alt text inside a marker. The paper finds flattened captions are
the *weakest* representation, below even raw frames. Two cheap steps, both to be **measured with
`scripts/eval_vision.py` and a find evaluation before adoption**:

- make sure the Enricher's entities for an image note include what is visible in it (objects, people,
  text), so Gap 1's entity links reach images;
- keep a way back to the pixels. The paper's keyframe result (63.2 → 66.0) and its stated limitation both
  say text loses detail. D11 stores only a file reference, not the bytes, and already says to revisit this
  for a real Normalizer.

### Be careful with a "semantic graph"

EM²Mem's semantic facts are generalizations written by an LLM. In Notebase every note is something the user
thought. A generated generalization must not look like one of the user's notes. If we build anything similar,
it belongs in a **derived layer**: cluster labels from the Labeler, or typed links from the offline
consolidation proposed in [lightmem.md](lightmem.md). Like the paper, it should always point back to the
notes that support it.

### What this means for the Spreader ([#4](https://github.com/saltfish-len/Notebase/issues/4))

Expanding along the graph without the other components scored *below* the plain baseline. Spreading more is
not automatically better. Hop count, decay and which link types to follow should be set by measurement,
which is also the README's open question "how many hops should spreading activation take".

### Borrow the metric: strict evidence recall@k

The paper measures retrieval directly: a hit counts only if the retrieved unit is the annotated evidence. No
LLM judge is needed. Proposed find evaluation for Notebase:

- a set of notes, and cues each labelled with the note ids that should be found, in Chinese, English and
  across the two;
- **recall@k for seeds** (the Retriever) and **recall within the returned region** (after the Spreader),
  reported separately;
- the baseline is the Retriever alone. A Spreader, Clusterer or new link type is adopted only if it raises
  region recall without burying the seeds;
- LongMemEval ships evidence labels and could be converted (user turns as writes, questions as cues). It
  is dialogue, not notes, so it would supplement a small hand-made Notebase set, not replace it.

## Proposed actions

| Action | Where | Status |
|---|---|---|
| Find evaluation: evidence recall@k for seeds and region, with a Retriever-only baseline | `scripts/`, [#4](https://github.com/saltfish-len/Notebase/issues/4) | proposed (shared with [lightmem.md](lightmem.md)) |
| Linker: `shared-entity` links, then `same-capture` and `temporal` links; keep only what the evaluation supports | Linker / Spreader | proposed |
| Image notes: entities for visible content; revisit keeping image bytes (D11) | Normalizer / Enricher | proposed, unmeasured |
| Any generated generalization goes in a derived layer that points back to its notes, never as a user note | design rule | proposed |
