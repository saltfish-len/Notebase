# Paper notes: LightMem

**LightMem: Lightweight and Efficient Memory-Augmented Generation.** Fang et al., Zhejiang University.
ICLR 2026. [arXiv:2510.18866](https://arxiv.org/abs/2510.18866) (v4, 2026-02-28). Code:
[zjunlp/LightMem](https://github.com/zjunlp/LightMem).

Read on 2026-11-03 against Notebase `main` at `f82ad8b`. Numbers below are the paper's, not ours.

## What the paper does

LightMem is a memory system for LLM agents in long multi-turn dialogue. Its argument is that existing
systems (Mem0, A-MEM, MemoryOS, LangMem) spend most of their cost on three things: feeding redundant raw
dialogue to an LLM, summarizing at a fixed granularity (every turn or every session), and updating memory
online, one entry after another. It borrows the Atkinson–Shiffrin model of human memory and uses three
stages.

### 1. Sensory memory: pre-compress, then segment by topic

- **Pre-compression.** LLMLingua-2 classifies each token as keep or drop, and only the top `r` fraction
  is kept (`r` = 0.5–0.8). Fed straight to an LLM for QA, text compressed at 50–80% scores about the same as
  the uncompressed text.
- **Topic segmentation.** Compressed turns fill a 512-token buffer. When it is full, boundaries are the
  **intersection** of two candidate sets:
  - B1: local maxima of attention between consecutive turns (from the compression model);
  - B2: adjacent turns whose embedding similarity is below a threshold.

  Measured against session boundaries in LongMemEval, the hybrid finds over 80% of boundaries, better
  than either signal alone.

### 2. Short-term memory: summarize per topic, in batches

Topic segments wait in a second buffer (`th` = 256–1024 tokens). When it fills, one LLM call summarizes each
topic. The stored entry is `{topic, embedding(summary), raw user turn, raw model turn}`: the summary is
indexed, and the raw text is kept next to it.

### 3. Long-term memory: soft updates online, consolidation during "sleep"

- **Online: insert only.** New entries go straight into the store with a timestamp. Nothing is
  rewritten or deleted while the user is waiting.
- **Offline update queue.** For each entry `e_i`, the candidates that may update it are its top-k most
  similar entries with a **later or equal timestamp**: `Q(e_i) = Topk{e_j : sim(v_i, v_j), t_j ≥ t_i}`.
  This is only a similarity lookup, so it is cheap.
- **Parallel consolidation.** Each queue is independent, so the LLM update calls (merge, de-duplicate,
  resolve conflicts) run in parallel, not one after another as in online systems.

The paper's case study (§5.6) explains why online updates are risky. Given "Monday 2 PM: planning a trip to
Tokyo" and "Monday 4 PM: asks about trains to Kyoto", an LLM doing a hard update rewrote the memory to "user
plans Kyoto trip" and lost Tokyo. An LLM asked to update in real time can mistake related information for
conflicting information and delete the old entry for good. Appending keeps both.

## Results

Incremental, turn-by-turn feeding. LongMemEval-S (500 questions) and LoCoMo, with an LLM judge
(GPT-4o-mini). Backbones: GPT-4o-mini, Qwen3-30B-A3B, GLM-4.6.

LongMemEval-S accuracy (Table 2):

| Backbone | FullText | NaiveRAG | A-MEM (strongest baseline) | Mem0 | LightMem (range over `r`, `th`) |
|---|---|---|---|---|---|
| GPT-4o-mini | 56.8 | 61.0 | 62.6 | 53.6 | 64.3–68.6 |
| Qwen3-30B-A3B | 54.8 | 60.8 | 65.2 | 39.5 | 61.9–70.2 |
| GLM-4.6 | 36.7 | **73.2** | 70.6 | – | 72.8–73.2 |

Efficiency, online plus offline, against the baselines: up to 38× fewer tokens, 30× fewer API calls and
12× faster memory construction on LongMemEval with GPT. Counting online costs only, the gaps are about
100× in tokens and calls.

Ablations:

- **Topic segmentation removed:** accuracy −6.3 points (GPT) and −5.4 (Qwen), with only small savings.
- **Buffer size `th`:** bigger is always cheaper, but the effect on accuracy is non-monotonic, and the
  best value depends on the model and on `r`.
- **Compression ratio:** the best `r` for accuracy averages 0.6.

## How far to trust it

- **NaiveRAG is a strong baseline.** On GLM-4.6 it ties LightMem (73.2 vs 72.8–73.2) and beats A-MEM. On
  GPT, NaiveRAG trails A-MEM by only 1.6 points. Much of the gap between elaborate memory systems and plain
  vector retrieval is small, and sometimes zero.
- **The headline numbers are efficiency.** The 38× and 100× figures come from comparing against systems
  that call an LLM on every turn. A system that already batches does not gain them.
- **Retrieval is held fixed.** Every method uses the same retriever, top-k and answer prompt, so the paper
  says nothing about how to *find* memories.
- The accuracy results are single numbers per configuration, with no variance reported.

## Feedback for Notebase

### What it supports in our current design

- **Atomizing.** Topic segmentation is worth 5–6 points in the ablation. Our Atomizer splits more finely
  (one idea per note), but in the same direction.
- **Keeping the raw input.** LightMem stores the raw turns next to the summary. We keep the capture and
  link every note to it (`capture_id`).
- **Mostly append-only.** The §5.6 case study is direct evidence for the README principle. An LLM updating
  memory in real time can mistake related notes for conflicting ones and delete one of them. This is the
  same idea as D8 (a write never fails or loses data because of a model), carried over to updates.

### What to adopt: consolidation during "sleep" (the design for v2 superseding)

The roadmap has open items that all need superseding and none of them has a design yet: "Superseding and
versioning of thoughts", "Re-import of an edited file replaces its old notes" (D14 defers this), and the
tag drift that `docs/llm.md` lists as a known Enricher weakness. LightMem suggests a shape for them:

1. **The write path stays insert-only.** No supersede or merge decisions while the user waits.
2. **A maintenance tool runs consolidation offline** (for example `notebase consolidate`, also exposed
   through MCP). CLAUDE.md allows maintenance tools, so this adds no user verb.
3. **The update queue is almost free.** The Linker already finds each new note's nearest neighbours. The
   pairs where a newer note is similar to an older one are the LightMem queue. They can be recorded at
   write time, or recomputed from the stored links.
4. **The LLM judges each pair and writes a typed link, never a deletion:** `supersedes`, `contradicts`,
   `elaborates`, `duplicate`. Pairs are independent, so they can be batched or run in parallel. This is
   also where the "LLM-judged relations" planned for the Linker in `docs/architecture.md` should go, so
   writes don't get slower.
5. **Re-importing an edited file** becomes a special case. The new version's notes are inserted, and
   consolidation marks the old version's notes as superseded, with the import path or the previous
   fingerprint as a strong hint.
6. **Tag merging** runs in the same pass. That answers the README open question "fixed vocabulary or emerge
   and merge": let tags emerge, then merge them offline.

### What not to adopt

- **Token pre-compression.** The redundancy it removes comes from agent dialogue. A note is something the
  user wrote on purpose, so dropping tokens loses information and breaks "keep the user's voice". Cost is
  not a problem for us either: $0.36 per 1000 writes on Haiku (D7). LLMLingua-2 is also not measured on
  Chinese.
- **Batching across writes (the STM buffer).** Each write is a separate user action, and the Enricher is
  already batched per capture (D7). Waiting for a buffer to fill would only delay when a note can be found.

### Where its segmentation does apply: long imports

Since the Obsidian import (D15), one Markdown file becomes one capture, up to `MAX_TEXT_BYTES` = 200 KB
(`notebase/importing.py`), and goes to the Atomizer in a single call with `max_tokens=16000`
(`notebase/blackbox/llm.py`). For a long file the atomized output can exceed that limit. The call then fails
with "output truncated" and the whole file falls back to the paragraph-split stub.
**This is an inference from the code; it has not been measured** (the DEVLOG says a large real vault has not
been tested). A LightMem-style split before the Atomizer would fix it:

- split on Markdown headings first, since they are free and reliable;
- if a section is still too long, cut where adjacent paragraphs have low embedding similarity. This is the
  paper's B2 signal, computed with our existing Qwen3 embedder, with the threshold on the calibrated scale
  (D5);
- then atomize each segment, keeping the file as the capture.

The same similarity split could also improve the Atomizer's stub fallback, which today splits only on blank
lines.

### On the find side

The paper doesn't study retrieval. The useful lesson is indirect: since NaiveRAG ties much more complex
systems, **every find-side box should be measured against a plain-retrieval baseline** (Retriever only, no
spreading) before it is adopted. See the evaluation proposal in [em2mem.md](em2mem.md).

## Proposed actions

| Action | Where | Status |
|---|---|---|
| Find-side evaluation with a Retriever-only baseline, before [#4](https://github.com/saltfish-len/Notebase/issues/4) | `scripts/`, [#4](https://github.com/saltfish-len/Notebase/issues/4) | proposed |
| Offline consolidation tool: update queue from Linker neighbours, typed links (`supersedes` / `contradicts` / `elaborates` / `duplicate`), tag merging, no deletions | v2 superseding | proposed |
| Segment long imports (headings, then embedding similarity) before the Atomizer; first measure whether long files actually hit `max_tokens` | `notebase/importing.py`, Atomizer | proposed, unmeasured |
