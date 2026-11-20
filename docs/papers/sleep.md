# Paper notes: the sleep framework (Google), and an idea: memory inserted like context

**The sleep framework: a new way for LLMs to retain knowledge in continual learning.** Behrouz,
Hashemi, Javanmard, Mirrokni (Google Research, Cornell, USC).
[Blog post](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/),
2026-10-27. Paper: [arXiv:2606.03979](https://arxiv.org/abs/2606.03979).

Recorded on 2026-11-20. **Only the blog post was read, not the paper.** Numbers are the blog's, not ours.
It is a research result, not a Gemini feature.

## What it does (from the blog)

Two phases, like waking and sleeping:

- **Wake.** The model takes inputs and keeps what it learns in fast, short-term ("high-frequency")
  memory, such as the context.
- **Sleep.** It stops taking inputs and consolidates:
  - *Consolidation*: low-rank experts in pre-allocated, empty MoE slots are unlocked and joined to stable,
    "lower-frequency" memory blocks (built on Google's Hope / nested-learning architecture), so new
    knowledge does not overwrite old parameters.
  - *Knowledge seeding*: what sits in short-term memory is distilled into the new experts (generalized
    knowledge distillation on self-sampled sequences, plus an imitation reward by edit distance to the
    teacher's text).
  - *Dreaming*: the model generates its own practice samples, the router activates unusual experts at
    random, samples are filtered by gradient-based data selection, and the best are used for fine-tuning.
  - *Pruning*: the temporary short-term parameters are reset.

Reported: near-zero forgetting on continual translation of unseen languages (Manchu, Kalamang) where
ICL, Cartridges and SFT forget; "near-perfect" BABILong up to 10M tokens; 48.9% / 46.2% on SQuAD
knowledge incorporation (single task / continued pre-training); 79.2 / 69.0 / 46.1 on AIME 24 / AIME 25 /
HMMT 25 (mapping inferred from order). The blog gives no model sizes, no baseline numbers, no compute and
no limitations; the paper may.

## The idea (user, 2026-11-20)

**Goal: memory that is no longer external, but goes into the model as efficiently as context does.**
Today's loop is external: retrieve notes, paste them in as text, pay for every token, every time. The
goal is for the notes to be part of the model's forward pass instead.

Proposed route:

1. Take a small open model (a small Qwen) and add **empty MoE branches reserved for memory**.
2. Their weights are not trained by gradient descent at sleep time. A **hypernetwork** generates them
   from the notes, in one forward pass.
3. The generated weights *are* the memory: plugged in, the model knows the notes without retrieving or
   reading them.

## Feedback

### The spectrum between context and weights

"As efficient as context" has a precise meaning: inserting memory should be cheap, need no training at
insert time, and compose (add a group of notes, take it out again). The ways to hand knowledge to a
model sit on one line:

| Form | Insert cost | Use cost per query | Exact? |
| --- | --- | --- | --- |
| Text in the context (Notebase today, after *find*) | none | every token, every query | yes |
| Precomputed KV cache of the notes | one prefill | memory grows with the notes | yes |
| Compressed soft tokens per note (gist tokens, ICAE, xRAG) | one encoder pass | a few tokens per note | lossy |
| Trained KV cache per corpus (Cartridges) | training | fixed | lossy |
| Trained experts / LoRA (the sleep framework) | training | fixed | lossy |
| **Generated experts (this idea)** | **one hypernetwork pass** | **fixed** | lossy |

The idea is the last row: the hypernetwork turns "training" into "insertion", which is exactly what makes
weights behave like context. That is the right target. The rows above it are the stepping stones, and
each one can be measured against the first row on the same evaluation.

### What stays external

The goal moves memory *into* the model; it does not remove the notes. Weights are lossy, cannot cite,
cannot be cleanly edited or deleted, and can make things up. So the split is: **the notes are what is
remembered (source of truth, written by *write*), the generated weights are the loaded form (how the
model uses them)**. The weights can be rebuilt from the notes at any time, like embeddings today, and are
produced after the write (a sleep job is a maintenance tool, not a new verb), so a write never fails
because of a model. Deleting a note means regenerating its group's expert, which is cheap only because
generation is a forward pass and not training: one more reason the hypernetwork matters.

*Find* keeps its contract: what it shows the user are real notes. With memory inside the model, it can
also show what the model concluded, as long as each conclusion points at notes that the store confirms.

### What it would fix: F1

[F1](../find-cases.md#f1-吃什么-bad-example-2026-11-20): `吃什么` should bring up "a new malatang place
opened downstairs". Cosine similarity cannot make that jump, because the cue and the note share no topic,
only an inference (malatang is food, a new place is an option). A model that holds the notes in its
weights does the inference in the same pass that reads the cue. External retrieval has to guess the right
notes *before* the model ever thinks about them; internal memory does not.

### Prior work to read first (from memory, not checked in this session)

- Hypernetworks that turn text into adapter weights in one forward pass: Text-to-LoRA (Sakana AI, 2025),
  Generative Adapter (Chen et al., 2024), Drag-and-Drop LLMs (2025). These are the closest thing to the
  idea's step 2.
- Hypernetwork-based model editing: MEND (Mitchell et al., 2021).
- Trained per-corpus memory instead of context: Cartridges (2025, a baseline in the sleep paper).
- Large sparse memory layers: Memory Layers at Scale (Meta, 2024).
- Compressing text into a few soft tokens: Gisting (Mu et al., 2023), In-Context Autoencoder (Ge et al.,
  2023), xRAG (2024).

### Open problems

1. **Training the hypernetwork** is the expensive part: it is meta-learning over many (notes, questions)
   sets, needs a GPU, and must generalise to notes it has never seen. The sleep framework sidesteps this
   by training the experts directly.
2. **Many notes, one set of weights.** Generating a delta per note and adding them up interferes as the
   number grows; generating one expert from all notes runs into the hypernetwork's input length. Some
   grouping (per cluster, per time window) is needed, and its capacity is unknown.
3. **Base model.** The small Qwen models are dense (Qwen3-0.6B, the one used for embeddings); the Qwen
   MoE models are not small. "Empty experts" on a dense model means LoRA experts with a router, which is
   what the sleep framework's low-rank experts look like anyway.
4. **Compute here.** Notebase runs locally on CPU. Inference with a 0.6B model plus adapters is fine;
   training is not.

### Order of work (proposal)

Each step is measured against "notes as text in the context" on the find evaluation, so it is clear
what each step costs in quality and saves in tokens.

1. The find evaluation (see [em2mem.md](em2mem.md)), with F1 and later cases, and a Retriever-only
   baseline.
2. Soft tokens per note: an encoder (ICAE / gist style, on a small Qwen) compresses each note into a few
   vectors at write time, stored next to its embedding; found notes go in as those vectors instead of
   text. Cheapest test of "memory inserted, not read".
3. Does parametric memory help at all: a sleep job that trains one LoRA on a small Qwen with ordinary
   SFT on questions generated from the notes. Compare with steps 1 and 2.
4. Only if step 3 wins: replace that training with a hypernetwork that generates memory experts per group
   of notes (start from Text-to-LoRA or Generative Adapter rather than from scratch), and measure what is
   lost against the trained LoRA. This is the goal state.

## Proposed actions

| Action | Where | Status |
| --- | --- | --- |
| Read the paper (model sizes, compute, baselines, limitations) | this note | not done |
| Find evaluation with F1 as a case | `scripts/`, [#4](https://github.com/saltfish-len/Notebase/issues/4) | proposed (shared with lightmem.md, em2mem.md) |
| Soft-token note memory (ICAE / gist style) against text in context | Embedder side + find | idea |
| SFT LoRA of the notes in a sleep job | maintenance tool | idea |
| Hypernetwork-generated memory experts (the goal) | model + maintenance tool | idea, after the above |
