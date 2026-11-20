# Paper notes: the sleep framework (Google), and an idea: notes as generated expert weights

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

Do the same for Notebase, but replace the training step with a model:

1. Take a small open model (a small Qwen) and add **empty MoE branches reserved for memory**.
2. Their weights are not trained by gradient descent at sleep time. A **hypernetwork** generates them
   from the notes.
3. The generated weights *are* the memory: the model "knows" the notes without retrieving them.

## Feedback

### Where it fits, and where it does not

Notebase's promise is that *find* returns **your notes**, verbatim and traceable to a capture. Weights
cannot cite, cannot be cleanly edited or deleted, and can make things up. So parametric memory cannot
replace the store. It can sit *beside* it, as a derived index, the same status embeddings have today:
rebuilt from the notes at any time, never the source of truth. That also keeps the rule that a write
never fails because of a model: the weights are produced later, in a sleep job (a maintenance tool, not
a new verb), and a find still works without them.

The most useful place for it is the **Retriever / QueryInterpreter**, and the motivating case is
[F1](../find-cases.md#f1-吃什么-bad-example-2026-11-20): `吃什么` should bring up "a new malatang place
opened downstairs". Cosine similarity cannot make that jump; a model that has absorbed the notes might.
Concretely: the memory-augmented model reads the cue and **generates note ids** (or rewrites the cue into
note-like text), and those ids are then looked up in the store. Every answer is still a real note, and a
wrong id is harmless. This is generative retrieval in the style of the Differentiable Search Index
(Tay et al., 2022), personalised by the memory experts.

### Prior work to read first (from memory, not checked in this session)

- Hypernetworks that turn text into adapter weights in one forward pass: Text-to-LoRA (Sakana AI, 2025),
  Generative Adapter (Chen et al., 2024), Drag-and-Drop LLMs (2025). These are the closest thing to the
  idea's step 2.
- Hypernetwork-based model editing: MEND (Mitchell et al., 2021).
- Trained per-corpus memory instead of context: Cartridges (2025, a baseline in the sleep paper).
- Large sparse memory layers: Memory Layers at Scale (Meta, 2024).

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

1. The find evaluation first (see [em2mem.md](em2mem.md)), with F1 and later cases in it, and a
   Retriever-only baseline. Nothing below is worth building without it.
2. Cheap check that parametric memory helps at all: a sleep job that trains one LoRA on a small Qwen with
   ordinary SFT, on synthetic (question → note id) pairs generated from the notes. Compare with the
   baseline on the evaluation.
3. Only if step 2 wins: replace that training with a hypernetwork (start from Text-to-LoRA or Generative
   Adapter rather than from scratch), and measure what is lost against the trained LoRA.

## Proposed actions

| Action | Where | Status |
| --- | --- | --- |
| Read the paper (model sizes, compute, baselines, limitations) | this note | not done |
| Find evaluation with F1 as a case | `scripts/`, [#4](https://github.com/saltfish-len/Notebase/issues/4) | proposed (shared with lightmem.md, em2mem.md) |
| Generative-retrieval LoRA trained in a sleep job, against the baseline | maintenance tool + Retriever | idea |
| Hypernetwork-generated memory experts | Retriever | idea, after the above |
