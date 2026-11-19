# Paper notes: EmbeddingGemma 2

**EmbeddingGemma 2: Unified and Efficient Multimodal Representations.** Schechter Vera, Dua et al.,
EmbeddingGemma Team, Google DeepMind. [arXiv:2610.37364v2](https://arxiv.org/abs/2610.37364) (2026-11-09).
Model: [ai.google.dev/gemma/docs/embeddinggemma](https://ai.google.dev/gemma/docs/embeddinggemma).

Read on 2026-11-19 against Notebase `main` at `4052e14`. Numbers below are the paper's, not ours. This is
the first paper in `docs/papers/` that is about the **Embedder** box, not memory design.

## What the paper does

EmbeddingGemma 2 is an open embedding model that puts text, images, audio, video and mixtures of them
(for example a caption with a photo) into **one vector space**. It is small for a multimodal model, and
the parts are modular:

| Part | Parameters | Loaded for |
|---|---|---|
| Bidirectional transformer backbone (from Gemma 4), with the token embedder | 271M | everything; text-only use needs nothing else |
| Vision encoder (ViT, trained along with the backbone) | 167M | images, video frames, visual documents |
| Audio encoder (USM conformer, frozen, from Gemma 4) | 305M | speech and other sound |

So it can run as a 271M text model, a 439M text + vision model, or the full 744M. Media become "soft
tokens" interleaved with text tokens in one sequence; the output is mean-pooled, projected and
L2-normalized to **768 dimensions**, with Matryoshka truncation to 512, 256 and 128.

- **Images** take 280 vision tokens by default (70–1120 selectable at inference). **Audio** is 25 tokens per
  second. **Video** is up to 32 frames at 1 fps, optionally interleaved with the soundtrack and `MM:SS`
  timestamps as text.
- **Task prefixes**, not free-form instructions. Retrieval queries are `task: search result | query: {text}`,
  documents are `title: {title | none} | text: {text}`. Symmetric tasks (clustering, sentence similarity)
  put the same prefix on both sides (Table 12).
- **Training** is a three-stage curriculum, about 2.9T tokens: text-only distillation from Gemini Embedding
  2, then multimodal distillation, then a contrastive finetuning stage with no distillation, followed by
  averaging two checkpoints of that last run. int4 quantized checkpoints are trained separately.
- **Score calibration is a training goal.** The contrastive loss adds "small squared-hinge penalties on
  positive and hard-negative similarities to calibrate the absolute score distribution for threshold-based
  tasks", and a spread-out regularizer pushes random pairs toward orthogonality (cosine near 0). The paper
  says both make "scores more calibrated and interpretable".

## Results

Summary against the specialists (Table 6, selected columns; higher is better):

| Benchmark | EmbeddingGemma 2 | EmbeddingGemma 1 (308M) | **Qwen3-Embedding-0.6B** (our default) |
|---|---|---|---|
| MTEB Multilingual v2, mean (task) | 61.4 | 61.2 | **64.3** |
| MTEB English v2, mean (task) | 68.5 | 69.7 | **70.5** |
| MTEB Code, mean (task) | **78.7** | 68.8 | 75.4 |
| MMEB-v2 (image, video, visual documents) | 59.0 | text only | text only |
| MSEB retrieval, spoken query → text (MRR@10) | 69.5 | 51.0 (via ASR) | text only |

From the multilingual leaderboard (Table 13), the task types closest to what Notebase does:

| MTEB Multilingual v2 | EmbeddingGemma 2 (768-d) | at 256-d | Qwen3-Embedding-0.6B |
|---|---|---|---|
| Retrieval | 60.8 | 58.7 | **64.6** |
| STS | 74.7 | 74.1 | **76.2** |
| Clustering | 50.4 | 51.1 | **52.3** |
| Reranking | **62.9** | 62.0 | 61.4 |

The 986M multimodal peer in Table 6 (by the text most likely jina-embeddings-v5-omni-nano; the rotated
column labels don't extract from the PDF) is ahead on text (MTEB Multilingual 65.5) and visual documents,
and far behind on images and video (MMEB-v2 43.6).

Ablations (MMEB-v2, intermediate recipe, one run each):

- **Native audio vs. ASR then text**, same model: 69.5 vs. 57.6 MRR@10 (Table 11). Even 256-d native audio
  (66.8) beats full-size ASR. Transcription loses accents, homophones and non-speech sound.
- **Training the vision encoder** instead of freezing it: 52.3 vs. 48.2 overall (Table 4).
- **Contrastive finetuning after distillation**, against more distillation for the same compute: 55.5 vs.
  52.3 (Table 3). Distillation plateaus.
- **Vision token budget at inference**: overall quality rises with tokens (70 → 560: 50.1 → 57.0), visual
  documents gain the most (51.6 → 67.6), and video gets worse past 280 tokens per frame (Table 5).
- **Truncation**: text quality barely moves down to 256-d (MTEB Multilingual 61.4 → 60.4) and drops at 128-d
  (57.9).
- **int4 weights**: MTEB Multilingual 61.4 → 60.8–61.0; images and audio lose more (MIEB 60.2 → 56.6–56.7,
  MSEB 69.5 → 63.9–65.0, with a different eval harness). Peak memory on a phone: 191 MB text-only, 567 MB
  full.

## How far to trust it

- **On text, it is not better than what we have.** By the paper's own tables, Qwen3-Embedding-0.6B is ahead
  on multilingual and English MTEB, and on retrieval, STS and clustering in particular. The win is code,
  which Notebase does not need. The abstract's "new Pareto frontier" is about quality *per parameter*
  across all modalities, not about absolute text quality.
- **No Chinese breakdown.** MTEB Multilingual averages over 250+ languages. Our requirement 2 (Chinese and
  cross-lingual zh↔en) is not measured on its own, and our requirement 1 (association without shared words)
  is not a benchmark task at all.
- **Baselines come from public leaderboards**, except the one multimodal peer that the authors ran
  themselves. The ablations use an intermediate recipe, single runs, no variance.
- **The calibration claim is qualitative.** The paper shows no score distributions for unrelated pairs, so
  "cosine near 0 for noise" is the training intent, not a measured result.

## Feedback for Notebase

### Not a reason to switch the text Embedder

The current choice rests on our own pair measurements (`docs/embedding.md`), not on MTEB. The paper gives no
reason to expect EmbeddingGemma 2 to beat Qwen3 there: it is behind on the closest MTEB task types. Two
things could still make it worth measuring:

- **Separation.** If the spread-out regularizer and the hinge penalties work as described, unrelated notes
  should score near cosine 0, which means a low `similarity_floor` and a wide related/unrelated gap
  (requirement 3). This is cheap to check with the same pairs, and the result decides it; the benchmark
  numbers don't.
- **Size.** Text-only it is 271M parameters and 768-d, against 596M and 1024-d. That could shorten the ~7 s
  model load per CLI command ([#2](https://github.com/saltfish-len/Notebase/issues/2)) and shrink stored
  vectors. Not measured.

If it is measured, the preset needs `document_prompt="title: none | text: "` and a query prefix. Test both the
paper's `task: search result | query: ` and our notes instruction, the way the Qwen3 instruction was tested.
`ModelPreset` already has both fields, so this is configuration, not code.

### Where it could matter: images without the API, and voice memos

Today an image becomes **text** before anything is embedded: the `VisionNormalizer` asks Claude for alt
text, and without an API key the image is only its file name. A model that embeds images natively offers a
second path:

- **Offline image find.** With `NOTEBASE_LLM=none`, an image note would still be findable by what is in it.
- **Mixed input.** A note's text and its image could be embedded together, so the vector carries what the
  alt text leaves out. This is the same direction as EM²Mem's Gap 3 ("keep a way back to the pixels",
  [em2mem.md](em2mem.md)).
- **Voice memos.** The roadmap plans local speech-to-text for audio. Table 11 says embedding the audio
  natively retrieves much better than embedding its transcript. The transcript is still needed, because
  the Atomizer and Enricher work on words, but the vector could come from the audio.

What it would cost, and why it is not a small change:

- **The `Embedder` Protocol takes text only** (`embed_documents(texts)`). Images or audio would need an
  interface change, which touches the write pipeline, not just one box.
- **One database, one vector space.** Image vectors from EmbeddingGemma 2 can't sit next to Qwen3 text
  vectors (`EmbeddingMismatch`). Native image embedding means moving the *whole* database to EmbeddingGemma 2,
  so it only makes sense if the text measurement above comes out acceptable.
- **Bytes are needed at embed time.** At write time they are there. A later re-embed is not possible,
  because captures keep only file references (D11). This is one more reason to revisit D11.
- Alt text stays in any case: it is what the user sees, what the LLM boxes read, and what keyword search
  matches.

### What not to adopt

- **Video** (32 frames, timestamps): Notebase has no video capture, and the roadmap doesn't plan one.
- **Their training recipe** (distillation, curriculum, checkpoint averaging): we use models, we don't train
  them.
- **Embedding translation between models** (refs. 53, 54, mentioned as a way to avoid re-embedding): we keep
  the raw text, so re-embedding is simpler and exact. Translation would only matter for image vectors whose
  bytes we no longer have.

## Proposed actions

| Action | Where | Status |
|---|---|---|
| Add an `embeddinggemma-2` preset (text-only) and run the `docs/embedding.md` pair measurement: association, zh, en↔zh, unrelated, plus the query-prefix test. Adopt only if it separates at least as well as Qwen3 | `notebase/blackbox/embedders.py`, `docs/embedding.md` | proposed, unmeasured |
| If it does: measure image find with native image vectors against alt text through Qwen3, with `scripts/eval_vision.py` and the find evaluation proposed in [lightmem.md](lightmem.md) | Embedder Protocol, Normalizer | proposed, unmeasured |
| When the audio Normalizer is built: compare finding voice memos by audio vector and by transcript vector | Normalizer, Embedder | later |
