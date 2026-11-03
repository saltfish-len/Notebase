# Atomizer and Enricher (LLM)

Both boxes run on Claude through the Anthropic SDK (`notebase/blackbox/llm.py`).
The default model is **Claude Haiku 5.5** (`claude-haiku-5-5`).

| Box | Calls per write | Effort | Output |
|---|---|---|---|
| Atomizer | 1 | `medium` | `{"notes": [str]}`: self-contained notes in the user's own words and language |
| Enricher | 1 for all notes of the capture | `low` | per note: keywords (with synonyms and zh/en equivalents), 1–3 tags, entities |

## Is Haiku enough?

**Yes, for both boxes, at the current efforts (Atomizer `medium`, Enricher
`low`). Measured on 2026-11-02 ([#1](https://github.com/saltfish-len/Notebase/issues/1))
with `scripts/eval_llm.py`, and checked end to end with MongoDB and the real
embedder.** The original prompt was *not* good enough on Haiku. Four prompt
changes fixed it (below), and they are part of the default.

| | Haiku 5.5, original prompt | Sonnet 5.5, original prompt | **Haiku 5.5, final prompt** | Haiku 5.5, final prompt, Atomizer `high` | Sonnet 5.5, final prompt |
|---|---|---|---|---|---|
| Cases passing all checks | 1/8 | 8/8 | **10/10, 9/10, 9/10** (3 runs) | 10/10, 9/10, 9/10 (3 runs) | 10/10 |
| Fallbacks to the stub | 0 | 0 | **0** | 0 | 0 |
| Latency per call, p50 / p90 | 2.1 s / n/a | 2.4 s / n/a | **1.5 s / 2.3 s** | 2.1 s / 3.6 s | 2.2 s / 3.1 s |
| Tokens per write, in / out (both calls) | 1801 / 348 | 1819 / 350 | **1980 / 325** | 1980 / 610 | 1988 / 317 |
| Cost per 1000 writes | $0.35 | $7.14 | **$0.36** | $0.50 | $7.14 |

The original-prompt columns use the 8 original cases, scored with the
corrected checks (see "Fixes to the eval"). Output tokens include thinking. A
write makes two calls, one after the other, so a Haiku write spends about
3 s in the LLM.

The cost is about 20× lower than Sonnet, which matches the price ratio. The
back-of-envelope estimate of about $1 per 1000 writes was too high. Real
writes are longer than the eval captures, and the prompt carries up to 200
existing tags instead of about 10. **Not measured:** a large tag vocabulary
adds roughly 500–1000 input tokens, which is about $0.05–0.10 more per 1000
writes on Haiku.

### What reading the outputs showed

The automatic checks catch only obvious failures. These findings come from
reading the outputs side by side.

* **Atomizer: Haiku matched Sonnet from the start.** Both models made the
  same splits on every original case and kept every fact. Haiku's one miss
  was `"It also says caffeine…"`, which became `"Caffeine has…"`: the
  attribution to the book was dropped.
* **Enricher: Haiku had two real problems with the original prompt.**
  1. It never added Chinese keywords to English notes. It did add English
     keywords to Chinese notes, and once a Spanish "cafeina".
  2. It stretched existing tags. With `sleep, memory` in the vocabulary,
     every note in `brain-dump-en` was tagged `memory`, including the flight
     booking and the onboarding-UX notes.
* **End to end, Haiku translated a sentence.** In a mixed zh/en capture,
  `"Also need to renew my passport…"` became `"需要在十二月的东京之行前更新护照。"`.
  It did this 3 out of 3 times before the fix. The same run tagged a
  note-taking idea `documents`, a tag the passport note had just created.

### Prompt changes (all in `notebase/blackbox/llm.py`)

1. Atomizer: name the source when a sentence reports what a source says, and
   never drop the attribution.
2. Atomizer: captures often mix Chinese and English, so keep each part in the
   language the user wrote it in. An earlier wording, "the language its source
   *sentence* was written in", led to one sentence-by-sentence over-split, so
   it was rephrased.
3. Enricher: 5–10 keywords *in total*, always including at least two in the
   other language (Simplified Chinese for English notes, English for Chinese
   notes).
4. Enricher: reuse a tag only when the note is really about that topic, and
   never use catch-all tags (`other`, `misc`, `general`). One run produced
   `other`.

5. Atomizer (added with the Normalizer, 2026-11-03): lines like
   `[image: whiteboard sprint (whiteboard_sprint.jpg)]` stand for attached
   files. Copy each one verbatim into the note it goes with, even when that
   text already describes the file; a marker is a note of its own only when
   the capture has no text. The first wording ("put it in the note about what
   the file shows... a marker with nothing to go with is a note of its own")
   split the caption case into caption + bare marker in 3 of 3 runs. With the
   current wording, a 5-case marker check passed 15/15 (3 runs), and the full
   eval with the 4 new file cases passed 14/14, 14/14 and 13/14 on Haiku, 0
   fallbacks, ≈ $0.37 per 1000 writes. The one failure was `brain-dump-zh`
   returning too many keywords, the Enricher quirk listed below.

### What is still imperfect (accepted)

* **Translation still happens occasionally:** 1 of 9 runs on `mixed-dump`
  after the fix, against 3 of 3 before. Facts are kept and *find* still
  works, because the embedder is cross-lingual. Raising the Atomizer to
  `high` did not help (1 of 3), cost 40% more and added 0.6 s per call.
  Sonnet kept the language in 3 of 3 runs. If this matters to you, set
  `NOTEBASE_ATOMIZER_MODEL=claude-sonnet-5-5` and keep Haiku for enrichment.
  That costs roughly $3–4 per 1000 writes (estimated from the totals above;
  the cost per box was not measured).
* **Tags drift between neighbouring topics.** Notes on memory or learning
  land in `learning`, `productivity` or `neuroscience` depending on the run.
  Sonnet does the same with other pairs (`cooking` + `baking`, `design` for
  the Mind Map Book). In a real store the vocabulary is sorted by frequency,
  so the dominant tag should win over time. **Not measured.**
* **Haiku sometimes adds entities that are implied but not written**
  (Kahneman for *Thinking, Fast and Slow*, Tony Buzan for *The Mind Map
  Book*), and occasionally returns 11–14 keywords instead of at most 10.
  Both are harmless for *find*.

### Fixes to the eval (the checks were wrong, not the outputs)

* `refs_resolved` required `caffeine|coffee` to appear in at least 2 notes.
  Caffeine is a single idea, so the check failed Sonnet's correct output and
  missed Haiku's dropped attribution. It now checks pairs: *the note
  containing X must also name Y*.
* `url` required exactly 1 note. Both models split off the "read the part
  about numCandidates" task, which the prompt itself asks for (tasks are
  separate notes). It now accepts 1–2.
* `fallbacks` counted only failed API calls. It now counts every time a box
  actually used its stub, including validation failures such as a missing
  note index.
* New checks: `tags_vary` (a brain dump of unrelated ideas must not get one
  identical tag set) and `no_catchall_tags`.
* New cases: `mixed-dump` (each part keeps its language) and
  `bilingual-followup`. That second case was first written to expect 1 note,
  but every model split off the Chinese reason sentence, and the split is
  defensible, so it now accepts 1–2.

### Running the eval

```bash
export ANTHROPIC_API_KEY=...       # from https://platform.claude.com/
python scripts/eval_llm.py         # Haiku 5.5 vs Sonnet 5.5
```

```bash
python scripts/eval_llm.py --models claude-haiku-5-5 claude-opus-5-5
```

The eval runs 14 captures (English, Chinese, mixed, brain dumps, references,
URLs, hashtags, attached files) through both boxes. Captures with files go
through the stub Normalizer first, and `markers_once` checks that every file
marker lands in exactly one note. `--atomizer-effort high` tries a different
Atomizer effort, and `--tag=-run2` keeps the output files of repeated runs
apart; results vary from run to run, so run it more than once before
deciding. For each model it prints which automatic
checks fail, how many calls fell back to the stub, the resulting tag
vocabulary, latency, tokens and cost. Full outputs are saved to
`eval-out/<model>.json` for side-by-side reading. Automatic checks only catch
the obvious failures, so read the notes themselves before deciding. Running it
costs well under a cent on Haiku and about 7 cents on Sonnet.

## Design notes

* **Writes never fail because of the LLM.** The raw capture is stored first. On
  any API error, `refusal` stop reason (Haiku 5.5 has no server-side fallback),
  truncation, or unusable output (missing note index, empty split), the box
  logs a warning and uses its stub for that capture.
* **One Enricher call per capture**, not per note. This costs fewer calls and
  gives sibling notes consistent tags. Results are matched by `index`, not by
  position.
* **Tag vocabulary.** The store's most-used tags (`top_tags`, up to 200) are
  included in the prompt so that tags converge instead of multiplying. Tags are
  normalized (lowercase, spaces → `-`). `#hashtags` the user typed always
  become tags.
* **Cross-lingual keywords.** Chinese notes get English keywords and vice versa.
  This also makes MongoDB's `$text` index (which does not segment Chinese)
  useful for Chinese notes.
* **Contextualizer stays the identity stub.** The Atomizer prompt already makes
  each note self-contained, so a separate pass would cost another call for
  little gain.
* **Haiku 5.5 specifics.** Thinking is on by default and counts toward
  `max_tokens`, so the cap is 16000 and depth is set with `effort`. No
  `temperature` (it returns a 400) and no assistant prefill. The output format
  comes from structured outputs (`messages.parse` + Pydantic).
* **Prompt caching** is not used: the system prompts are below the minimum
  cacheable length, so caching would have no effect.
