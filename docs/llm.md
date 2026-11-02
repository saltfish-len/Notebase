# Atomizer and Enricher (LLM)

Both boxes run on Claude through the Anthropic SDK (`notebase/blackbox/llm.py`).
The default model is **Claude Haiku 5.5** (`claude-haiku-5-5`).

| Box | Calls per write | Effort | Output |
|---|---|---|---|
| Atomizer | 1 | `medium` | `{"notes": [str]}`: self-contained notes in the user's own words and language |
| Enricher | 1 for all notes of the capture | `low` | per note: keywords (with synonyms and zh/en equivalents), 1–3 tags, entities |

## Is Haiku enough?

**Probably yes, and it is the default. But this has not been measured yet:
run `scripts/eval_llm.py` with an API key to settle it.**

The case for Haiku:

* Both tasks are **bounded extraction and rewriting over short input**, not
  open-ended reasoning, and that is Haiku's intended workload. Structured
  outputs guarantee the JSON shape whatever model is used, so a smaller model
  can't break the format.
* It is roughly 20× cheaper than Sonnet 5.5 ($0.10/$0.50 vs $2/$10 per million
  tokens) and faster, and *write* sits on the user's critical path.
* Back-of-envelope estimate (not measured): a typical write is about 2–3K input
  and 1–2K output tokens across the two calls. That is about **$1 per 1000
  writes on Haiku** versus about $20 on Sonnet.

Where Haiku is most likely to fall short, and what the eval checks:

| Risk | Why | Eval case |
|---|---|---|
| Over- or under-splitting long brain dumps | Deciding what counts as "one idea" is a judgment call | `brain-dump-en`, `brain-dump-zh`, `one-idea-many-sentences` |
| Missing a reference ("he", "it") | Making a note self-contained needs the whole capture in view | `references` |
| Tag vocabulary fragmenting (`sleep` / `sleep-science` / `睡眠`) | Reusing tags needs consistency across calls | tag vocabulary printed at the end |
| Dropping facts while "cleaning up" | Rewriting can lose details | `facts_kept` on every case |

If Haiku fails specifically on atomization, the cheap fix is to upgrade only
that box: `NOTEBASE_ATOMIZER_MODEL=claude-sonnet-5-5` and keep Haiku for
enrichment. Raising the Atomizer's effort to `high` on Haiku is a cheaper
first step to try.

### Running the eval

```bash
export ANTHROPIC_API_KEY=...       # from https://platform.claude.com/
python scripts/eval_llm.py         # Haiku 5.5 vs Sonnet 5.5
```

```bash
python scripts/eval_llm.py --models claude-haiku-5-5 claude-opus-5-5
```

The eval runs 8 captures (English, Chinese, mixed, brain dumps, references,
URLs, hashtags) through both boxes. For each model it prints which automatic
checks fail, how many calls fell back to the stub, the resulting tag
vocabulary, latency, tokens and cost. Full outputs are saved to
`eval-out/<model>.json` for side-by-side reading. Automatic checks only catch
the obvious failures, so read the notes themselves before deciding. Running it
costs a few cents.

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
