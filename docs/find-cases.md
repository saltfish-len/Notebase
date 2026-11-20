# Find cases

Real finds whose result was wrong, kept as test cases for the find evaluation
proposed in [papers/em2mem.md](papers/em2mem.md) (evidence recall@k for seeds
and region, with a Retriever-only baseline). Each case records the cue, the
notes in the database that matter, what came back, and what should have come
back. Diagnoses are hypotheses until the evaluation measures them.

## F1: 吃什么 (bad example, 2026-11-20)

**Cue:** `吃什么` ("what should I eat"), in the web UI.

**Relevant notes:**

| Note | Role |
| --- | --- |
| 楼下开了一家新的麻辣烫。 ("A new malatang place opened downstairs.") | **wanted** |
| 今天炒饭放了三个小碗 ("Put three small bowls in the fried rice today.") | topically close, not an answer |
| 吃什么 | the cue itself, written earlier as a note |
| #all | the reserved tag, written as a note |

**Got:** 4 notes in 4 clusters, 604 ms. Seeds were `吃什么` and the fried-rice
note. The malatang note was not a seed; it came in by spreading through one
link, with activation 0.08, the smallest node in the graph.

**Wanted:** the malatang note first. The cue is a decision ("what to eat"), and
the useful answer is an option for that decision (a new place to eat nearby),
not the note most similar in topic (cooking). Relevance by similarity is not
what the user meant.

**Why it fails (hypotheses, not measured):**

1. The QueryInterpreter passes the cue on as is. Nothing turns a question of
   intent into what would answer it (places, dishes, restaurants). The
   embedding of `吃什么` is closer to a note about cooking than to a sentence
   that names a restaurant without the word "eat".
2. Nothing in the ranking rewards novelty or actionability ("新开的", "new").
3. The malatang note has character-bigram tags and keywords (`#楼下 #下开`,
   `了一`, `的麻`, ...), which looks like the stub Enricher, so it has no tags
   such as food or restaurant that a cue could match.
4. Polluting notes: `吃什么` was stored as a note and became a seed for its own
   text, and `#all` was stored as a note though it is a reserved find cue.

**Directions to try** (proposals, check each against this case and the
evaluation): query expansion of intent questions in the QueryInterpreter
(HyDE-style: write the kind of note that would answer, embed that); Enricher
tags for entity type (food, place) so a cue can match by type; a recency term
in the Spreader or ranking; and checking whether a write should refuse or flag
text that is only a reserved cue like `#all`.
