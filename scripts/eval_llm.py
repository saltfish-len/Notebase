"""Is Haiku enough? Compare models on the LLM Atomizer + Enricher.

Runs a fixed set of captures (English, Chinese, mixed brain dumps, single
thoughts, references to resolve) through both boxes for each model, then
prints automatic checks, tokens, latency and cost, and saves every output
to JSON for side-by-side reading.

    ANTHROPIC_API_KEY=... python scripts/eval_llm.py   # or CLAUDE_API_KEY
    python scripts/eval_llm.py --models claude-haiku-5-5 claude-sonnet-5-5 --out eval-out

Costs very little: about 20 calls per model, a few thousand tokens each.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from notebase.blackbox.llm import LLMAtomizer, LLMEnricher, build_client
from notebase.models import Capture

# $ per million tokens (input, output), first-party API, prompts <= 100K tokens
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass
class Case:
    name: str
    text: str
    notes: tuple[int, int]  # acceptable range of atomic notes
    must_keep: list[str] = field(default_factory=list)  # facts that must survive verbatim
    must_resolve: list[str] = field(default_factory=list)  # should appear after resolving refs
    language: str = "en"


CASES = [
    Case("single-thought", "Deep sleep is when the hippocampus replays the day's memories.",
         (1, 1), ["hippocampus"]),
    Case("brain-dump-en", """Met Alice from the design team today. She thinks the onboarding flow is
too long, we should cut it to 3 steps. Also: book flights to Tokyo before Nov 20, prices jump
after that. Random thought - spaced repetition might work for learning people's names.""",
         (3, 5), ["Alice", "3 steps", "Nov 20", "Tokyo"]),
    Case("references", """Read "Why We Sleep" by Matthew Walker. He argues that even one night of
short sleep hurts memory. It also says caffeine has a half-life of about 6 hours, which is why
afternoon coffee matters.""",
         (2, 4), ["Matthew Walker", "6 hours"], must_resolve=["Why We Sleep|Walker", "caffeine|coffee"]),
    Case("brain-dump-zh", """今天和老王聊了新项目，他觉得用MongoDB做向量检索没问题。
另外要记得周五之前交季度预算。
突然想到：笔记其实就是记忆的外挂，按文件夹存反而违背人的联想方式。""",
         (3, 3), ["老王", "MongoDB", "周五", "季度预算"], language="zh"),
    Case("mixed-language", """读了一篇关于 spaced repetition 的文章，间隔重复对记单词特别有效。
Should try Anki for the Japanese vocab before the Tokyo trip.""",
         (2, 2), ["Anki", "Tokyo"], language="mixed"),
    Case("one-idea-many-sentences", """The reason my sourdough failed: the starter wasn't active
enough. It had been in the fridge for two weeks. Next time feed it twice the day before baking
and wait until it doubles.""",
         (1, 2), ["two weeks", "doubles"]),
    Case("list-with-hashtags", "#reading 1) Thinking, Fast and Slow 2) The Mind Map Book 3) Make It Stick",
         (1, 3), ["Thinking, Fast and Slow", "Make It Stick"]),
    Case("url", "Great explainer on vector search https://www.mongodb.com/docs/atlas/atlas-vector-search/ - read the part about numCandidates",
         (1, 1), ["https://www.mongodb.com/docs/atlas/atlas-vector-search/", "numCandidates"]),
]


class RecordingClient:
    """Wraps the real client to record usage and latency of every call."""

    def __init__(self, client):
        self._client = client
        self.messages = self
        self.records = []

    def parse(self, **kwargs):
        t = time.monotonic()
        response = self._client.messages.parse(**kwargs)
        self.records.append(
            {
                "latency_s": time.monotonic() - t,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason,
            }
        )
        return response


def _cjk_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z㐀-鿿]", text)
    return sum(1 for c in letters if "㐀" <= c <= "鿿") / max(1, len(letters))


def check(case: Case, notes: list[str], enrichments) -> dict:
    joined = "\n".join(notes)
    result = {
        "note_count_ok": case.notes[0] <= len(notes) <= case.notes[1],
        "facts_kept": all(f in joined for f in case.must_keep),
        "refs_resolved": all(
            sum(bool(re.search(alt, n, re.I)) for n in notes) >= min(2, len(notes))
            for alt in case.must_resolve
        ) if case.must_resolve else True,
        "has_tags": all(e.tags for e in enrichments),
        "keywords_5_10": all(5 <= len(e.keywords) <= 12 for e in enrichments),
    }
    if case.language == "zh":
        result["language_kept"] = _cjk_ratio(joined) > 0.5
        result["cross_lingual_keywords"] = all(
            any(re.search(r"[A-Za-z]", k) for k in e.keywords) for e in enrichments
        )
    elif case.language == "en":
        result["language_kept"] = _cjk_ratio(joined) < 0.1
        result["cross_lingual_keywords"] = all(
            any(re.search(r"[㐀-鿿]", k) for k in e.keywords) for e in enrichments
        )
    return result


def run_model(model: str, out_dir: Path) -> dict:
    client = RecordingClient(build_client())
    vocabulary: list[str] = []  # grows across cases, like a real store
    atomizer = LLMAtomizer(client, model=model)
    enricher = LLMEnricher(client, model=model, tag_vocabulary=lambda: list(vocabulary))

    rows = []
    for case in CASES:
        notes = atomizer.atomize(Capture(case.text))
        enrichments = enricher.enrich(notes)
        new_tags = {t for e in enrichments for t in e.tags} - set(vocabulary)
        reused = sum(t in vocabulary for e in enrichments for t in e.tags)
        vocabulary.extend(sorted(new_tags))
        rows.append(
            {
                "case": case.name,
                "notes": notes,
                "enrichments": [asdict(e) for e in enrichments],
                "checks": check(case, notes, enrichments),
                "tags_reused": reused,
            }
        )

    calls = client.records
    tokens_in = sum(r["input_tokens"] for r in calls)
    tokens_out = sum(r["output_tokens"] for r in calls)
    price_in, price_out = PRICES.get(model, (float("nan"), float("nan")))
    summary = {
        "model": model,
        "checks_passed": sum(all(r["checks"].values()) for r in rows),
        "cases": len(rows),
        "failed_checks": {
            r["case"]: [k for k, v in r["checks"].items() if not v]
            for r in rows
            if not all(r["checks"].values())
        },
        "fallbacks": 2 * len(CASES) - len(calls)
        + sum(r["stop_reason"] != "end_turn" for r in calls),
        "tag_vocabulary": vocabulary,
        "latency_p50_s": sorted(r["latency_s"] for r in calls)[len(calls) // 2] if calls else None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out,
    }
    summary["cost_per_1000_writes_usd"] = summary["cost_usd"] / len(CASES) * 1000
    (out_dir / f"{model}.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["claude-haiku-5-5", "claude-sonnet-5-5"])
    parser.add_argument("--out", default="eval-out")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        s = run_model(model, out_dir)
        print(f"\n== {model}")
        print(f"  all checks passed: {s['checks_passed']}/{s['cases']} cases")
        for case, failed in s["failed_checks"].items():
            print(f"    ✗ {case}: {', '.join(failed)}")
        print(f"  fallbacks to stub: {s['fallbacks']}")
        print(f"  tag vocabulary ({len(s['tag_vocabulary'])}): {', '.join(s['tag_vocabulary'])}")
        print(f"  latency p50: {s['latency_p50_s']:.1f}s per call")
        print(
            f"  tokens: {s['tokens_in']} in / {s['tokens_out']} out"
            f"  → ${s['cost_usd']:.4f} total, ≈ ${s['cost_per_1000_writes_usd']:.2f} per 1000 writes"
        )
    print(f"\nFull outputs for side-by-side reading: {out_dir}/<model>.json")


if __name__ == "__main__":
    main()
