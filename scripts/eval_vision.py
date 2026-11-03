"""Does the VisionNormalizer write alt text you could find an image by?

Builds a small image set: three synthetic images with text (a bilingual
whiteboard, a receipt, a terminal error screenshot) and four photos from
Wikimedia Commons, one converted to HEIC like an iPhone photo. File names are
camera-style (IMG_4471.jpg) so they carry no meaning. Each image goes through
the VisionNormalizer with or without typed text; the checks look for the words
a person would search for, the language, and the marker format.

    ANTHROPIC_API_KEY=... python scripts/eval_vision.py --runs 3
    python scripts/eval_vision.py --model claude-sonnet-5-5 --images eval-out/images

Costs about $0.001 per run on Haiku.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from notebase.blackbox.llm import VisionNormalizer, build_client
from notebase.models import Attachment, Capture

PRICES = {  # $ per million tokens (input, output)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/{}?width=960"
PHOTOS = {  # file name here -> Wikimedia Commons file
    "IMG_3301.jpg": "Cat_November_2010-1a.jpg",
    "IMG_1187.jpg": "Tour_Eiffel_Wikimedia_Commons.jpg",
    "IMG_2090.jpg": "Sushi_platter.jpg",
    "DSC00412.jpg": "Good_Food_Display_-_NCI_Visuals_Online.jpg",
}


@dataclass
class Case:
    name: str
    file: str
    text: str = ""
    must_mention: list[str] = field(default_factory=list)  # regexes, all must match
    language: str = "en"


CASES = [
    Case("whiteboard-zh", "IMG_4471.jpg", "今天周会的白板",
         ["导出|export", "分词", "12月15日|12/15", "小李"], language="zh"),
    Case("receipt", "IMG_4502.png", "",
         ["(?i)blue bottle", "1,?130", "[Ll]atte|拿铁"]),
    Case("error-screenshot", "Screenshot 2026-11-02 at 21.40.11.png", "docker 又起不来了",
         ["27017", "address already in use|端口|占用"], language="zh"),
    Case("cat-heic", "IMG_3301.HEIC", "", ["[Cc]at|tabby"]),
    Case("landmark", "IMG_1187.jpg", "Paris trip, day 2", ["Eiffel"]),
    Case("food-photo", "IMG_2090.jpg", "", ["[Ss]ushi"]),
    Case("food-display-zh", "DSC00412.jpg", "健康饮食的参考图", ["水果|蔬菜|食物|食品|食材"], language="zh"),
]


def build_images(out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    out.mkdir(parents=True, exist_ok=True)
    font_path = next(
        (p for p in [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ] if Path(p).exists()),
        None,
    )
    if font_path is None:
        raise SystemExit("need a CJK font for the synthetic images")
    big, small = ImageFont.truetype(font_path, 40), ImageFont.truetype(font_path, 26)

    im = Image.new("RGB", (1400, 900), (245, 245, 240))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 1380, 880], outline=(120, 120, 120), width=8)
    for i, line in enumerate(["Q4 路线图 Roadmap", "1. 砍掉导出功能 (cut export)", "2. 搜索支持中文分词",
                              "3. Web UI 原型 - 12月15日", "负责人: 小李 / Alice"]):
        d.text((80, 80 + i * 140), line, fill=(180, 20, 20) if i == 0 else (20, 40, 160), font=big)
    im.save(out / "IMG_4471.jpg", quality=88)

    im = Image.new("RGB", (600, 1000), "white")
    d = ImageDraw.Draw(im)
    for i, line in enumerate(["BLUE BOTTLE COFFEE", "Kyoto Rokkaku", "2026-10-28 09:14", "",
                              "Latte           ¥650", "Cinnamon roll   ¥480", "",
                              "TOTAL          ¥1,130", "VISA ****4421"]):
        d.text((50, 60 + i * 80), line, fill="black", font=small)
    im.save(out / "IMG_4502.png")

    im = Image.new("RGB", (1200, 500), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for i, line in enumerate(["$ docker compose up -d",
                              "Error response from daemon: Ports are not available:",
                              "exposing port TCP 0.0.0.0:27017 -> bind: address already in use"]):
        d.text((30, 40 + i * 60), line, fill=(220, 220, 220) if i == 0 else (255, 110, 110), font=small)
    im.save(out / "Screenshot 2026-11-02 at 21.40.11.png")

    for name, commons in PHOTOS.items():
        if not (out / name).exists():
            req = urllib.request.Request(COMMONS.format(commons), headers={"User-Agent": "notebase-eval/0.1"})
            with urllib.request.urlopen(req) as r, open(out / name, "wb") as f:
                shutil.copyfileobj(r, f)
    import pillow_heif

    pillow_heif.register_heif_opener()
    Image.open(out / "IMG_3301.jpg").save(out / "IMG_3301.HEIC", format="HEIF")


class RecordingClient:
    def __init__(self, client):
        self._client = client
        self.messages = self
        self.records = []

    def parse(self, **kwargs):
        t = time.monotonic()
        response = self._client.messages.parse(**kwargs)
        self.records.append({
            "latency_s": time.monotonic() - t,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })
        return response


def _cjk_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z㐀-鿿]", text)
    return sum(1 for c in letters if "㐀" <= c <= "鿿") / max(1, len(letters))


def _prose(text: str) -> str:
    """The description without what it quotes from the image."""
    return re.sub(r"[\"“'‘「][^\"”'’」]*[\"”'’」]", " ", text)


def check(case: Case, alt: str | None) -> dict:
    if alt is None:
        return {"described": False}
    return {
        "described": True,
        "mentions": all(re.search(p, alt) for p in case.must_mention),
        # judged on the prose: quotes from the image keep their own language
        "language": (_cjk_ratio(_prose(alt)) > 0.4 if case.language == "zh"
                     else _cjk_ratio(_prose(alt)) < 0.1),
        # the typed text is kept next to the alt text; repeating it adds nothing
        "no_echo": not case.text or case.text.lower() not in alt.lower(),
        "no_preamble": not re.match(r"(this|an?) (image|photo|picture)|图片显示|这张", alt, re.I),
        "length_ok": 20 <= len(alt) <= 600,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--images", default="eval-out/images")
    parser.add_argument("--out", default="eval-out")
    args = parser.parse_args()

    images = Path(args.images)
    if not all((images / c.file).exists() for c in CASES):
        build_images(images)
    client = RecordingClient(build_client())
    normalizer = VisionNormalizer(client, model=args.model, effort=args.effort)

    rows, passed = [], 0
    for run in range(args.runs):
        for case in CASES:
            attachment = Attachment.from_path(images / case.file)
            try:
                alt = normalizer.describe(attachment, case.text)
            except Exception as e:  # noqa: BLE001 - recorded as a failed case
                alt, error = None, repr(e)
            else:
                error = None
            checks = check(case, alt)
            ok = all(checks.values())
            passed += ok
            rows.append({"run": run, "case": case.name, "alt": alt, "error": error, "checks": checks})
            failed = [k for k, v in checks.items() if not v]
            print(f"[{run}] {case.name:18} {'ok ' if ok else 'FAIL ' + ','.join(failed)}  {alt}")

    calls = client.records
    tin, tout = sum(r["input_tokens"] for r in calls), sum(r["output_tokens"] for r in calls)
    pin, pout = PRICES.get(args.model, (float("nan"), float("nan")))
    cost = (tin * pin + tout * pout) / 1e6
    lat = sorted(r["latency_s"] for r in calls)
    summary = {
        "model": args.model, "effort": args.effort, "passed": passed, "cases": len(rows),
        "p50_latency_s": lat[len(lat) // 2] if lat else None,
        "p90_latency_s": lat[int(len(lat) * 0.9)] if lat else None,
        "input_tokens_per_image": tin / max(1, len(calls)),
        "output_tokens_per_image": tout / max(1, len(calls)),
        "cost_per_1000_images_usd": cost / max(1, len(calls)) * 1000,
    }
    print(json.dumps(summary, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"vision_{args.model}_{args.effort}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
