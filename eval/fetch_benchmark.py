"""Fetch a real, public prompt-injection benchmark and cache it locally.

The seven hand-built scenarios in `data/sessions/` prove the *mechanism* — the
trajectory failures a stateless checker cannot see. They cannot prove *accuracy*,
because we wrote them. A reviewer is right to ask "did you tune the detector to
pass your own cases?". The only honest answer is a corpus we did NOT write.

This pulls `deepset/prompt-injections` (546 labelled prompts, benign vs
injection, multilingual) from the Hugging Face datasets-server as JSON — no
`datasets` library, no parquet, just the public rows API — and caches it to
`data/benchmark/prompt_injections.jsonl` so `eval/benchmark.py` then runs fully
offline and reproducibly.

    python -m eval.fetch_benchmark

Provenance: deepset/prompt-injections, CC-BY-4.0, https://huggingface.co/datasets/deepset/prompt-injections
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmark" / "prompt_injections.jsonl"

DATASET = "deepset/prompt-injections"
CONFIG = "default"
SPLIT = "train"
PAGE = 100  # datasets-server caps `length` at 100
BASE = "https://datasets-server.huggingface.co/rows"


def _fetch_page(offset: int) -> dict:
    q = urllib.parse.urlencode({
        "dataset": DATASET, "config": CONFIG, "split": SPLIT,
        "offset": offset, "length": PAGE,
    })
    req = urllib.request.Request(f"{BASE}?{q}", headers={"User-Agent": "preflight-bench"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> None:
    first = _fetch_page(0)
    if "error" in first:
        print(f"[fetch] dataset error: {first['error']}", file=sys.stderr)
        sys.exit(1)
    total = first.get("num_rows_total", 0)
    print(f"[fetch] {DATASET} split={SPLIT}: {total} rows")

    rows: list[dict] = []
    offset = 0
    while offset < total:
        page = first if offset == 0 else _fetch_page(offset)
        for r in page["rows"]:
            row = r["row"]
            rows.append({
                "text": row["text"],
                # 1 = injection/attack, 0 = benign, per the dataset's schema.
                "label": "attack" if int(row["label"]) == 1 else "benign",
                "source": DATASET,
            })
        offset += PAGE
        print(f"[fetch] {min(offset, total)}/{total}")
        if offset < total:
            time.sleep(0.3)  # be polite to the public API

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_attack = sum(1 for r in rows if r["label"] == "attack")
    print(f"[fetch] wrote {len(rows)} rows -> {OUT.relative_to(ROOT)} "
          f"({n_attack} attack / {len(rows) - n_attack} benign)")


if __name__ == "__main__":
    main()
