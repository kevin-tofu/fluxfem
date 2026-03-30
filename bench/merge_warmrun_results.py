#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge warm-run result JSON files into a single payload.")
    p.add_argument("--input", action="append", required=True, help="Input JSON file. May be passed multiple times.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    inputs = [(ROOT / item).resolve() for item in args.input]
    payloads = [_load(path) for path in inputs]
    if not payloads:
        raise SystemExit("no inputs")
    out = {
        "mode": payloads[0].get("mode"),
        "config": payloads[0].get("config"),
        "results": [],
    }
    for payload in payloads:
        out["results"].extend(payload.get("results", []))
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
