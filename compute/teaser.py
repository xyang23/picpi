#!/usr/bin/env python3
"""Compute and store teaser-figure results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picpi.paths import cached
from picpi.teaser import compute_teaser, save_teaser


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute teaser figure inputs.")
    parser.add_argument("--output-dir", type=Path, default=cached("teaser"))
    args = parser.parse_args()
    payload = compute_teaser()
    out = save_teaser(payload, args.output_dir)
    print(f"Saved teaser results to {out}")


if __name__ == "__main__":
    main()
