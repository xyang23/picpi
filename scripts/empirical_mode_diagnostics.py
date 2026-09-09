#!/usr/bin/env python3
"""Rerun the empirical-mode diagnostics GPU experiment.

Requires CUDA. Cached paper CSVs already live in cached_results/empirical_mode/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picpi.paths import cached


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute empirical-mode diagnostics results."
    )
    parser.add_argument("--output-dir", type=Path, default=cached("empirical_mode"))
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=1_000_000)
    parser.add_argument(
        "--n-calib-grid",
        nargs="+",
        type=int,
        default=[1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000],
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_calib = [1000, 3000] if args.smoke else args.n_calib_grid
    reps = 2 if args.smoke else args.reps
    n_eval = 10_000 if args.smoke else args.n_eval
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("empirical_mode_diagnostics_helper.py")),
        "--gpu-ids",
        args.gpu_ids,
        "--output-dir",
        str(args.output_dir),
        "--reps",
        str(reps),
        "--n-eval",
        str(n_eval),
        "--n-calib-grid",
        *[str(n) for n in n_calib],
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
