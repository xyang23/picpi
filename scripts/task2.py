#!/usr/bin/env python3
"""Rerun the Task 2 multiclass DGP sweep and store summary CSVs.

Paper settings: 1000 replications, n_train=n_cal=n_eval=2000, 100 bins.
This is expensive (hours). A smoke run uses --smoke.
Cached paper results already live in cached_results/task2/.
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
    parser = argparse.ArgumentParser(description="Compute Task 2 DGP-sweep results.")
    parser.add_argument("--output-dir", type=Path, default=cached("task2"))
    parser.add_argument("--mc-reps", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DGP result folders.",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("task2_helper.py")),
        "--mc-reps",
        str(2 if args.smoke else args.mc_reps),
        "--seed",
        "17",
        "--n-train",
        "2000",
        "--n-cal",
        "2000",
        "--n-eval",
        "2000",
        "--num-bins-picpi",
        "100",
        "--baseline-num-bins",
        "100",
        "--n-workers",
        str(args.n_workers),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.smoke:
        cmd.append("--smoke")
    if args.overwrite:
        cmd.append("--overwrite")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
