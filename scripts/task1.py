#!/usr/bin/env python3
"""Compute and store Task 1 visualization, tree, and Monte Carlo table results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picpi.paths import cached
from scripts.task1_helper import (
    N_MC_TABLE,
    compute_misspecified_tree,
    compute_multivariate_table,
    compute_univariate_visualization,
    save_task1,
)


def main() -> None:
    """Parse options, run the Task 1 experiments, and save their results."""
    parser = argparse.ArgumentParser(description="Compute Task 1 paper results.")
    parser.add_argument("--output-dir", type=Path, default=cached("task1"))
    parser.add_argument("--n-mc", type=int, default=N_MC_TABLE)
    parser.add_argument(
        "--skip-mc",
        action="store_true",
        help="Skip the 100-replication table (keep existing CSV if present).",
    )
    args = parser.parse_args()

    print("Computing univariate visualization...")
    univariate = compute_univariate_visualization()
    print("Computing misspecified decision-tree illustration...")
    tree = compute_misspecified_tree()

    mc_path = args.output_dir / "multivariate_mc_reps.csv"
    if args.skip_mc and mc_path.exists():
        import pandas as pd

        print(f"Reusing cached Monte Carlo table at {mc_path}")
        mc_results = pd.read_csv(mc_path)
    else:
        print(f"Computing multivariate Monte Carlo table ({args.n_mc} replications)...")
        mc_results = compute_multivariate_table(n_mc=args.n_mc)

    out = save_task1(univariate, tree, mc_results, args.output_dir)
    print(f"Saved Task 1 results to {out}")


if __name__ == "__main__":
    main()
