#!/usr/bin/env python3
"""Load stored intermediates, verify summaries, and write paper figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_helper import (
    load_and_verify_empirical,
    load_and_verify_task1,
    load_and_verify_task2,
    load_and_verify_thm,
)
from scripts.paths import figures_dir
from scripts.plot_helper import (
    plot_empirical_mode,
    plot_task1_tree,
    plot_task1_visualization,
    plot_task2,
    plot_teaser,
    plot_thm,
    write_task1_table,
)
from scripts.teaser_helper import load_teaser


def main() -> None:
    """Verify cached summaries and reproduce every paper figure and table."""
    out = figures_dir()
    teaser = load_teaser()
    print("teaser", plot_teaser(teaser, out))

    thm_df, thm_summary = load_and_verify_thm()
    print("verified thm intermediates -> summary")
    print("thm", plot_thm(thm_df, thm_summary, out))

    emp_summary, emp_width = load_and_verify_empirical()
    print("verified empirical-mode intermediates -> summaries")
    print("empirical", plot_empirical_mode(emp_summary, emp_width, out))

    task1 = load_and_verify_task1()
    print("verified Task 1 Monte Carlo intermediates -> summary")
    print("task1 vis", plot_task1_visualization(task1["univariate"], out))
    print("task1 tree", plot_task1_tree(task1["tree"], out))
    print("task1 table", write_task1_table(task1["mc_summary"], out))

    task2 = load_and_verify_task2()
    print("verified Task 2 long.csv intermediates -> summaries")
    print("task2", plot_task2(task2, out))


if __name__ == "__main__":
    main()
