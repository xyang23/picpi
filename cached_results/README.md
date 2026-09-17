# Stored intermediates

These files are the numerical inputs for Stage 2. GPU jobs are not rerun.

| Experiment | Hardware used originally | Intermediate files used to rebuild figures |
|---|---|---|
| Teaser | CPU | `teaser/scatter.npz`, `teaser/intervals.csv` |
| Task 1 | CPU | `task1/univariate_visualization.npz`, `task1/misspecified_tree.npz`, `task1/multivariate_mc_reps.csv` (includes midpoint ECE and mean-score ECE) |
| Task 2 | CPU | `task2/*/long.csv` (1,000 seeds), plus `config.json` |
| Width shrinkage | GPU | `thm/thm_df_results.csv` (900 rows) |
| Empirical-mode diagnostics | GPU | `empirical_mode/heldout_picpi_results.csv`, `empirical_mode/heldout_picpi_interval_results.csv` |

The `*summary.csv` files are the original aggregates. Stage 2 rebuilds them from the intermediates and checks equality before plotting.
