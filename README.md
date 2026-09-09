# PICPI paper reproduction

This folder reproduces the figures and tables in the PICPI paper from stored numerical results. A GPU is not required.

There are two stages:

1. **Compute and store results** into `cached_results/`
2. **Load those results and draw the paper figures/tables** into `figures/`

The expensive Monte Carlo and GPU intermediates are already included. The default reproduction path only loads those files.

## Environment setup (`uv`)

Install [`uv`](https://docs.astral.sh/uv/) if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from this folder:

```bash
cd picpi_reproducibility
uv sync
```

This creates `.venv` from `pyproject.toml` (Python >= 3.10, plus numpy, pandas, matplotlib, scipy, scikit-learn, tqdm, and Jupyter). All commands below use `uv run`.

Optional Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name picpi-reproducibility
```

## Reproduce the paper figures and table

Default path: load the stored intermediates, rebuild the summaries, check they match the cached summaries, and write the paper filenames into `figures/`.

```bash
cd picpi_reproducibility
uv sync
uv run python compute/plot_all.py
```

The same steps can be run interactively:

```bash
uv run jupyter notebook notebooks/02_reproduce_figures.ipynb
```

Outputs:

```text
figures/picpi_teaser_green.pdf
figures/thm52_width_mean_boxplot.pdf
figures/thm52_width_vs_rate.pdf
figures/heldout_picpi_failure_rate_by_method.pdf
figures/heldout_picpi_pass_rate_by_interval_length.pdf
figures/task1_interval_visualization.pdf
figures/task1_misspecified_point_estimator_decision_tree.pdf
figures/task1_multivariate_mc_summary.tex
figures/task2_multiclass_dgp_sweep.pdf
```

## Optional: recompute cheap CPU results

These two jobs run on CPU. They overwrite `cached_results/teaser/` and `cached_results/task1/`.

```bash
uv run python compute/teaser.py
uv run python compute/task1.py
```

`compute/task1.py` takes about two minutes for the 100-replication table. A smoke test:

```bash
uv run python compute/task1.py --n-mc 2
```

The same cheap jobs are also in `notebooks/01_compute_results.ipynb`:

```bash
uv run jupyter notebook notebooks/01_compute_results.ipynb
```

After recomputing, redraw the figures:

```bash
uv run python compute/plot_all.py
```

Do **not** rerun the GPU scripts to reproduce the paper. Width-shrinkage and empirical-mode diagnostics already have their per-replication CSVs in `cached_results/thm52/` and `cached_results/empirical_mode/`. Task 2 was a CPU sweep; its per-seed `long.csv` files are already in `cached_results/task2/`.

## Figure and table mapping

| Paper object | Paper file | This folder, compute | This folder, plot | Stored result |
|---|---|---|---|---|
| Teaser (`fig:teaser`) | `Figures/picpi_teaser_green.pdf` | `compute/teaser.py` | `picpi/plot.py::plot_teaser` | `cached_results/teaser/` |
| Width shrinkage (`fig:thm52_box_rate`) | `Figures/thm52_width_mean_boxplot.pdf`, `Figures/thm52_width_vs_rate.pdf` | cached GPU CSV, no rerun | `picpi/plot.py::plot_thm52` | `cached_results/thm52/thm52_df_results.csv` |
| Empirical-mode diagnostics (`fig:emp_mode_diagnostics`) | `Figures/heldout_picpi_failure_rate_by_method.pdf`, `Figures/heldout_picpi_pass_rate_by_interval_length.pdf` | cached GPU CSV, no rerun | `picpi/plot.py::plot_empirical_mode` | `cached_results/empirical_mode/heldout_picpi_*.csv` |
| Task 1 univariate (`fig:task1_interval_visualization`) | `artifacts/task1_interval_visualization.pdf` | `compute/task1.py` | `picpi/plot.py::plot_task1_visualization` | `cached_results/task1/univariate_visualization.npz` |
| Task 1 misspecified tree (`fig:task1_misspecified_tree`) | `artifacts/task1_misspecified_point_estimator_decision_tree.pdf` | `compute/task1.py` | `picpi/plot.py::plot_task1_tree` | `cached_results/task1/misspecified_tree.npz` |
| Task 1 table (`tab:task1_multivariate_mc_summary`) | `artifacts/task1_multivariate_mc_summary.tex` | `compute/task1.py` | `picpi/plot.py::write_task1_table` | `cached_results/task1/multivariate_mc_reps.csv` |
| Task 2 (`fig:task2`) | `Figures/task2_multiclass_dgp_sweep.pdf` | cached CPU `long.csv`, no rerun | `picpi/plot.py::plot_task2` | `cached_results/task2/*/long.csv` |

PICPI interval construction is `picpi/calibration.py`.

## Folder layout

```text
picpi/                 PICPI algorithm, Task 1/teaser compute, aggregators, plotters
compute/               CLIs
vendor/                original Task 2 and GPU runners (record only)
notebooks/
  01_compute_results.ipynb
  02_reproduce_figures.ipynb
cached_results/        stored numerical outputs
figures/               regenerated paper PDFs and the Task 1 table
```
