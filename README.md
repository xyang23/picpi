# PICPI 

This folder implements the experiments in the PICPI paper.

1. **Reproduce the paper figures/tables** into `figures/` from stored results
2. **(Optional) Compute and store results** into `cached_results/`


## Environment setup (`uv`)

Install [`uv`](https://docs.astral.sh/uv/) if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from this folder:

```bash
uv sync
```

This creates `.venv` from `pyproject.toml` (Python >= 3.10, plus numpy, pandas, matplotlib, scipy, scikit-learn, and tqdm). All commands below use `uv run`.

## PICPI usage example

The commented example in `scripts/example_use.py` demonstrates how to construct population- and empirical-mode PICPIs from a fitted model and a held-out calibration sample. It also shows how to obtain the applicable interval or intervals for new test inputs.

Run it with:

```bash
uv run python scripts/example_use.py
```

The example DGP and plotting details are isolated in `scripts/example_use_helper.py`. The script prints all constructed PICPIs, demonstrates test-input inference using the shortest matching population-mode interval, and saves empirical- and population-mode figures under `figures/example_use/`.

## Reproduce the paper figures and table

Default path: load the stored intermediates, rebuild the summaries, check they match the cached summaries, and write the paper filenames into `figures/`.

```bash
uv sync
uv run python scripts/plot_all.py
```

Outputs:

```text
figures/picpi_teaser_green.pdf
figures/thm_width_mean_boxplot.pdf
figures/thm_width_vs_rate.pdf
figures/heldout_picpi_failure_rate_by_method.pdf
figures/heldout_picpi_pass_rate_by_interval_length.pdf
figures/task1_interval_visualization.pdf
figures/task1_misspecified_point_estimator_decision_tree.pdf
figures/task1_multivariate_mc_summary.tex
figures/task2_multiclass_dgp_sweep.pdf
```

## Optional: recompute results

All precomputed results required to reproduce the figures and table are provided under `cached_results/`. The following experiments can be rerun to regenerate those results.

### Intro figure (CPU)

Generates the calibration intervals and predicted-versus-true probability data used in the introductory figure.

```bash
uv run python scripts/teaser.py
```

Results are written to `cached_results/teaser/` by default.

### Task 1: probability intervals (CPU)

Computes the univariate method comparison, the misspecified decision-tree example, and the 100-replication multivariate Monte Carlo table.

```bash
uv run python scripts/task1.py
```

For a reduced Monte Carlo run that leaves the stored results unchanged:

```bash
uv run python scripts/task1.py --n-mc 2 --output-dir /tmp/picpi-task1-smoke
```

### Task 2: multiclass label sets (CPU)

Compares label-set methods across target coverage levels under linear-Gaussian and nonlinear-mixture data-generating processes. Pass `--overwrite` to replace the stored Task 2 results.

```bash
uv run python scripts/task2.py --overwrite
```

For a short verification run that leaves the stored results unchanged:

```bash
uv run python scripts/task2.py --smoke --output-dir /tmp/picpi-task2-smoke
```

### Width shrinkage (GPU)

Evaluates how PICPI interval width changes as the calibration sample size increases. This experiment requires a CUDA GPU.

```bash
uv sync --extra gpu
uv run python scripts/thm_width.py
```

For a reduced run that leaves the stored results unchanged:

```bash
uv run python scripts/thm_width.py --smoke --output-dir /tmp/picpi-thm-width-smoke
```

### Empirical-mode diagnostics (GPU)

Compares empirical- and population-mode PICPI behavior across calibration sample sizes, including held-out property checks and interval widths. This experiment requires a CUDA GPU.

```bash
uv sync --extra gpu
uv run python scripts/empirical_mode_diagnostics.py
```

For a reduced run that leaves the stored results unchanged:

```bash
uv run python scripts/empirical_mode_diagnostics.py --smoke --output-dir /tmp/picpi-empirical-mode-diagnostics-smoke
```

For either GPU experiment, use `--gpu-ids` to select one or more CUDA devices. The default is GPU 0; for example, `--gpu-ids 0,1` uses GPUs 0 and 1.

After recomputing, redraw the figures:

```bash
uv run python scripts/plot_all.py
```

## Figure and table mapping

| Paper object | Paper file | Computation entry point | Plotting function | Stored result |
|---|---|---|---|---|
| Teaser (`fig:teaser`) | `Figures/picpi_teaser_green.pdf` | `scripts/teaser.py` | `scripts/plot_helper.py::plot_teaser` | `cached_results/teaser/` |
| Width shrinkage (`fig:thm_box_rate`) | `Figures/thm_width_mean_boxplot.pdf`, `Figures/thm_width_vs_rate.pdf` | `scripts/thm_width.py` (CUDA, optional) | `scripts/plot_helper.py::plot_thm` | `cached_results/thm/thm_df_results.csv` |
| Empirical-mode diagnostics (`fig:emp_mode_diagnostics`) | `Figures/heldout_picpi_failure_rate_by_method.pdf`, `Figures/heldout_picpi_pass_rate_by_interval_length.pdf` | `scripts/empirical_mode_diagnostics.py` (CUDA, optional) | `scripts/plot_helper.py::plot_empirical_mode` | `cached_results/empirical_mode/heldout_picpi_*.csv` |
| Task 1 univariate (`fig:task1_interval_visualization`) | `artifacts/task1_interval_visualization.pdf` | `scripts/task1.py` | `scripts/plot_helper.py::plot_task1_visualization` | `cached_results/task1/univariate_visualization.npz` |
| Task 1 misspecified tree (`fig:task1_misspecified_tree`) | `artifacts/task1_misspecified_point_estimator_decision_tree.pdf` | `scripts/task1.py` | `scripts/plot_helper.py::plot_task1_tree` | `cached_results/task1/misspecified_tree.npz` |
| Task 1 table (`tab:task1_multivariate_mc_summary`) | `artifacts/task1_multivariate_mc_summary.tex` | `scripts/task1.py` | `scripts/plot_helper.py::write_task1_table` | `cached_results/task1/multivariate_mc_reps.csv` |
| Task 2 (`fig:task2`) | `Figures/task2_multiclass_dgp_sweep.pdf` | `scripts/task2.py` (CPU, optional) | `scripts/plot_helper.py::plot_task2` | `cached_results/task2/*/long.csv` |

PICPI interval construction is `picpi/calibration.py`.

## Folder layout

Experiment files in `scripts/` follow a wrapper/helper convention: the shorter file, such as `task1.py`, is the user-facing command, while the corresponding `_helper.py` file contains the full implementation.

```text
picpi/                            Reusable PICPI core
  calibration.py                 PICPI interval-construction algorithm
  inference.py                   Assign test predictions to calibrated intervals

scripts/                          Paper-specific computation and plotting workflows
  paths.py                        Shared repository and output paths
  example_use.py                  Commented empirical/population PICPI usage example
  example_use_helper.py           Teaser DGP and example plotting functions
  teaser.py                       Teaser command
  teaser_helper.py                Teaser data generation and storage
  task1.py                        Task 1 command
  task1_helper.py                 Task 1 simulations and result storage
  task2.py                        Task 2 command
  task2_helper.py                 Task 2 multiclass DGP-sweep implementation
  thm_width.py                    Width-shrinkage GPU command
  thm_width_helper.py             Full width-shrinkage GPU implementation
  empirical_mode_diagnostics.py  Empirical-mode diagnostics GPU command
  empirical_mode_diagnostics_helper.py
                                  Full empirical-mode GPU implementation
  aggregate_helper.py             Summary rebuilding and cached-result verification
  plot_all.py                     Regenerate paper output
  plot_helper.py                  Figure and table generation functions

cached_results/                   Precomputed experiment outputs and summaries
  teaser/                         Teaser inputs
  task1/                          Task 1 simulation results
  task2/                          Task 2 per-seed results and summaries
  thm/                            Width-shrinkage results
  empirical_mode/                 Empirical-mode diagnostics results

figures/                          Generated PDF/PNG figures and Task 1 table files
```
