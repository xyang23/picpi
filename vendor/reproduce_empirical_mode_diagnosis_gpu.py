#!/usr/bin/env python3
"""GPU reproduction of empirical_mode_diagnosis_figures.ipynb.

The notebook repeatedly materializes 20-dimensional Gaussian samples and then
rescans them for every candidate interval.  This script preserves the same DGP
and interval definitions while using two exact reductions:

1. Downstream quantities depend on X only through the fitted and true linear
   predictors, so those two jointly Gaussian projections are sampled directly.
2. All interval endpoints lie on one grid per task, so samples are streamed
   into per-bin sufficient statistics and discarded.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

DEFAULT_N_CALIB_GRID = (
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
)
MULTIVARIATE_DIM = 20
MULTIVARIATE_BETA0 = 0.0
MULTIVARIATE_BETA = np.concatenate(
    [np.array([1.6, -1.1, 0.9], dtype=np.float64), np.zeros(MULTIVARIATE_DIM - 3)]
)
MULTIVARIATE_EPSILON = 0.10
GAUSS_HERMITE_ORDER = 40
GAUSS_HERMITE_TILE = 8


@dataclass(frozen=True)
class Config:
    n_train: int
    n_eval: int
    reps: int
    num_bin: int
    delta: float
    seed: int
    chunk_size: int


@dataclass(frozen=True)
class Task:
    n_idx: int
    n_calib: int
    num_bin: int
    rep: int
    seed: int


@dataclass(frozen=True)
class ProjectionParams:
    """Parameters for the joint fitted-score/true-score Gaussian reduction."""

    fitted_intercept: float
    fitted_scale: float
    true_intercept: float
    true_on_fitted_axis: float
    true_residual_scale: float


@dataclass
class BinnedStats:
    """Sufficient statistics on cells (edge[k], edge[k + 1]]."""

    counts: torch.Tensor
    successes: torch.Tensor
    p_true_sums: torch.Tensor | None
    edge_counts: torch.Tensor
    edge_successes: torch.Tensor


@dataclass(frozen=True)
class IntervalIndices:
    """Intervals represented by their integer endpoint indices."""

    left: torch.Tensor
    right: torch.Tensor

    def __len__(self) -> int:
        return int(self.left.numel())


def parse_gpu_ids(raw: str) -> list[int]:
    ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("--gpu-ids must contain at least one id")
    if any(gpu_id < 0 for gpu_id in ids):
        raise argparse.ArgumentTypeError("--gpu-ids must be non-negative")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("--gpu-ids must not contain duplicates")
    return ids


def adaptive_num_bin(n_calib: int, c_value: float) -> int:
    """Return K(n, C) using the schedule from reproduce_thm52_gpu_adaptive.py."""
    return max(1, int(round(c_value * (n_calib ** (1.0 / 3.0)))))


def make_edges(num_bin: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, num_bin + 1, device=device, dtype=torch.float64)


def empty_binned_stats(
    num_bin: int,
    device: torch.device,
    include_p_true: bool,
) -> BinnedStats:
    return BinnedStats(
        counts=torch.zeros(num_bin, device=device, dtype=torch.int64),
        successes=torch.zeros(num_bin, device=device, dtype=torch.float64),
        p_true_sums=(
            torch.zeros(num_bin, device=device, dtype=torch.float64)
            if include_p_true
            else None
        ),
        edge_counts=torch.zeros(num_bin + 1, device=device, dtype=torch.int64),
        edge_successes=torch.zeros(num_bin + 1, device=device, dtype=torch.float64),
    )


def accumulate_binned_stats(
    stats: BinnedStats,
    p_hat: torch.Tensor,
    y: torch.Tensor,
    edges: torch.Tensor,
    p_true: torch.Tensor | None = None,
) -> None:
    """Accumulate observations while preserving grid-boundary masses exactly."""
    num_bin = stats.counts.numel()
    in_base_cells = (p_hat > edges[0]) & (p_hat <= edges[-1])
    base_p = p_hat[in_base_cells]
    base_y = y[in_base_cells]
    base_idx = torch.bucketize(base_p, edges[1:], right=False)

    stats.counts.add_(torch.bincount(base_idx, minlength=num_bin))
    success_idx = base_idx[base_y > 0.5]
    stats.successes.add_(
        torch.bincount(success_idx, minlength=num_bin).to(torch.float64)
    )

    if stats.p_true_sums is not None:
        if p_true is None:
            raise ValueError("p_true is required for held-out statistics")
        stats.p_true_sums.add_(
            torch.bincount(
                base_idx,
                weights=p_true[in_base_cells],
                minlength=num_bin,
            )
        )

    edge_idx = torch.searchsorted(edges, p_hat, right=False)
    in_edge_range = edge_idx <= num_bin
    safe_edge_idx = edge_idx.clamp(max=num_bin)
    on_edge = in_edge_range & (p_hat == edges[safe_edge_idx])
    matched_edge_idx = edge_idx[on_edge]
    stats.edge_counts.add_(torch.bincount(matched_edge_idx, minlength=num_bin + 1))
    matched_success_idx = matched_edge_idx[y[on_edge] > 0.5]
    stats.edge_successes.add_(
        torch.bincount(matched_success_idx, minlength=num_bin + 1).to(torch.float64)
    )


def binned_stats_from_observations(
    p_hat: torch.Tensor,
    y: torch.Tensor,
    num_bin: int,
    p_true: torch.Tensor | None = None,
) -> BinnedStats:
    """Build sufficient statistics from already materialized test observations."""
    if p_hat.ndim != 1 or y.ndim != 1 or p_hat.numel() != y.numel():
        raise ValueError("p_hat and y must be one-dimensional and equally sized")
    if p_true is not None and (p_true.ndim != 1 or p_true.numel() != p_hat.numel()):
        raise ValueError("p_true must have the same one-dimensional shape")

    device = p_hat.device
    edges = make_edges(num_bin, device)
    stats = empty_binned_stats(num_bin, device, p_true is not None)
    accumulate_binned_stats(stats, p_hat, y, edges, p_true)
    return stats


def empirical_intervals_from_stats(stats: BinnedStats) -> IntervalIndices:
    """Reproduce empirical calibration's adaptive (left, right] scan."""
    num_bin = int(stats.counts.numel())
    counts_prefix = torch.cat(
        (
            torch.zeros(1, device=stats.counts.device, dtype=torch.float64),
            stats.counts.to(torch.float64).cumsum(0),
        )
    )
    successes_prefix = torch.cat(
        (
            torch.zeros(1, device=stats.counts.device, dtype=torch.float64),
            stats.successes.cumsum(0),
        )
    )

    accepted_left: list[int] = []
    accepted_right: list[int] = []
    left = 0
    for right in range(1, num_bin + 1):
        count = float((counts_prefix[right] - counts_prefix[left]).item())
        if count == 0.0:
            current_ok = False
        else:
            mean_y = float(
                ((successes_prefix[right] - successes_prefix[left]) / count).item()
            )
            current_ok = (left / num_bin) < mean_y <= (right / num_bin)

        tail_count = float((counts_prefix[num_bin] - counts_prefix[right]).item())
        if tail_count == 0.0:
            tail_ok = True
        else:
            tail_mean = float(
                (
                    (successes_prefix[num_bin] - successes_prefix[right]) / tail_count
                ).item()
            )
            tail_ok = (right / num_bin) < tail_mean <= 1.0

        if current_ok and tail_ok:
            accepted_left.append(left)
            accepted_right.append(right)
            left = right

    device = stats.counts.device
    return IntervalIndices(
        left=torch.tensor(accepted_left, device=device, dtype=torch.long),
        right=torch.tensor(accepted_right, device=device, dtype=torch.long),
    )


def population_intervals_from_stats(
    stats: BinnedStats,
    delta: float,
) -> IntervalIndices:
    """Vectorize population calibration over all closed grid intervals."""
    num_bin = int(stats.counts.numel())
    device = stats.counts.device
    counts_prefix = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.float64),
            stats.counts.to(torch.float64).cumsum(0),
        )
    )
    successes_prefix = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.float64),
            stats.successes.cumsum(0),
        )
    )

    left = torch.arange(num_bin, device=device, dtype=torch.long)[:, None]
    right = torch.arange(1, num_bin + 1, device=device, dtype=torch.long)[None, :]
    candidate = right > left

    counts = (
        counts_prefix[right]
        - counts_prefix[left]
        + stats.edge_counts[left].to(torch.float64)
    )
    successes = (
        successes_prefix[right] - successes_prefix[left] + stats.edge_successes[left]
    )
    nonempty = counts > 0
    mean_y = torch.zeros_like(counts)
    mean_y[nonempty] = successes[nonempty] / counts[nonempty]

    gap = torch.full_like(counts, math.inf)
    gap[nonempty] = 2.0 * torch.sqrt(math.log(num_bin**2 / delta) / counts[nonempty])
    lower = left.to(torch.float64) / num_bin
    upper = right.to(torch.float64) / num_bin
    valid = candidate & nonempty & ((lower + gap) <= mean_y) & (mean_y <= (upper - gap))
    valid_indices = valid.nonzero(as_tuple=False)
    return IntervalIndices(
        left=valid_indices[:, 0],
        right=valid_indices[:, 1] + 1,
    )


def intervals_to_numpy(
    intervals: IntervalIndices,
    num_bin: int,
) -> np.ndarray:
    if len(intervals) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack(
        (
            intervals.left.detach().cpu().numpy() / num_bin,
            intervals.right.detach().cpu().numpy() / num_bin,
        )
    )


def _empty_metrics(n_intervals: int, n_empty: int) -> dict[str, float | int]:
    return {
        "n_intervals": n_intervals,
        "eval_nonempty_intervals": 0,
        "eval_empty_intervals": n_empty,
        "heldout_ptrue_fail_rate": math.nan,
        "heldout_label_fail_rate": math.nan,
        "heldout_ptrue_pass_rate": math.nan,
        "heldout_label_pass_rate": math.nan,
        "median_heldout_ptrue_min_slack": math.nan,
        "q10_heldout_ptrue_min_slack": math.nan,
        "median_heldout_label_min_slack": math.nan,
        "median_eval_count_per_interval": math.nan,
    }


def heldout_interval_metrics(
    intervals: IntervalIndices,
    eval_stats: BinnedStats,
    return_detail: bool = True,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    """Evaluate returned intervals on held-out (left, right] statistics."""
    if eval_stats.p_true_sums is None:
        raise ValueError("held-out statistics must include p_true sums")

    n_intervals = len(intervals)
    empty_detail = {
        "interval_left": np.empty(0, dtype=np.float64),
        "interval_right": np.empty(0, dtype=np.float64),
        "interval_length": np.empty(0, dtype=np.float64),
        "satisfies_picpi": np.empty(0, dtype=bool),
    }
    if n_intervals == 0:
        return _empty_metrics(0, 0), empty_detail

    device = eval_stats.counts.device
    counts_prefix = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.float64),
            eval_stats.counts.to(torch.float64).cumsum(0),
        )
    )
    successes_prefix = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.float64),
            eval_stats.successes.cumsum(0),
        )
    )
    p_true_prefix = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.float64),
            eval_stats.p_true_sums.cumsum(0),
        )
    )

    counts = counts_prefix[intervals.right] - counts_prefix[intervals.left]
    nonempty = counts > 0
    n_nonempty = int(nonempty.sum().item())
    if n_nonempty == 0:
        return _empty_metrics(n_intervals, n_intervals), empty_detail

    left_idx = intervals.left[nonempty]
    right_idx = intervals.right[nonempty]
    nonempty_counts = counts[nonempty]
    mean_y = (
        successes_prefix[right_idx] - successes_prefix[left_idx]
    ) / nonempty_counts
    mean_p_true = (p_true_prefix[right_idx] - p_true_prefix[left_idx]) / nonempty_counts

    num_bin = int(eval_stats.counts.numel())
    left = left_idx.to(torch.float64) / num_bin
    right = right_idx.to(torch.float64) / num_bin
    ptrue_slack = torch.minimum(mean_p_true - left, right - mean_p_true)
    label_slack = torch.minimum(mean_y - left, right - mean_y)
    ptrue_ok = (left < mean_p_true) & (mean_p_true <= right)
    label_ok = (left < mean_y) & (mean_y <= right)

    metrics: dict[str, float | int] = {
        "n_intervals": n_intervals,
        "eval_nonempty_intervals": n_nonempty,
        "eval_empty_intervals": n_intervals - n_nonempty,
        "heldout_ptrue_fail_rate": float((~ptrue_ok).to(torch.float64).mean().item()),
        "heldout_label_fail_rate": float((~label_ok).to(torch.float64).mean().item()),
        "heldout_ptrue_pass_rate": float(ptrue_ok.to(torch.float64).mean().item()),
        "heldout_label_pass_rate": float(label_ok.to(torch.float64).mean().item()),
        "median_heldout_ptrue_min_slack": float(
            torch.quantile(ptrue_slack, 0.50).item()
        ),
        "q10_heldout_ptrue_min_slack": float(torch.quantile(ptrue_slack, 0.10).item()),
        "median_heldout_label_min_slack": float(
            torch.quantile(label_slack, 0.50).item()
        ),
        "median_eval_count_per_interval": float(
            torch.quantile(nonempty_counts, 0.50).item()
        ),
    }
    detail = (
        {
            "interval_left": left.detach().cpu().numpy(),
            "interval_right": right.detach().cpu().numpy(),
            "interval_length": (right - left).detach().cpu().numpy(),
            "satisfies_picpi": ptrue_ok.detach().cpu().numpy(),
        }
        if return_detail
        else empty_detail
    )
    return metrics, detail


def fit_projection_params(cfg: Config) -> tuple[ProjectionParams, np.ndarray]:
    """Fit the notebook's sklearn model and derive the 2-D projection basis."""
    rng = np.random.default_rng(cfg.seed)
    x_train = rng.normal(0.0, 1.0, size=(cfg.n_train, MULTIVARIATE_DIM))
    noise = rng.normal(0.0, MULTIVARIATE_EPSILON, size=cfg.n_train)
    label_probability = expit(MULTIVARIATE_BETA0 + x_train @ MULTIVARIATE_BETA + noise)
    y_train = rng.binomial(1, label_probability)

    model = LogisticRegression(solver="lbfgs", max_iter=2000)
    model.fit(x_train, y_train)
    fitted_weight = np.asarray(model.coef_[0], dtype=np.float64)
    fitted_scale = float(np.linalg.norm(fitted_weight))
    if fitted_scale == 0.0:
        true_on_fitted_axis = 0.0
        true_residual_scale = float(np.linalg.norm(MULTIVARIATE_BETA))
    else:
        true_on_fitted_axis = float(
            np.dot(MULTIVARIATE_BETA, fitted_weight) / fitted_scale
        )
        residual_variance = float(
            np.dot(MULTIVARIATE_BETA, MULTIVARIATE_BETA) - true_on_fitted_axis**2
        )
        true_residual_scale = math.sqrt(max(0.0, residual_variance))

    params = ProjectionParams(
        fitted_intercept=float(model.intercept_[0]),
        fitted_scale=fitted_scale,
        true_intercept=MULTIVARIATE_BETA0,
        true_on_fitted_axis=true_on_fitted_axis,
        true_residual_scale=true_residual_scale,
    )
    return params, fitted_weight


def gauss_hermite_p_true(
    true_linear_predictor: torch.Tensor,
    offsets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply the notebook's 40-node logistic-normal quadrature in small tiles."""
    result = torch.zeros_like(true_linear_predictor)
    for start in range(0, offsets.numel(), GAUSS_HERMITE_TILE):
        stop = min(start + GAUSS_HERMITE_TILE, offsets.numel())
        probabilities = torch.sigmoid(
            true_linear_predictor[:, None] + offsets[None, start:stop]
        )
        result.add_(probabilities @ weights[start:stop])
    return result / math.sqrt(math.pi)


@torch.no_grad()
def simulate_binned_stats(
    total_count: int,
    cfg: Config,
    params: ProjectionParams,
    device: torch.device,
    generator: torch.Generator,
    include_p_true: bool,
    quadrature_offsets: torch.Tensor,
    quadrature_weights: torch.Tensor,
) -> BinnedStats:
    """Stream a fold on GPU and retain only fixed-grid sufficient statistics."""
    edges = make_edges(cfg.num_bin, device)
    stats = empty_binned_stats(cfg.num_bin, device, include_p_true)

    remaining = total_count
    while remaining > 0:
        current = min(cfg.chunk_size, remaining)
        standard_normals = torch.randn(
            (3, current),
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        fitted_linear = (
            params.fitted_intercept + params.fitted_scale * standard_normals[0]
        )
        true_linear = (
            params.true_intercept
            + params.true_on_fitted_axis * standard_normals[0]
            + params.true_residual_scale * standard_normals[1]
        )
        label_probability = torch.sigmoid(
            true_linear + MULTIVARIATE_EPSILON * standard_normals[2]
        )
        y = torch.bernoulli(label_probability, generator=generator)
        p_hat = torch.sigmoid(fitted_linear)
        p_true = (
            gauss_hermite_p_true(
                true_linear,
                quadrature_offsets,
                quadrature_weights,
            )
            if include_p_true
            else None
        )
        accumulate_binned_stats(stats, p_hat, y, edges, p_true)
        remaining -= current

    return stats


def worker_run(
    worker_id: int,
    gpu_id: int,
    tasks: list[Task],
    cfg: Config,
    params: ProjectionParams,
) -> tuple[
    list[dict[str, float | int | str]], list[dict[str, float | int | bool | str]]
]:
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    nodes_np, weights_np = np.polynomial.hermite.hermgauss(GAUSS_HERMITE_ORDER)
    quadrature_offsets = torch.as_tensor(
        math.sqrt(2.0) * MULTIVARIATE_EPSILON * nodes_np,
        device=device,
        dtype=torch.float64,
    )
    quadrature_weights = torch.as_tensor(
        weights_np,
        device=device,
        dtype=torch.float64,
    )

    result_rows: list[dict[str, float | int | str]] = []
    interval_rows: list[dict[str, float | int | bool | str]] = []
    for task in tasks:
        generator = torch.Generator(device=device)
        generator.manual_seed(task.seed)
        task_cfg = replace(cfg, num_bin=task.num_bin)

        calibration_stats = simulate_binned_stats(
            task.n_calib,
            task_cfg,
            params,
            device,
            generator,
            include_p_true=False,
            quadrature_offsets=quadrature_offsets,
            quadrature_weights=quadrature_weights,
        )
        empirical_intervals = empirical_intervals_from_stats(calibration_stats)
        population_intervals = population_intervals_from_stats(
            calibration_stats, task_cfg.delta
        )
        eval_stats = simulate_binned_stats(
            task_cfg.n_eval,
            task_cfg,
            params,
            device,
            generator,
            include_p_true=True,
            quadrature_offsets=quadrature_offsets,
            quadrature_weights=quadrature_weights,
        )

        for method, intervals in (
            ("Empirical mode", empirical_intervals),
            ("Population mode", population_intervals),
        ):
            metrics, detail = heldout_interval_metrics(
                intervals,
                eval_stats,
                return_detail=method == "Empirical mode",
            )
            result_rows.append(
                {
                    "n_calib": task.n_calib,
                    "K": task.num_bin,
                    "rep": task.rep,
                    "task_seed": task.seed,
                    "method": method,
                    **metrics,
                }
            )
            if method == "Empirical mode":
                interval_rows.extend(
                    {
                        "n_calib": task.n_calib,
                        "K": task.num_bin,
                        "rep": task.rep,
                        "task_seed": task.seed,
                        "method": method,
                        "interval_left": float(left),
                        "interval_right": float(right),
                        "interval_length": float(width),
                        "satisfies_picpi": bool(ok),
                    }
                    for left, right, width, ok in zip(
                        detail["interval_left"],
                        detail["interval_right"],
                        detail["interval_length"],
                        detail["satisfies_picpi"],
                    )
                )

    print(
        f"worker {worker_id} on cuda:{gpu_id} completed {len(tasks)} tasks",
        flush=True,
    )
    return result_rows, interval_rows


def build_tasks(
    n_grid: Iterable[int],
    reps: int,
    seed: int,
    num_bin: int = 100,
    adaptive_c: float | None = None,
) -> list[Task]:
    master_rng = np.random.default_rng(seed + 424_242)
    tasks: list[Task] = []
    for n_idx, n_calib in enumerate(n_grid):
        task_num_bin = (
            adaptive_num_bin(int(n_calib), adaptive_c)
            if adaptive_c is not None
            else num_bin
        )
        for rep in range(reps):
            task_seed = int(master_rng.integers(0, 2**32 - 1))
            tasks.append(
                Task(
                    n_idx=n_idx,
                    n_calib=int(n_calib),
                    num_bin=task_num_bin,
                    rep=rep,
                    seed=task_seed,
                )
            )
    return tasks


def shard_tasks(tasks: list[Task], num_shards: int) -> list[list[Task]]:
    shards = [[] for _ in range(num_shards)]
    for index, task in enumerate(tasks):
        shards[index % num_shards].append(task)
    return shards


def validate_cuda(gpu_ids: list[int]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this script requires a CUDA GPU.")
    device_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= device_count]
    if invalid:
        raise ValueError(
            f"Invalid GPU ids {invalid}; this system exposes {device_count} CUDA devices."
        )


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["n_calib", "K", "method"], as_index=False)
        .agg(
            n_intervals_mean=("n_intervals", "mean"),
            n_intervals_std=("n_intervals", "std"),
            heldout_ptrue_fail_rate_mean=("heldout_ptrue_fail_rate", "mean"),
            heldout_ptrue_fail_rate_std=("heldout_ptrue_fail_rate", "std"),
            heldout_ptrue_pass_rate_mean=("heldout_ptrue_pass_rate", "mean"),
            heldout_ptrue_pass_rate_std=("heldout_ptrue_pass_rate", "std"),
        )
        .sort_values(["n_calib", "method"])
        .reset_index(drop=True)
    )


def aggregate_interval_widths(interval_results: pd.DataFrame) -> pd.DataFrame:
    width_results = interval_results.copy()
    width_results["interval_length"] = (
        width_results["interval_length"].astype(float).round(12)
    )
    width_results["satisfies_picpi"] = width_results["satisfies_picpi"].astype(bool)
    result = (
        width_results.groupby(
            ["method", "n_calib", "K", "interval_length"], as_index=False
        )
        .agg(
            picpi_pass_rate=("satisfies_picpi", "mean"),
            n_intervals=("satisfies_picpi", "size"),
        )
        .sort_values(["n_calib", "interval_length"])
        .reset_index(drop=True)
    )
    result["interval_frequency"] = result["n_intervals"] / result.groupby(
        ["method", "n_calib"]
    )["n_intervals"].transform("sum")
    return result


def aggregate_interval_width_bins(
    interval_results: pd.DataFrame,
    num_width_bins: int = 40,
) -> pd.DataFrame:
    """Aggregate varying-K interval lengths on one shared logarithmic grid."""
    widths = interval_results["interval_length"].to_numpy(dtype=float)
    min_width = float(widths.min())
    max_width = float(widths.max())
    if min_width == max_width:
        result = aggregate_interval_widths(interval_results)
        result["interval_length_lower"] = min_width
        result["interval_length_upper"] = max_width
        return result

    edges = np.geomspace(min_width, max_width, num_width_bins + 1)
    bin_index = np.searchsorted(edges, widths, side="right") - 1
    bin_index = np.clip(bin_index, 0, num_width_bins - 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    width_results = interval_results.copy()
    width_results["width_bin"] = bin_index
    width_results["satisfies_picpi"] = width_results["satisfies_picpi"].astype(bool)
    result = (
        width_results.groupby(["method", "n_calib", "K", "width_bin"], as_index=False)
        .agg(
            picpi_pass_rate=("satisfies_picpi", "mean"),
            n_intervals=("satisfies_picpi", "size"),
        )
        .sort_values(["n_calib", "width_bin"])
        .reset_index(drop=True)
    )
    result["interval_length"] = centers[result["width_bin"].to_numpy()]
    result["interval_length_lower"] = edges[result["width_bin"].to_numpy()]
    result["interval_length_upper"] = edges[result["width_bin"].to_numpy() + 1]
    result["interval_frequency"] = result["n_intervals"] / result.groupby(
        ["method", "n_calib"]
    )["n_intervals"].transform("sum")
    return result


def save_figures(
    summary: pd.DataFrame,
    width_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8")
    method_styles = {
        "Empirical mode": {"color": "tab:blue", "marker": "o"},
        "Population mode": {"color": "tab:green", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for method, style in method_styles.items():
        current = summary[summary["method"] == method].sort_values("n_calib")
        ax.errorbar(
            current["n_calib"],
            current["heldout_ptrue_pass_rate_mean"],
            yerr=current["heldout_ptrue_pass_rate_std"],
            capsize=4,
            linewidth=2.0,
            markersize=7,
            label=method,
            **style,
        )

    calibration_sizes = np.sort(summary["n_calib"].unique())
    ax.set_xscale("log")
    ax.set_xticks(calibration_sizes)
    ax.set_xticklabels([f"{int(n):g}" for n in calibration_sizes], rotation=20)
    ax.set_xlabel("Calibration sample size", fontsize=16)
    ax.set_ylabel("Fraction of intervals passing PICPI property", fontsize=16)
    ax.set_title("PICPI property pass rate on held-out data", fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        output_dir / "heldout_picpi_failure_rate_by_method.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "heldout_picpi_failure_rate_by_method.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    interval_lengths = np.sort(width_summary["interval_length"].unique())
    calibration_sizes = np.sort(width_summary["n_calib"].unique())
    frequency_matrix = (
        width_summary.pivot(
            index="interval_length",
            columns="n_calib",
            values="interval_frequency",
        )
        .reindex(index=interval_lengths, columns=calibration_sizes)
        .fillna(0)
    )
    pass_rate_matrix = width_summary.pivot(
        index="interval_length",
        columns="n_calib",
        values="picpi_pass_rate",
    ).reindex(index=interval_lengths, columns=calibration_sizes)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.8), layout="constrained")
    frequency_image = axes[0].imshow(
        frequency_matrix.to_numpy(dtype=float),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
    )
    frequency_colorbar = fig.colorbar(frequency_image, ax=axes[0])
    frequency_colorbar.set_label(
        "Interval frequency\n(conditioned on sample size)",
        fontsize=14,
    )
    frequency_colorbar.ax.tick_params(labelsize=12)

    pass_rate_cmap = plt.cm.viridis.copy()
    pass_rate_cmap.set_bad(color="lightgray")
    pass_rate_image = axes[1].imshow(
        np.ma.masked_invalid(pass_rate_matrix.to_numpy(dtype=float)),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=pass_rate_cmap,
        vmin=0.0,
        vmax=1.0,
    )
    pass_rate_colorbar = fig.colorbar(pass_rate_image, ax=axes[1])
    pass_rate_colorbar.set_label("Held-out PICPI pass rate", fontsize=14)
    pass_rate_colorbar.ax.tick_params(labelsize=12)

    width_axis_label = (
        "Returned interval length (log bins)"
        if "width_bin" in width_summary.columns
        else "Returned interval length"
    )
    y_tick_positions = np.unique(
        np.linspace(
            0, len(interval_lengths) - 1, num=min(10, len(interval_lengths))
        ).astype(int)
    )
    for ax in axes:
        ax.set_xlabel("Calibration sample size", fontsize=16)
        ax.set_ylabel(width_axis_label, fontsize=16)
        ax.set_xticks(np.arange(len(calibration_sizes)))
        ax.set_xticklabels(
            [f"{int(n):,}" for n in calibration_sizes],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(
            [f"{interval_lengths[position]:.3g}" for position in y_tick_positions]
        )
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_title("Interval-length frequency", fontsize=18)
    axes[1].set_title("Held-out PICPI property pass rate", fontsize=18)
    fig.savefig(
        output_dir / "heldout_picpi_pass_rate_by_interval_length.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "heldout_picpi_pass_rate_by_interval_length.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce empirical-mode PICPI diagnosis figures with chunked GPU "
            "sufficient statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpu-ids",
        "--gpu_ids",
        dest="gpu_ids",
        type=parse_gpu_ids,
        default=parse_gpu_ids("0"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("script/outputs"))
    parser.add_argument(
        "--n-calib-grid",
        nargs="+",
        type=int,
        default=list(DEFAULT_N_CALIB_GRID),
    )
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=1_000_000)
    parser.add_argument("--n-train", type=int, default=2_000)
    parser.add_argument("--num-bin", type=int, default=100)
    parser.add_argument(
        "--adaptive-c",
        type=float,
        default=None,
        help="Use K=round(C*n_calib^(1/3)); omitted means fixed --num-bin.",
    )
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    n_grid = tuple(sorted(set(args.n_calib_grid)))
    if not n_grid or min(n_grid) <= 0:
        raise ValueError("--n-calib-grid values must be positive")
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if args.n_eval <= 0 or args.n_train <= 0:
        raise ValueError("--n-eval and --n-train must be positive")
    if args.num_bin <= 0:
        raise ValueError("--num-bin must be positive")
    if args.adaptive_c is not None and args.adaptive_c <= 0:
        raise ValueError("--adaptive-c must be positive")
    if not 0.0 < args.delta < 1.0:
        raise ValueError("--delta must be in (0, 1)")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    validate_cuda(args.gpu_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        n_train=args.n_train,
        n_eval=args.n_eval,
        reps=args.reps,
        num_bin=args.num_bin,
        delta=args.delta,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    params, fitted_weight = fit_projection_params(cfg)
    print(
        "Estimator: "
        f"intercept={params.fitted_intercept:.6f}, "
        f"weight_norm={np.linalg.norm(fitted_weight):.6f}"
    )

    tasks = build_tasks(
        n_grid,
        cfg.reps,
        cfg.seed,
        num_bin=cfg.num_bin,
        adaptive_c=args.adaptive_c,
    )
    if args.adaptive_c is not None:
        schedule = ", ".join(
            f"{n}:K={adaptive_num_bin(n, args.adaptive_c)}" for n in n_grid
        )
        print(f"Adaptive bins with C={args.adaptive_c:g}: {schedule}")
    shards = shard_tasks(tasks, len(args.gpu_ids))
    print(
        f"Running {len(tasks)} calibration/evaluation tasks on GPUs {args.gpu_ids} ..."
    )

    result_rows: list[dict[str, float | int | str]] = []
    interval_rows: list[dict[str, float | int | bool | str]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(args.gpu_ids),
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(
                worker_run,
                worker_id,
                gpu_id,
                shard,
                cfg,
                params,
            )
            for worker_id, (gpu_id, shard) in enumerate(zip(args.gpu_ids, shards))
            if shard
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="GPU workers",
        ):
            worker_results, worker_intervals = future.result()
            result_rows.extend(worker_results)
            interval_rows.extend(worker_intervals)

    results = (
        pd.DataFrame(result_rows)
        .sort_values(["n_calib", "rep", "method"])
        .reset_index(drop=True)
    )
    interval_results = (
        pd.DataFrame(interval_rows)
        .sort_values(["n_calib", "rep", "interval_left", "interval_right"])
        .reset_index(drop=True)
    )
    summary = aggregate_results(results)
    width_summary = aggregate_interval_widths(interval_results)
    plot_width_summary = width_summary
    if interval_results["K"].nunique() > 1:
        plot_width_summary = aggregate_interval_width_bins(interval_results)
        plot_width_summary.to_csv(
            args.output_dir / "heldout_picpi_width_binned_summary.csv",
            index=False,
        )

    results.to_csv(args.output_dir / "heldout_picpi_results.csv", index=False)
    interval_results.to_csv(
        args.output_dir / "heldout_picpi_interval_results.csv",
        index=False,
    )
    summary.to_csv(args.output_dir / "heldout_picpi_summary.csv", index=False)
    width_summary.to_csv(
        args.output_dir / "heldout_picpi_width_summary.csv",
        index=False,
    )
    save_figures(summary, plot_width_summary, args.output_dir)
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
