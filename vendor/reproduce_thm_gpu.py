#!/usr/bin/env python3
"""GPU reproduction of experiments/veri_Thm5.2/thm_demo.ipynb.

The notebook samples calibration rows directly and then scans them once for
every candidate interval. This script keeps the same Task 1 discrete X grid and
population-calibration criterion, but uses aggregate sufficient statistics:
counts per X_GRID point and successes per X_GRID point.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


DEFAULT_N_GRID = (
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
NUM_ARM = 1000


@dataclass(frozen=True)
class Config:
    beta: float
    sigma: float
    n_train: int
    n_eval: int
    n_eps_eval: int
    reps: int
    bins: int
    delta: float
    seed: int


@dataclass(frozen=True)
class Task:
    n_idx: int
    n_calib: int
    rep: int
    seed: int


def parse_gpu_ids(raw: str) -> list[int]:
    ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("--gpu-ids must contain at least one id")
    if any(gpu_id < 0 for gpu_id in ids):
        raise argparse.ArgumentTypeError("--gpu-ids must be non-negative")
    return ids


def format_n(n: int) -> str:
    if n >= 1_000:
        exponent = int(math.floor(math.log10(n)))
        coefficient = n / (10**exponent)
        coefficient_text = (
            str(int(round(coefficient)))
            if abs(coefficient - round(coefficient)) < 1e-12
            else f"{coefficient:g}"
        )
        return f"{coefficient_text}e{exponent}"
    return str(n)


def make_x_grid(device: torch.device) -> torch.Tensor:
    return torch.linspace(-2.0, 2.0, NUM_ARM, device=device, dtype=torch.float64)


def task1_label_prob_grid(
    x_grid: torch.Tensor,
    beta: float,
    sigma: float,
    quadrature_order: int = 64,
) -> torch.Tensor:
    """Compute E_Z[sigmoid(beta*x + sigma*Z)] on X_GRID."""
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(quadrature_order)
    nodes = torch.as_tensor(nodes_np, device=x_grid.device, dtype=torch.float64)
    weights = torch.as_tensor(weights_np, device=x_grid.device, dtype=torch.float64)
    logits = beta * x_grid[:, None] + math.sqrt(2.0) * sigma * nodes[None, :]
    return (torch.sigmoid(logits) * weights[None, :]).sum(dim=1) / math.sqrt(math.pi)


def sample_uniform_multinomial_counts(
    total_count: int,
    num_categories: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample exact multinomial counts for uniform categories by binary splits."""
    counts = torch.empty(num_categories, device=device, dtype=torch.float64)
    starts = torch.zeros(1, device=device, dtype=torch.long)
    sizes = torch.full((1,), num_categories, device=device, dtype=torch.long)
    totals = torch.full((1,), float(total_count), device=device, dtype=torch.float64)

    while starts.numel() > 0:
        leaf_mask = sizes == 1
        if leaf_mask.any():
            counts[starts[leaf_mask]] = totals[leaf_mask]

        split_mask = ~leaf_mask
        if not split_mask.any():
            break

        split_starts = starts[split_mask]
        split_sizes = sizes[split_mask]
        split_totals = totals[split_mask]

        left_sizes = split_sizes // 2
        right_sizes = split_sizes - left_sizes
        probs = left_sizes.to(torch.float64) / split_sizes.to(torch.float64)
        left_totals = torch.distributions.Binomial(
            total_count=split_totals,
            probs=probs,
        ).sample()
        right_totals = split_totals - left_totals

        starts = torch.cat((split_starts, split_starts + left_sizes))
        sizes = torch.cat((left_sizes, right_sizes))
        totals = torch.cat((left_totals, right_totals))

    return counts


def train_torch_logistic(cfg: Config, device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    x_grid = make_x_grid(device)
    train_idx = torch.randint(NUM_ARM, (cfg.n_train,), device=device)
    x_train = x_grid[train_idx]
    noise = torch.randn(cfg.n_train, device=device, dtype=torch.float64)
    y_prob = torch.sigmoid(cfg.beta * x_train + cfg.sigma * noise)
    y_train = torch.bernoulli(y_prob)

    weight = torch.zeros((), device=device, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), device=device, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=500,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        logits = weight * x_train + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        return loss

    opt.step(closure)

    p_hat_grid = torch.sigmoid(weight.detach() * x_grid + bias.detach())
    p_true_grid = torch.sigmoid(cfg.beta * x_grid)
    label_prob_grid = task1_label_prob_grid(x_grid, cfg.beta, cfg.sigma)

    torch.manual_seed(cfg.seed + 1)
    torch.cuda.manual_seed_all(cfg.seed + 1)
    eps_counts = sample_uniform_multinomial_counts(cfg.n_eps_eval, NUM_ARM, device)
    eps_abs_err = torch.abs(p_hat_grid - p_true_grid)
    eps_hat = float((eps_abs_err * eps_counts).sum().cpu().item() / cfg.n_eps_eval)

    torch.manual_seed(cfg.seed + 2)
    torch.cuda.manual_seed_all(cfg.seed + 2)
    eval_counts = sample_uniform_multinomial_counts(cfg.n_eval, NUM_ARM, device)

    return (
        p_hat_grid.detach().cpu(),
        p_true_grid.detach().cpu(),
        label_prob_grid.detach().cpu(),
        eval_counts.detach().cpu(),
        torch.stack((weight.detach(), bias.detach())).cpu(),
        torch.tensor(eps_hat, dtype=torch.float64),
    )


def build_interval_tensors(
    bins: int,
    p_hat_grid: torch.Tensor,
    p_true_grid: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    idx_i, idx_j = zip(*((i, j) for i in range(bins) for j in range(i + 1, bins + 1)))
    starts = torch.tensor(idx_i, device=device, dtype=torch.float64) / bins
    ends = torch.tensor(idx_j, device=device, dtype=torch.float64) / bins
    spans = ends - starts

    p_hat_grid = p_hat_grid.to(device=device, dtype=torch.float64)
    p_true_grid = p_true_grid.to(device=device, dtype=torch.float64)
    calib_contains = (p_hat_grid[None, :] >= starts[:, None]) & (
        p_hat_grid[None, :] <= ends[:, None]
    )
    eval_contains = (p_true_grid[None, :] >= starts[:, None]) & (
        p_true_grid[None, :] <= ends[:, None]
    )
    return (
        starts,
        ends,
        spans,
        calib_contains.to(torch.float64),
        eval_contains,
    )


def run_one_task(
    task: Task,
    cfg: Config,
    label_prob_grid: torch.Tensor,
    eval_counts: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    spans: torch.Tensor,
    calib_contains: torch.Tensor,
    eval_contains: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int]:
    torch.manual_seed(task.seed)
    torch.cuda.manual_seed_all(task.seed)

    x_counts = sample_uniform_multinomial_counts(task.n_calib, NUM_ARM, device)
    y_success = torch.distributions.Binomial(
        total_count=x_counts,
        probs=label_prob_grid,
    ).sample()

    denom = calib_contains @ x_counts
    numerator = calib_contains @ y_success
    nonzero = denom > 0
    mean_y = torch.zeros_like(denom)
    mean_y[nonzero] = numerator[nonzero] / denom[nonzero]

    gen_gap = torch.full_like(denom, float("inf"))
    gen_gap[nonzero] = 2.0 * torch.sqrt(
        math.log(cfg.bins**2 / cfg.delta) / denom[nonzero]
    )
    valid = nonzero & ((starts + gen_gap) <= mean_y) & (mean_y <= (ends - gen_gap))

    candidate_widths = torch.where(valid[:, None] & eval_contains, spans[:, None], math.inf)
    widths_grid = candidate_widths.min(dim=0).values
    widths_grid = torch.where(torch.isinf(widths_grid), torch.ones_like(widths_grid), widths_grid)

    eval_widths = torch.repeat_interleave(widths_grid, eval_counts.to(torch.long))
    rate_term = (math.log(task.n_calib) / task.n_calib) ** (1.0 / 3.0)

    return {
        "n_calib": task.n_calib,
        "width_mean": float(eval_widths.mean().cpu().item()),
        "width_median": float(torch.quantile(eval_widths, 0.50).cpu().item()),
        "width_q90": float(torch.quantile(eval_widths, 0.90).cpu().item()),
        "width_q95": float(torch.quantile(eval_widths, 0.95).cpu().item()),
        "coverage": float((eval_widths < 1.0).to(torch.float64).mean().cpu().item()),
        "n_raw_intervals": int(valid.sum().cpu().item()),
        "n_intervals": int(valid.sum().cpu().item()),
        "rate_term": rate_term,
        "rep": task.rep,
    }


def worker_run(
    worker_id: int,
    gpu_id: int,
    tasks: list[Task],
    cfg: Config,
    p_hat_grid_cpu: torch.Tensor,
    p_true_grid_cpu: torch.Tensor,
    label_prob_grid_cpu: torch.Tensor,
    eval_counts_cpu: torch.Tensor,
) -> list[dict[str, float | int]]:
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    label_prob_grid = label_prob_grid_cpu.to(device=device, dtype=torch.float64)
    eval_counts = eval_counts_cpu.to(device=device, dtype=torch.float64)
    tensors = build_interval_tensors(cfg.bins, p_hat_grid_cpu, p_true_grid_cpu, device)
    starts, ends, spans, calib_contains, eval_contains = tensors

    records = []
    for task in tasks:
        records.append(
            run_one_task(
                task,
                cfg,
                label_prob_grid,
                eval_counts,
                starts,
                ends,
                spans,
                calib_contains,
                eval_contains,
                device,
            )
        )

    print(f"worker {worker_id} on cuda:{gpu_id} completed {len(records)} tasks", flush=True)
    return records


def aggregate_results(df_results: pd.DataFrame) -> pd.DataFrame:
    return (
        df_results.groupby("n_calib")
        .agg(
            width_median_mean=("width_median", "mean"),
            width_mean_mean=("width_mean", "mean"),
            width_median_sd=("width_median", "std"),
            width_q90_mean=("width_q90", "mean"),
            width_q90_sd=("width_q90", "std"),
            coverage_mean=("coverage", "mean"),
            n_raw_intervals_mean=("n_raw_intervals", "mean"),
            n_intervals_mean=("n_intervals", "mean"),
            rate_term=("rate_term", "mean"),
        )
        .reset_index()
    )


def estimate_loglog_slope(summary: pd.DataFrame, width_col: str) -> float:
    x = np.log(summary["rate_term"].to_numpy())
    y = np.log(summary[width_col].to_numpy())
    return float(np.polyfit(x, y, deg=1)[0])


def save_figures(df_results: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(summary["rate_term"], summary["width_mean_mean"], "o-", label="Mean width")
    ax.plot(summary["rate_term"], summary["width_median_mean"], "o-", label="Median width")
    ax.set_xlabel(r"Reference rate $(\log n / n)^{1/3}$")
    ax.set_ylabel("Interval width")
    ax.set_title(r"Width vs. $(\log n / n)^{1/3}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "thm_width_vs_rate.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "thm_width_vs_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    n_calib_list = sorted(df_results["n_calib"].unique())
    width_data = [
        np.array(df_results[df_results["n_calib"] == n]["width_mean"])
        for n in n_calib_list
    ]
    ax.boxplot(
        width_data,
        positions=range(len(n_calib_list)),
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        meanprops=dict(
            marker="D",
            markerfacecolor="green",
            markeredgecolor="green",
            markersize=6,
        ),
    )
    ax.set_xticks(range(len(n_calib_list)))
    ax.set_xticklabels([format_n(int(n)) for n in n_calib_list], rotation=30, ha="right")
    ax.set_xlabel("Calibration sample size n")
    ax.set_ylabel("Mean interval width per repetition")
    ax.set_title("Width shrinkage with increasing n")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", linewidth=2, label="Median"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="green",
            markeredgecolor="green",
            markersize=6,
            label="Mean",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "thm_width_mean_boxplot.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "thm_width_mean_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def shard_tasks(tasks: list[Task], num_shards: int) -> list[list[Task]]:
    shards = [[] for _ in range(num_shards)]
    for idx, task in enumerate(tasks):
        shards[idx % num_shards].append(task)
    return shards


def validate_cuda(gpu_ids: list[int]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this script is intended for GPU execution.")
    device_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= device_count]
    if invalid:
        raise ValueError(f"Invalid GPU ids {invalid}; this system exposes {device_count} CUDA devices.")


def build_tasks(n_grid: Iterable[int], reps: int, seed: int) -> list[Task]:
    tasks = []
    for n_idx, n_calib in enumerate(n_grid):
        for rep in range(reps):
            task_seed = seed + 10_000_019 * (n_idx + 1) + rep
            tasks.append(Task(n_idx=n_idx, n_calib=int(n_calib), rep=rep, seed=task_seed))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the Theorem 5.2 notebook with GPU aggregate calibration.",
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
    parser.add_argument("--n-grid", nargs="+", type=int, default=list(DEFAULT_N_GRID))
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-eval", type=int, default=10_000)
    parser.add_argument("--n-eps-eval", type=int, default=100_000)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--delta", type=float, default=0.1)
    args = parser.parse_args()

    n_grid = tuple(sorted(set(args.n_grid)))
    if min(n_grid) <= 0:
        raise ValueError("--n-grid values must be positive")
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    if not 0.0 < args.delta < 1.0:
        raise ValueError("--delta must be in (0, 1)")

    validate_cuda(args.gpu_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        beta=args.beta,
        sigma=args.sigma,
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_eps_eval=args.n_eps_eval,
        reps=args.reps,
        bins=args.bins,
        delta=args.delta,
        seed=args.seed,
    )

    first_device = torch.device(f"cuda:{args.gpu_ids[0]}")
    torch.cuda.set_device(first_device)
    print(f"Training torch logistic estimator on cuda:{args.gpu_ids[0]} ...")
    (
        p_hat_grid,
        p_true_grid,
        label_prob_grid,
        eval_counts,
        params,
        eps_hat,
    ) = train_torch_logistic(cfg, first_device)
    print(
        "Estimator: "
        f"weight={float(params[0]):.6f}, bias={float(params[1]):.6f}, "
        f"eps_hat={float(eps_hat):.6f}"
    )

    tasks = build_tasks(n_grid, args.reps, args.seed)
    shards = shard_tasks(tasks, len(args.gpu_ids))
    print(f"Running {len(tasks)} calibration replications on GPUs {args.gpu_ids} ...")

    records: list[dict[str, float | int]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(args.gpu_ids), mp_context=ctx) as executor:
        futures = [
            executor.submit(
                worker_run,
                worker_id,
                gpu_id,
                shard,
                cfg,
                p_hat_grid,
                p_true_grid,
                label_prob_grid,
                eval_counts,
            )
            for worker_id, (gpu_id, shard) in enumerate(zip(args.gpu_ids, shards))
            if shard
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="GPU workers"):
            records.extend(future.result())

    df_results = pd.DataFrame(records).sort_values(["n_calib", "rep"]).reset_index(drop=True)
    summary = aggregate_results(df_results)

    df_results.to_csv(args.output_dir / "thm_df_results.csv", index=False)
    summary.to_csv(args.output_dir / "thm_summary.csv", index=False)
    save_figures(df_results, summary, args.output_dir)

    if len(summary) >= 2:
        median_slope = estimate_loglog_slope(summary, "width_median_mean")
        q90_slope = estimate_loglog_slope(summary, "width_q90_mean")
        print(f"Estimated slope (median widths) ~= {median_slope:.3f}")
        print(f"Estimated slope (0.90-quantile widths) ~= {q90_slope:.3f}")
    else:
        print("Skipped slope estimation because fewer than two n values were run.")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
