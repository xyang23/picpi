"""Reusable aggregation and verification helpers for stored experiment results.

These aggregations are the same as the original GPU / sweep runners.
They let the plotting path prove that the cached intermediates determine
the paper figures, without rerunning CUDA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from picpi.paths import CACHED_RESULTS
from scripts.plot_helper import DGP_DISPLAY_ORDER
from scripts.task1_helper import TASK1_METHOD_ORDER, aggregate_multivariate_table


def aggregate_thm(df_results: pd.DataFrame) -> pd.DataFrame:
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


def aggregate_empirical_results(results: pd.DataFrame) -> pd.DataFrame:
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


def aggregate_empirical_widths(interval_results: pd.DataFrame) -> pd.DataFrame:
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


def aggregate_task2_long(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby(["method", "target_coverage"])
        .agg(
            n_completed_seeds=("seed", "nunique"),
            mean_avg_set_size=("avg_set_size", "mean"),
            sd_avg_set_size=("avg_set_size", "std"),
            mean_macro_coverage=("macro_coverage", "mean"),
            sd_macro_coverage=("macro_coverage", "std"),
            mean_worst_coverage=("worst_coverage", "mean"),
            sd_worst_coverage=("worst_coverage", "std"),
        )
        .reset_index()
        .sort_values(["method", "target_coverage"])
        .fillna(0.0)
    )
    summary["dgp_name"] = long_df["dgp_name"].iloc[0]
    summary["dgp_title"] = long_df["dgp_title"].iloc[0]
    return summary


def _assert_frame_close(
    rebuilt: pd.DataFrame,
    stored: pd.DataFrame,
    *,
    name: str,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> None:
    rebuilt = rebuilt.reset_index(drop=True)
    stored = stored.reset_index(drop=True)
    if list(rebuilt.columns) != list(stored.columns):
        raise AssertionError(
            f"{name}: column mismatch\nrebuilt={list(rebuilt.columns)}\nstored={list(stored.columns)}"
        )
    if len(rebuilt) != len(stored):
        raise AssertionError(
            f"{name}: row count {len(rebuilt)} != stored {len(stored)}"
        )
    for column in rebuilt.columns:
        left = rebuilt[column]
        right = stored[column]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                raise AssertionError(f"{name}: numeric mismatch in {column}")
        else:
            if not left.astype(str).equals(right.astype(str)):
                raise AssertionError(f"{name}: value mismatch in {column}")


def load_and_verify_thm(
    results_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = Path(results_dir) if results_dir is not None else CACHED_RESULTS / "thm"
    df_results = pd.read_csv(results_dir / "thm_df_results.csv")
    stored = pd.read_csv(results_dir / "thm_summary.csv")
    rebuilt = aggregate_thm(df_results)
    _assert_frame_close(rebuilt, stored, name="thm summary")
    return df_results, rebuilt


def load_and_verify_empirical(
    results_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = (
        Path(results_dir)
        if results_dir is not None
        else CACHED_RESULTS / "empirical_mode"
    )
    results = pd.read_csv(results_dir / "heldout_picpi_results.csv")
    interval_results = pd.read_csv(results_dir / "heldout_picpi_interval_results.csv")
    stored_summary = pd.read_csv(results_dir / "heldout_picpi_summary.csv")
    stored_width = pd.read_csv(results_dir / "heldout_picpi_width_summary.csv")
    rebuilt_summary = aggregate_empirical_results(results)
    rebuilt_width = aggregate_empirical_widths(interval_results)
    _assert_frame_close(rebuilt_summary, stored_summary, name="empirical summary")
    _assert_frame_close(rebuilt_width, stored_width, name="empirical width summary")
    return rebuilt_summary, rebuilt_width


def load_and_verify_task1(results_dir: Path | None = None) -> pd.DataFrame:
    from scripts.task1_helper import load_task1

    payload = load_task1(results_dir)
    rebuilt = aggregate_multivariate_table(payload["mc_results"])
    _assert_frame_close(rebuilt, payload["mc_summary"], name="task1 MC summary")
    return payload


def load_and_verify_task2(
    results_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    results_dir = Path(results_dir) if results_dir is not None else CACHED_RESULTS / "task2"
    bundles: dict[str, dict[str, object]] = {}
    for dgp_name in DGP_DISPLAY_ORDER:
        dgp_dir = results_dir / dgp_name
        long_df = pd.read_csv(dgp_dir / "long.csv")
        stored = pd.read_csv(dgp_dir / "summary.csv")
        rebuilt = aggregate_task2_long(long_df)
        # The nonlinear stored summary still contains leftover 200/800-seed
        # incremental rows. Compare only the complete 1000-seed rows; plot
        # from the rebuilt long.csv aggregate.
        complete_stored = stored.loc[
            stored["n_completed_seeds"] == stored["n_completed_seeds"].max()
        ].copy()
        metric_cols = [
            "method",
            "target_coverage",
            "n_completed_seeds",
            "mean_avg_set_size",
            "sd_avg_set_size",
            "mean_macro_coverage",
            "sd_macro_coverage",
            "mean_worst_coverage",
            "sd_worst_coverage",
        ]
        overlap = rebuilt.merge(
            complete_stored[metric_cols],
            on=["method", "target_coverage"],
            suffixes=("_rebuilt", "_stored"),
        )
        if overlap.empty:
            raise AssertionError(f"task2 {dgp_name}: no overlapping complete-seed rows")
        for column in metric_cols:
            if column in {"method", "target_coverage"}:
                continue
            left = overlap[f"{column}_rebuilt"].to_numpy(dtype=float)
            right = overlap[f"{column}_stored"].to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True):
                raise AssertionError(
                    f"task2 {dgp_name}: mismatch in {column} on complete-seed rows"
                )
        bundles[dgp_name] = {
            "summary": rebuilt,
            "config": json.loads((dgp_dir / "config.json").read_text()),
        }
    return bundles
