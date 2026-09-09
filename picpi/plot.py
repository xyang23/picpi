"""Draw the paper figures and table from stored results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from picpi.paths import figures_dir
from picpi.task1 import TASK1_METHOD_ORDER

DGP_DISPLAY_ORDER = ("true_linear_gaussian_d10", "population_nonlinear")
CURVE_METHOD_ORDER = [
    "Simultaneous CI",
    "PICPI",
    "Split conformal prediction",
    "Calibration-based interval",
    "Fixed-width binning",
    "Conformal classification",
]
CURVE_METHOD_STYLES = {
    "PICPI": {"color": "tab:orange", "marker": "o", "linewidth": 2.6, "markersize": 5},
    "Split conformal prediction": {
        "color": "tab:green",
        "marker": "D",
        "linewidth": 2.0,
        "markersize": 5,
    },
    "Calibration-based interval": {
        "color": "tab:red",
        "marker": "P",
        "linewidth": 2.0,
        "markersize": 5,
    },
    "Fixed-width binning": {
        "color": "tab:purple",
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 5,
    },
    "Conformal classification": {
        "color": "tab:brown",
        "marker": "X",
        "linewidth": 2.0,
        "markersize": 5,
    },
    "Simultaneous CI": {
        "color": "tab:blue",
        "marker": "^",
        "linewidth": 2.0,
        "markersize": 5,
    },
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_teaser(payload: dict[str, object], output_dir: Path | None = None) -> Path:
    """Paper Figure 1 / ``fig:teaser``: ``picpi_teaser_green.pdf``."""
    output_dir = Path(output_dir) if output_dir is not None else figures_dir()
    p_te = np.asarray(payload["p_te"], dtype=float)
    p_true_te = np.asarray(payload["p_true_te"], dtype=float)
    summary = payload["intervals"]

    box_edge, box_fill = "#2779B4", "#2779B41A"
    seg_color, diag_color, dot_edge = "#C44E52", "#555555", "#5F5F5F"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        p_te,
        p_true_te,
        s=35,
        alpha=0.8,
        facecolors="none",
        edgecolors=dot_edge,
        linewidths=0.8,
        zorder=1,
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (6, 4)),
        color=diag_color,
        linewidth=1.4,
        label="Perfect calibration",
        zorder=2,
    )

    first_box = first_seg = True
    for row in summary.itertuples(index=False):
        a, b, ybar = float(row.a), float(row.b), float(row.ybar)
        ax.add_patch(
            Rectangle(
                (a, a),
                b - a,
                b - a,
                facecolor=box_fill,
                edgecolor=box_edge,
                linewidth=1.6,
                zorder=3,
            )
        )
        ax.plot(
            [a, b],
            [ybar, ybar],
            color=seg_color,
            linewidth=1.6,
            solid_capstyle="round",
            linestyle="dashdot",
            zorder=4,
            path_effects=[
                pe.Stroke(linewidth=3.8, foreground="white"),
                pe.Normal(),
            ],
        )
        if first_box:
            ax.plot(
                [],
                [],
                color=box_edge,
                linewidth=1.6,
                label=r"PICPIs $\{I_j\}$",
            )
            first_box = False
        if first_seg:
            ax.plot(
                [],
                [],
                color=seg_color,
                linewidth=1.6,
                linestyle="dashdot",
                label=r"$\mathbb{P}(Y{=}1\mid p(x)\in I_j)$",
            )
            first_seg = False

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.4")
    ax.set_xlabel(r"Predicted probability $p(x)$", fontsize=10)
    ax.set_ylabel("Observed probability", fontsize=10)
    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
        framealpha=0.92,
        fancybox=True,
        edgecolor="0.75",
    )
    fig.tight_layout()
    return _save(fig, output_dir / "picpi_teaser_green.pdf")


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


def plot_thm(
    df_results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Paper ``fig:thm_box_rate``: boxplot and width-vs-rate panels."""
    output_dir = Path(output_dir) if output_dir is not None else figures_dir()
    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(summary["rate_term"], summary["width_mean_mean"], "o-", label="Mean width")
    ax.plot(
        summary["rate_term"], summary["width_median_mean"], "o-", label="Median width"
    )
    ax.set_xlabel(r"Reference rate $(\log n / n)^{1/3}$")
    ax.set_ylabel("Interval width")
    ax.set_title(r"Width vs. $(\log n / n)^{1/3}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    rate_path = _save(fig, output_dir / "thm_width_vs_rate.pdf")

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
    ax.set_xticklabels(
        [format_n(int(n)) for n in n_calib_list], rotation=30, ha="right"
    )
    ax.set_xlabel("Calibration sample size n")
    ax.set_ylabel("Mean interval width per repetition")
    ax.set_title("Width shrinkage with increasing n")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(
        handles=[
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
        ],
        loc="upper right",
    )
    fig.tight_layout()
    box_path = _save(fig, output_dir / "thm_width_mean_boxplot.pdf")
    return box_path, rate_path


def plot_empirical_mode(
    summary: pd.DataFrame,
    width_summary: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Paper ``fig:emp_mode_diagnostics``."""
    output_dir = Path(output_dir) if output_dir is not None else figures_dir()
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
    pass_path = _save(fig, output_dir / "heldout_picpi_failure_rate_by_method.pdf")

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
        "Interval frequency\n(conditioned on sample size)", fontsize=14
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

    y_tick_positions = np.unique(
        np.linspace(
            0, len(interval_lengths) - 1, num=min(10, len(interval_lengths))
        ).astype(int)
    )
    for ax in axes:
        ax.set_xlabel("Calibration sample size", fontsize=16)
        ax.set_ylabel("Returned interval length", fontsize=16)
        ax.set_xticks(np.arange(len(calibration_sizes)))
        ax.set_xticklabels(
            [f"{int(n):,}" for n in calibration_sizes], rotation=30, ha="right"
        )
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(
            [f"{interval_lengths[position]:.3g}" for position in y_tick_positions]
        )
        ax.tick_params(axis="both", labelsize=12)
    axes[0].set_title("Interval-length frequency", fontsize=18)
    axes[1].set_title("Held-out PICPI property pass rate", fontsize=18)
    heat_path = _save(
        fig, output_dir / "heldout_picpi_pass_rate_by_interval_length.pdf"
    )
    return pass_path, heat_path


def plot_task1_visualization(
    univariate: dict[str, object], output_dir: Path | None = None
) -> Path:
    """Paper ``fig:task1_interval_visualization``."""
    output_dir = Path(output_dir) if output_dir is not None else figures_dir()
    plt.style.use("seaborn-v0_8")
    x_sorted = univariate["x"]
    p_star_sorted = univariate["p_star"]
    p_hat_sorted = univariate["p_hat"]
    intervals_by_method = univariate["intervals"]
    summary_lookup = univariate["summary"].set_index("Method")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    range_interval_methods = {"Calibration-based interval", "Fixed-width binning"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for plot_idx, method in enumerate(TASK1_METHOD_ORDER):
        ax = axes[plot_idx]
        intervals = intervals_by_method[method]
        metrics = summary_lookup.loc[method]
        if method in range_interval_methods:
            rounded_intervals = np.round(intervals, 12)
            start = 0
            first_patch = True
            for idx in range(1, len(x_sorted) + 1):
                run_ended = idx == len(x_sorted) or not np.array_equal(
                    rounded_intervals[idx], rounded_intervals[start]
                )
                if not run_ended:
                    continue
                x_left = x_sorted[start]
                x_right = x_sorted[idx - 1]
                y_low, y_high = rounded_intervals[start]
                if x_left == x_right:
                    ax.vlines(
                        x_left,
                        y_low,
                        y_high,
                        color=colors[plot_idx],
                        alpha=0.35,
                        linewidth=1.0,
                        label="interval" if first_patch else None,
                    )
                else:
                    ax.fill_between(
                        [x_left, x_right],
                        [y_low, y_low],
                        [y_high, y_high],
                        alpha=0.30,
                        color=colors[plot_idx],
                        label="interval" if first_patch else None,
                    )
                first_patch = False
                start = idx
        else:
            ax.fill_between(
                x_sorted,
                intervals[:, 0],
                intervals[:, 1],
                alpha=0.30,
                color=colors[plot_idx],
                label="interval",
            )
        ax.plot(
            x_sorted,
            p_star_sorted,
            color="black",
            linewidth=1.6,
            label=r"true probability $p^*(x)$",
        )
        ax.plot(
            x_sorted,
            p_hat_sorted,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=r"predicted probability $\hat p(x)$",
        )
        ax.set_title(method, fontsize=18)
        ax.set_xlabel("x")
        ax.text(
            0.97,
            0.04,
            rf"avg length = {metrics['Average length']:.3f}, coverage of true prob = {metrics['Coverage p*']:.1%}",
            transform=ax.transAxes,
            fontsize=12,
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=12)
    axes[-1].axis("off")
    fig.tight_layout()
    return _save(fig, output_dir / "task1_interval_visualization.pdf")


def plot_task1_tree(tree: dict[str, object], output_dir: Path | None = None) -> Path:
    """Paper ``fig:task1_misspecified_tree``."""
    output_dir = Path(output_dir) if output_dir is not None else figures_dir()
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(
        tree["x"],
        tree["p_star"],
        label=r"true probability $p^*(x)$",
        linewidth=2.0,
    )
    ax.step(
        tree["x"],
        tree["p_hat"],
        where="mid",
        label=r"predicted probability $\hat p(x)$",
        linewidth=1.8,
    )
    ax.step(
        tree["x"],
        tree["intervals"][:, 1],
        where="mid",
        linestyle="--",
        label="PICPI upper endpoint",
    )
    ax.step(
        tree["x"],
        tree["intervals"][:, 0],
        where="mid",
        linestyle="--",
        label="PICPI lower endpoint",
    )
    ax.fill_between(
        tree["x"],
        tree["intervals"][:, 0],
        tree["intervals"][:, 1],
        step="mid",
        alpha=0.18,
        label="PICPI interval",
    )
    ax.set_xlabel("x")
    ax.set_title("Misspecified point estimator: decision tree", fontsize=15)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, output_dir / "task1_misspecified_point_estimator_decision_tree.pdf")


def _format_mean_sd(mean: float, sd: float, percent: bool = False) -> str:
    if np.isnan(mean):
        return "--"
    if percent:
        return f"{100 * mean:.2f}\\% $\\pm$ {100 * sd:.2f}\\%"
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def write_task1_table(
    mc_summary: pd.DataFrame, output_dir: Path | None = None
) -> Path:
    """Paper ``tab:task1_multivariate_mc_summary``."""
    output_dir = Path(output_dir) if output_dir is not None else figures_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    min_ece_mean = float(mc_summary["ece_mean"].min())
    rows = []
    for _, row in mc_summary.iterrows():
        rows.append(
            {
                "Method": row["Method"],
                "Avg Length": _format_mean_sd(
                    row["average_length_mean"], row["average_length_sd"]
                ),
                "Relative ECE": f"{(row['ece_mean'] / min_ece_mean):.2f}",
                "Coverage [0,1]": _format_mean_sd(
                    row["coverage_01_mean"], row["coverage_01_sd"], percent=True
                ),
                "Coverage p*": _format_mean_sd(
                    row["coverage_pstar_mean"],
                    row["coverage_pstar_sd"],
                    percent=True,
                ),
            }
        )
    display = pd.DataFrame(rows)
    lines = [
        "\\scriptsize",
        "\\begin{tabular}{lcccc}",
        "",
        "\\toprule",
        "Method & Avg Length & Relative ECE & Coverage [0,1] & Coverage p* \\\\",
        "\\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"{row['Method']} & {row['Avg Length']} & {row['Relative ECE']} & "
            f"{row['Coverage [0,1]']} & {row['Coverage p*']} \\\\"
        )
    lines.extend(["\\bottomrule", "", "\\end{tabular}"])
    path = output_dir / "task1_multivariate_mc_summary.tex"
    path.write_text("\n".join(lines) + "\n")
    display.to_csv(output_dir / "task1_multivariate_mc_summary.csv", index=False)
    return path


def load_task2_bundles(results_dir: Path) -> dict[str, dict[str, object]]:
    bundles = {}
    for dgp_name in DGP_DISPLAY_ORDER:
        dgp_dir = Path(results_dir) / dgp_name
        config = json.loads((dgp_dir / "config.json").read_text())
        bundles[dgp_name] = {
            "summary": pd.read_csv(dgp_dir / "summary.csv"),
            "config": config,
        }
    return bundles


def plot_task2(
    bundles: dict[str, dict[str, object]], output_dir: Path | None = None
) -> Path:
    """Paper ``fig:task2``: ``task2_multiclass_dgp_sweep.pdf``."""
    output_dir = Path(output_dir) if output_dir is not None else figures_dir()
    label_fontsize = 24
    title_fontsize = 24
    tick_fontsize = 18
    legend_fontsize = 28

    fig, axes = plt.subplots(
        len(DGP_DISPLAY_ORDER),
        3,
        figsize=(24, 4.2 * len(DGP_DISPLAY_ORDER) * 1.5),
        squeeze=False,
    )
    column_titles = [
        "Worst-class coverage vs. average set size",
        "Average class coverage by target coverage",
        "Worst-class coverage by target coverage",
    ]
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=title_fontsize, pad=24)

    for row_idx, dgp_name in enumerate(DGP_DISPLAY_ORDER):
        summary_df = bundles[dgp_name]["summary"].sort_values(
            ["method", "target_coverage"]
        )
        signature = bundles[dgp_name]["config"]["study_signature"]
        reference_coverage = 1.0 - float(signature["alpha"])
        row_axes = axes[row_idx]
        for method in CURVE_METHOD_ORDER:
            method_df = summary_df[summary_df["method"] == method].sort_values(
                "target_coverage"
            )
            if method_df.empty:
                continue
            style = dict(CURVE_METHOD_STYLES[method])
            row_axes[0].plot(
                method_df["mean_worst_coverage"],
                method_df["mean_avg_set_size"],
                label=method,
                **style,
            )
            row_axes[1].plot(
                method_df["target_coverage"],
                method_df["mean_macro_coverage"],
                **style,
            )
            row_axes[2].plot(
                method_df["target_coverage"],
                method_df["mean_worst_coverage"],
                **style,
            )
        for ax in row_axes:
            ax.axvline(reference_coverage, color="grey", linestyle="--", linewidth=1)
        row_axes[1].axhline(reference_coverage, color="grey", linestyle="--", linewidth=1)
        row_axes[2].axhline(reference_coverage, color="grey", linestyle="--", linewidth=1)
        row_axes[1].plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle=":", linewidth=1)
        row_axes[2].plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle=":", linewidth=1)
        row_axes[0].set_xlabel("Worst-class test coverage")
        row_axes[0].set_ylabel("Average prediction-set size", labelpad=18)
        row_axes[1].set_xlabel("Target coverage")
        row_axes[1].set_ylabel("Average class test coverage", labelpad=18)
        row_axes[2].set_xlabel("Target coverage")
        row_axes[2].set_ylabel("Worst-class test coverage", labelpad=18)
        for ax in row_axes:
            ax.grid(alpha=0.3)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.xaxis.label.set_size(label_fontsize)
            ax.yaxis.label.set_size(label_fontsize)
        row_axes[0].text(
            -0.22,
            0.5,
            str(signature["dgp_title"]),
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=title_fontsize,
            fontweight="bold",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=legend_fontsize,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.04, 0.03, 1.0, 0.84), w_pad=6.0, h_pad=6.0)
    return _save(fig, output_dir / "task2_multiclass_dgp_sweep.pdf")
