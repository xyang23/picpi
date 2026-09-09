"""Teaser DGP and plotting support for ``scripts/example_use.py``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

DGP_BETA_LIN = 1.6
DGP_BETA_QUAD = -0.25
DGP_INTERCEPT = 0.0
DGP_NOISE_SD = 1.0
N_TRAIN = 2_000
N_CALIB = 2_000
N_EVAL = 100
MC_SAMPLES = 256
MC_SEED = 12_345


def noisy_probability(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw probabilities from the paper Teaser's data-generating process."""
    z = (
        DGP_INTERCEPT
        + DGP_BETA_LIN * x
        + DGP_BETA_QUAD * x**2
        + rng.normal(0.0, DGP_NOISE_SD, size=x.shape)
    )
    return 1.0 / (1.0 + np.exp(-z))


def sample_binary_data(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw features and outcomes from the paper Teaser's DGP."""
    features = rng.uniform(-2.0, 2.0, size=n)
    outcomes = rng.binomial(1, noisy_probability(features, rng))
    return features, outcomes


def conditional_true_probability(x: np.ndarray, seed: int = MC_SEED) -> np.ndarray:
    """Approximate the Teaser DGP's true probability after averaging over noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, DGP_NOISE_SD, size=(MC_SAMPLES, x.size))
    z = (
        DGP_INTERCEPT
        + DGP_BETA_LIN * x[None, :]
        + DGP_BETA_QUAD * x[None, :] ** 2
        + noise
    )
    return (1.0 / (1.0 + np.exp(-z))).mean(axis=0)


def make_teaser_example_data(
    seed: int = 7,
) -> tuple[
    LogisticRegression,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit the Teaser model and return it with calibration/evaluation arrays."""
    rng = np.random.default_rng(seed)
    x_train, y_train = sample_binary_data(rng, N_TRAIN)
    x_calib, y_calib = sample_binary_data(rng, N_CALIB)
    x_eval, _ = sample_binary_data(rng, N_EVAL)

    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(x_train.reshape(-1, 1), y_train)
    predicted_probabilities = model.predict_proba(x_calib.reshape(-1, 1))[:, 1]
    evaluation_probabilities = model.predict_proba(x_eval.reshape(-1, 1))[:, 1]
    true_evaluation_probabilities = conditional_true_probability(x_eval)
    return (
        model,
        predicted_probabilities,
        y_calib,
        evaluation_probabilities,
        true_evaluation_probabilities,
    )


def observed_conditional_means(
    predicted_probabilities: np.ndarray,
    observed_outcomes: np.ndarray,
    intervals: list[tuple[float, float]],
    *,
    mode: str,
) -> np.ndarray:
    """Compute the calibration-sample mean outcome within every PICPI."""
    means = []
    for left, right in intervals:
        if mode == "empirical":
            selected = (predicted_probabilities > left) & (
                predicted_probabilities <= right
            )
        else:
            selected = (predicted_probabilities >= left) & (
                predicted_probabilities <= right
            )
        means.append(float(observed_outcomes[selected].mean()))
    return np.asarray(means)


def save_example_figure(
    *,
    mode: str,
    intervals: list[tuple[float, float]],
    predicted_probabilities: np.ndarray,
    observed_outcomes: np.ndarray,
    evaluation_probabilities: np.ndarray,
    true_evaluation_probabilities: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save the probability-model and PICPI panels for one mode."""
    conditional_means = observed_conditional_means(
        predicted_probabilities,
        observed_outcomes,
        intervals,
        mode=mode,
    )
    fig, (model_ax, interval_ax) = plt.subplots(1, 2, figsize=(11, 5.2))

    model_ax.scatter(
        evaluation_probabilities,
        true_evaluation_probabilities,
        s=18,
        alpha=0.55,
        facecolors="none",
        edgecolors="tab:blue",
    )
    model_ax.plot([0, 1], [0, 1], "--", color="0.35", label="Perfect calibration")
    model_ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted probability",
        ylabel="True probability",
        title="Probability model",
    )
    model_ax.set_aspect("equal", adjustable="box")
    model_ax.grid(alpha=0.25)
    model_ax.legend()

    positions = np.arange(1, len(intervals) + 1)
    for position, (left, right), mean in zip(
        positions, intervals, conditional_means
    ):
        interval_ax.plot(
            [left, right],
            [position, position],
            color="tab:blue",
            linewidth=5,
            alpha=0.55,
            solid_capstyle="round",
        )
        interval_ax.scatter(mean, position, color="tab:red", s=22, zorder=3)
    interval_ax.set(
        xlim=(-0.03, 1.03),
        xticks=np.linspace(0, 1, 6),
        xlabel="Probability scale",
        title=f"PICPI intervals: {mode} mode",
    )
    interval_ax.set_yticks([])
    interval_ax.grid(axis="x", alpha=0.25)
    interval_ax.plot([], [], color="tab:blue", linewidth=5, alpha=0.55, label="Interval")
    interval_ax.scatter(
        [], [], color="tab:red", s=22, label="Observed conditional mean outcome"
    )
    interval_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(f"PICPI example ({mode} mode)")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"picpi_{mode}_mode.pdf"
    png_path = output_dir / f"picpi_{mode}_mode.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path
