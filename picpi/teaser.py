"""Compute stored inputs for the paper teaser figure.

Original source: ``experiments/uncertainty_quantification/teaser.ipynb``.
Quadratic ground truth, linear logistic fit, empirical PICPI bins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from picpi.calibration import calibration
from picpi.paths import cached

SEED = 7
DGP_BETA_LIN = 1.6
DGP_BETA_QUAD = -0.25
DGP_INTERCEPT = 0.0
DGP_NOISE_SD = 1.0
N_TRAIN = 2000
N_CALIB = 2000
N_TEST = 100
DIAG_NUM_BINS = 10
MC_SAMPLES = 256
MC_SEED = 12345


def _p_star_noisy(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = (
        DGP_INTERCEPT
        + DGP_BETA_LIN * x
        + DGP_BETA_QUAD * (x**2)
        + rng.normal(0.0, DGP_NOISE_SD, size=x.shape)
    )
    return 1.0 / (1.0 + np.exp(-z))


def p_star_conditional_mc(x: np.ndarray, seed: int = MC_SEED) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    local_rng = np.random.default_rng(seed)
    eps = local_rng.normal(0.0, DGP_NOISE_SD, size=(MC_SAMPLES, x.size))
    z = (
        DGP_INTERCEPT
        + DGP_BETA_LIN * x[None, :]
        + DGP_BETA_QUAD * (x[None, :] ** 2)
        + eps
    )
    return (1.0 / (1.0 + np.exp(-z))).mean(axis=0)


def summarize_intervals(
    score: np.ndarray, y_ind: np.ndarray, intervals: list[tuple[float, float]]
) -> pd.DataFrame:
    score = np.asarray(score, dtype=float)
    y_ind = np.asarray(y_ind, dtype=int)
    rows = []
    for a, b in intervals:
        mask = (score > a) & (score <= b)
        n = int(mask.sum())
        rows.append(
            {
                "a": float(a),
                "b": float(b),
                "width": float(b - a),
                "n": n,
                "ybar": float(y_ind[mask].mean()) if n > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_teaser(seed: int = SEED) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    def sample_data(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.uniform(-2, 2, size=n)
        p = _p_star_noisy(x, rng)
        y = rng.binomial(1, p)
        return x, y

    x_tr, y_tr = sample_data(N_TRAIN)
    x_ca, y_ca = sample_data(N_CALIB)
    x_te, _y_te = sample_data(N_TEST)

    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(x_tr.reshape(-1, 1), y_tr)
    p_ca = model.predict_proba(x_ca.reshape(-1, 1))[:, 1]
    p_te = model.predict_proba(x_te.reshape(-1, 1))[:, 1]

    intervals = calibration(
        list(zip(p_ca.tolist(), (y_ca == 1).astype(int).tolist())),
        p_ca.tolist(),
        num_bin=DIAG_NUM_BINS,
        mode="empirical",
    )
    summary = summarize_intervals(p_ca, (y_ca == 1).astype(int), intervals)
    summary = summary.loc[summary["n"] > 0].sort_values("a").reset_index(drop=True)
    p_true_te = p_star_conditional_mc(x_te)
    return {
        "p_te": np.asarray(p_te, dtype=float),
        "p_true_te": np.asarray(p_true_te, dtype=float),
        "intervals": summary,
    }


def save_teaser(payload: dict[str, object], output_dir: Path | None = None) -> Path:
    output_dir = Path(output_dir) if output_dir is not None else cached("teaser")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "scatter.npz",
        p_te=payload["p_te"],
        p_true_te=payload["p_true_te"],
    )
    payload["intervals"].to_csv(output_dir / "intervals.csv", index=False)
    return output_dir


def load_teaser(output_dir: Path | None = None) -> dict[str, object]:
    output_dir = Path(output_dir) if output_dir is not None else cached("teaser")
    scatter = np.load(output_dir / "scatter.npz")
    return {
        "p_te": scatter["p_te"],
        "p_true_te": scatter["p_true_te"],
        "intervals": pd.read_csv(output_dir / "intervals.csv"),
    }
