"""Full Task 2 multiclass DGP-sweep implementation and helper functions.

The user-facing ``scripts/task2.py`` wrapper supplies the paper configuration.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

# Native math libraries (Apple Accelerate / OpenBLAS for NumPy & scikit-learn,
# plus the OpenMP runtime used by SciPy's HiGHS MILP backend) default to one
# thread per physical core. When several per-seed worker processes run at once
# that oversubscription can crash a worker on this macOS arm64 stack, which then
# deadlocks the parent pool because it waits forever for a result that never
# arrives. Pinning every native thread pool to a single thread removes the
# oversubscription. These MUST be set before NumPy/SciPy/sklearn are imported,
# and spawned worker processes inherit them from this parent environment.
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_env_var in THREAD_LIMIT_ENV_VARS:
    os.environ.setdefault(_thread_env_var, "1")

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

try:
    from scipy.optimize import Bounds, LinearConstraint, milp

    SCIPY_HAS_MILP = True
except Exception:
    SCIPY_HAS_MILP = False


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next(
    (path for path in [THIS_FILE.parent] + list(THIS_FILE.parents) if (path / "pyproject.toml").exists()),
    THIS_FILE.parent,
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from picpi.calibration import calibration  # noqa: E402


N_CLASSES = 4
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 17
DEFAULT_MC_REPS = 200
DEFAULT_N_TRAIN = 2000
DEFAULT_N_CAL = 2000
DEFAULT_N_EVAL = 2000
DEFAULT_NUM_BINS_PICPI = 100
DEFAULT_BASELINE_NUM_BINS = 100
DEFAULT_N_WORKERS = 4
SEED_STEP = 100
SPLIT_CONFORMAL_MAX_INTERVALS_PER_CLASS = 200
DEFAULT_COVERAGE_SWEEP_TARGETS = tuple(np.linspace(0.00, 1.00, 16))
SMOKE_COVERAGE_SWEEP_TARGETS = (0.25, 0.50, 0.75, 0.95)
DEFAULT_RESULTS_DIR = REPO_ROOT / "cached_results" / "task2"
MILP_SOLVE_LOCK = threading.Lock()
# Workers persist for the whole pool. Do NOT recycle with max_tasks_per_child:
# on POSIX that leaks the result-pipe FDs to replacement workers, so when a
# worker later dies the parent never sees EOF and deadlocks instead of raising
# BrokenProcessPool. Kept as a documented config value (None == no recycling).
MULTIPROCESSING_MAX_TASKS_PER_CHILD = None
# How many times to rebuild the pool after a worker crash before the run gives
# up. The crash is the concurrency-induced HiGHS MILP segfault, so each retry
# uses fewer workers; the last retries use a single isolated worker, which never
# crashes in practice and guarantees the run terminates.
MAX_POOL_ATTEMPTS = 8
# Hard wall-clock cap so a pathological HiGHS MILP instance can never hang a
# worker forever; on timeout we fall back to the (bounded) DP solver.
MILP_TIME_LIMIT_SECONDS = 30.0
# State resolution for the vectorised DP solver. The coverage axis is
# discretised into this many buckets relative to the (per-call) threshold, so the
# DP arrays stay this size regardless of the raw contribution magnitudes. 4000
# buckets reproduces the HiGHS MILP selections in practice while keeping each
# solve at ~1 ms.
DP_TARGET_STATES = 4000
# Default min-mass-cover solver. "dp" is a pure-NumPy exact dynamic program that
# cannot segfault and is therefore safe under process parallelism. "milp" uses
# SciPy's HiGHS backend (exact, but its native code segfaults when several solver
# processes run concurrently on this macOS arm64 stack -- use only with
# --n-workers 1 or the thread backend).
DEFAULT_SOLVER = "dp"
# Process-local active solver. Each (spawned) worker resets this from cfg.solver
# at the start of its per-seed job, so the choice propagates without threading an
# extra argument through every solver call.
_ACTIVE_SOLVER = DEFAULT_SOLVER

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
    "Split conformal prediction": {"color": "tab:green", "marker": "D", "linewidth": 2.0, "markersize": 5},
    "Calibration-based interval": {"color": "tab:red", "marker": "P", "linewidth": 2.0, "markersize": 5},
    "Fixed-width binning": {"color": "tab:purple", "marker": "s", "linewidth": 2.0, "markersize": 5},
    "Conformal classification": {"color": "tab:brown", "marker": "X", "linewidth": 2.0, "markersize": 5},
    "Simultaneous CI": {"color": "tab:blue", "marker": "^", "linewidth": 2.0, "markersize": 5},
}


@dataclass(frozen=True)
class SweepConfig:
    seed: int = DEFAULT_SEED
    alpha: float = DEFAULT_ALPHA
    mc_reps: int = DEFAULT_MC_REPS
    n_train: int = DEFAULT_N_TRAIN
    n_cal: int = DEFAULT_N_CAL
    n_eval: int = DEFAULT_N_EVAL
    n_classes: int = N_CLASSES
    num_bins_picpi: int = DEFAULT_NUM_BINS_PICPI
    baseline_num_bins: int = DEFAULT_BASELINE_NUM_BINS
    n_workers: int = DEFAULT_N_WORKERS
    parallel_backend: str = "multiprocessing"
    solver: str = DEFAULT_SOLVER
    coverage_sweep_targets: tuple[float, ...] = DEFAULT_COVERAGE_SWEEP_TARGETS
    split_conformal_max_intervals_per_class: int = SPLIT_CONFORMAL_MAX_INTERVALS_PER_CLASS


@dataclass(frozen=True)
class DGPDefinition:
    name: str
    title: str
    n_features: int
    description: str
    feature_distribution: str
    logit_description: str
    sample_features: Callable[[np.random.Generator, int], np.ndarray]
    true_class_probabilities: Callable[[np.ndarray], np.ndarray]
    metadata: dict[str, object]


DEFAULT_CONFIG = SweepConfig()


def plugin_interval_configs(cfg: SweepConfig) -> tuple[tuple[str, str, int | None], ...]:
    """List the plug-in interval variants enabled by the sweep configuration."""
    return (
        ("PICPI", "picpi", cfg.num_bins_picpi),
        ("Split conformal prediction", "split_conformal_neighborhood", None),
        ("Calibration-based interval", "calibration_based_interval", cfg.baseline_num_bins),
        ("Fixed-width binning", "fixed_bins", cfg.baseline_num_bins),
    )


def effective_parallel_backend(requested_backend: str, n_workers: int) -> str:
    """Resolve automatic parallelism to a concrete execution backend."""
    if n_workers <= 1:
        return "serial"
    if requested_backend == "threads":
        return "threads"
    # "processes" and "multiprocessing" both map to the crash-resilient,
    # spawn-based process pool.
    return "multiprocessing"


def ensure_2d_array(x: np.ndarray, expected_features: int | None = None) -> np.ndarray:
    """Normalize features to a two-dimensional array and validate its width."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("Expected a 1D or 2D feature array.")
    if expected_features is not None and x.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {x.shape[1]}.")
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert row-wise logits into numerically stable class probabilities."""
    logits = np.asarray(logits, dtype=float)
    centered = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(centered)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


UNIVARIATE_SLOPES = np.array([-1.8, -0.6, 0.6, 1.8], dtype=float)
UNIVARIATE_BIASES = np.array([0.25, 0.00, -0.05, -0.20], dtype=float)


def sample_univariate_softmax_features(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw features for the univariate softmax DGP."""
    return rng.uniform(-2.0, 2.0, size=(n, 1))


def univariate_softmax_probabilities(x: np.ndarray) -> np.ndarray:
    """Evaluate class probabilities for the univariate softmax DGP."""
    x = ensure_2d_array(x, expected_features=1)[:, 0]
    logits = x[:, None] * UNIVARIATE_SLOPES[None, :] + UNIVARIATE_BIASES[None, :]
    return softmax(logits)


GAUSSIAN_LINEAR_W = np.array(
    [
        [1.4, -0.9, 0.5],
        [-1.1, 0.8, 0.7],
        [0.6, 1.2, -1.0],
        [-0.3, -0.7, 1.1],
    ],
    dtype=float,
)
GAUSSIAN_LINEAR_B = np.array([0.4, -0.1, 0.2, -0.3], dtype=float)


def sample_gaussian_linear_d3_features(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw three-dimensional Gaussian features for the linear DGP."""
    return rng.normal(size=(n, 3))


def gaussian_linear_d3_probabilities(x: np.ndarray) -> np.ndarray:
    """Evaluate class probabilities for the three-dimensional linear DGP."""
    x = ensure_2d_array(x, expected_features=3)
    logits = x @ GAUSSIAN_LINEAR_W.T + GAUSSIAN_LINEAR_B
    return softmax(logits)


GAUSSIAN_NONLINEAR_W = np.array(
    [
        [1.0, -0.7, 0.4, 0.2, -0.1],
        [-0.9, 0.8, 0.3, -0.4, 0.4],
        [0.4, 0.5, -0.8, 0.7, -0.3],
        [-0.2, -0.4, 0.6, -0.6, 0.9],
    ],
    dtype=float,
)
GAUSSIAN_NONLINEAR_B = np.array([0.3, -0.2, 0.1, -0.2], dtype=float)
GAUSSIAN_NONLINEAR_LOADINGS = np.array([0.55, -0.35, 0.10, -0.40], dtype=float)


def sample_gaussian_nonlinear_d5_features(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw five-dimensional Gaussian features for the nonlinear DGP."""
    return rng.normal(size=(n, 5))


def gaussian_nonlinear_d5_probabilities(x: np.ndarray) -> np.ndarray:
    """Evaluate class probabilities for the five-dimensional nonlinear DGP."""
    x = ensure_2d_array(x, expected_features=5)
    nonlinear_basis = (x[:, 0] ** 2 - 1.0)[:, None]
    logits = x @ GAUSSIAN_NONLINEAR_W.T + GAUSSIAN_NONLINEAR_B + nonlinear_basis * GAUSSIAN_NONLINEAR_LOADINGS[None, :]
    return softmax(logits)


DGP_COMPONENT_WEIGHTS = np.array([0.32, 0.24, 0.27, 0.17], dtype=float)
DGP_COMPONENT_MEANS = np.array(
    [
        [-1.4, 0.5, -0.9, 0.2, -0.3, 0.6, -0.5, 0.4, -0.7, 0.3],
        [-0.2, -1.0, 0.7, -0.6, 0.8, -0.4, 0.3, -0.5, 0.6, -0.8],
        [0.9, 0.6, -0.2, 1.1, -0.7, 0.4, 0.9, 0.7, -0.1, 0.5],
        [1.6, -0.3, 1.0, 0.5, 0.4, -0.8, -1.0, -0.6, 0.9, -0.4],
    ],
    dtype=float,
)
DGP_COMPONENT_SCALES = np.array(
    [
        [0.60, 0.45, 0.55, 0.40, 0.50, 0.35, 0.45, 0.42, 0.50, 0.38],
        [0.45, 0.65, 0.50, 0.60, 0.55, 0.40, 0.35, 0.48, 0.44, 0.57],
        [0.55, 0.50, 0.70, 0.45, 0.60, 0.50, 0.55, 0.52, 0.46, 0.50],
        [0.50, 0.40, 0.60, 0.55, 0.45, 0.65, 0.60, 0.58, 0.54, 0.47],
    ],
    dtype=float,
)
DGP_SHARED_LOADINGS = np.array(
    [
        [0.95, -0.55, 0.35, 0.45, -0.30, 0.40, -0.20, 0.32, -0.28, 0.24],
        [0.30, 0.60, -0.45, 0.20, 0.55, -0.35, 0.65, -0.25, 0.48, -0.42],
    ],
    dtype=float,
)
DGP_COMPONENT_DRIFTS = np.array(
    [
        [0.30, -0.20, 0.10, 0.00, 0.12, -0.08, 0.05, 0.07, -0.06, 0.04],
        [-0.10, 0.28, -0.12, 0.08, -0.06, 0.14, -0.04, -0.09, 0.10, -0.05],
        [0.08, -0.12, 0.26, -0.16, 0.10, -0.06, 0.18, 0.11, -0.04, 0.09],
        [-0.14, 0.06, -0.08, 0.24, -0.10, 0.20, -0.22, -0.05, 0.12, -0.13],
    ],
    dtype=float,
)
DGP_COMPONENT_OFFSETS = np.array(
    [
        [-0.12, 0.06, -0.04, 0.02, -0.05, 0.03, -0.02, 0.04, -0.03, 0.01],
        [0.05, -0.10, 0.08, -0.06, 0.09, -0.04, 0.03, -0.02, 0.05, -0.07],
        [0.10, 0.04, -0.06, 0.11, -0.08, 0.06, 0.12, 0.03, -0.09, 0.08],
        [0.16, -0.02, 0.10, 0.07, 0.05, -0.11, -0.14, -0.08, 0.07, -0.05],
    ],
    dtype=float,
)

DGP_PARAM_SEED = 314159
DGP_PARAM_RNG = np.random.default_rng(DGP_PARAM_SEED)
POPULATION_CLASS_AXIS = np.linspace(-1.3, 1.3, N_CLASSES, dtype=float)
POPULATION_N_FEATURES = 10
POPULATION_N_INTERACTIONS = 10
POPULATION_N_NONLINEAR = 10

TRUE_LINEAR_WEIGHTS = DGP_PARAM_RNG.normal(scale=0.35, size=(POPULATION_N_FEATURES, N_CLASSES))
TRUE_LINEAR_WEIGHTS += np.outer(
    np.array([0.95, -0.70, 0.60, 0.75, -0.55, 0.50, -0.45, 0.40, -0.35, 0.30], dtype=float),
    POPULATION_CLASS_AXIS,
)
TRUE_LINEAR_WEIGHTS += 0.15 * np.outer(
    np.sin(np.linspace(0.0, np.pi, POPULATION_N_FEATURES, dtype=float)),
    np.cos(np.linspace(0.0, 2.0 * np.pi, N_CLASSES, dtype=float)),
)

TRUE_QUADRATIC_WEIGHTS = DGP_PARAM_RNG.normal(scale=0.09, size=(POPULATION_N_FEATURES, N_CLASSES))
TRUE_QUADRATIC_WEIGHTS += 0.03 * np.outer(
    np.cos(np.linspace(0.0, np.pi, POPULATION_N_FEATURES, dtype=float)),
    POPULATION_CLASS_AXIS**2 - POPULATION_CLASS_AXIS.mean() ** 2,
)

TRUE_INTERACTION_WEIGHTS = DGP_PARAM_RNG.normal(scale=0.12, size=(POPULATION_N_INTERACTIONS, N_CLASSES))
TRUE_INTERACTION_WEIGHTS += 0.05 * np.outer(
    np.linspace(-1.0, 1.0, POPULATION_N_INTERACTIONS, dtype=float),
    np.sin(np.linspace(0.0, np.pi, N_CLASSES, dtype=float)),
)

TRUE_NONLINEAR_WEIGHTS = DGP_PARAM_RNG.normal(scale=0.11, size=(POPULATION_N_NONLINEAR, N_CLASSES))
TRUE_NONLINEAR_WEIGHTS += 0.04 * np.outer(
    np.cos(np.linspace(0.0, np.pi, POPULATION_N_NONLINEAR, dtype=float)),
    np.cos(np.linspace(0.0, 2.0 * np.pi, N_CLASSES, dtype=float)),
)

RBF_CENTERS = np.array(
    [
        [-1.0, 0.3, -0.8, 0.2, -0.4, 0.5, -0.2, 0.4, -0.6, 0.1],
        [0.4, -0.6, 0.5, -0.3, 0.6, -0.2, 0.7, -0.4, 0.3, -0.8],
        [1.2, 0.7, 0.1, 0.9, -0.5, 0.3, 1.0, 0.6, -0.2, 0.5],
    ],
    dtype=float,
)
RBF_INV_SCALES = np.array([0.55, 0.75, 0.60], dtype=float)
TRUE_RBF_WEIGHTS = DGP_PARAM_RNG.normal(scale=0.16, size=(len(RBF_CENTERS), N_CLASSES))

TRUE_CONTEXT_WEIGHTS = DGP_PARAM_RNG.normal(scale=0.08, size=(4, N_CLASSES))
TRUE_CONTEXT_WEIGHTS += 0.03 * np.outer(
    np.array([1.0, -0.8, 0.6, -0.4], dtype=float),
    POPULATION_CLASS_AXIS,
)

TRUE_BIASES = np.linspace(0.42, -0.42, N_CLASSES, dtype=float)
TRUE_BIASES += DGP_PARAM_RNG.normal(scale=0.05, size=N_CLASSES)
TRUE_BIASES -= TRUE_BIASES.mean()


def sample_population_prototype_features(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw features for the population-prototype DGP."""
    component_ids = rng.choice(len(DGP_COMPONENT_WEIGHTS), size=n, p=DGP_COMPONENT_WEIGHTS)
    x = np.zeros((n, POPULATION_N_FEATURES), dtype=float)
    for comp_id in range(len(DGP_COMPONENT_WEIGHTS)):
        mask = component_ids == comp_id
        m = int(mask.sum())
        if m == 0:
            continue

        z_shared = rng.normal(size=(m, 2))
        eps = rng.normal(size=(m, POPULATION_N_FEATURES))
        chi = rng.chisquare(df=7.0, size=(m, 1))
        t_scale = np.sqrt(7.0 / np.maximum(chi, 1e-6))
        hetero = 1.0 + 0.25 * np.tanh(z_shared[:, [0]] + 0.45 * z_shared[:, [1]])

        x_comp = (
            DGP_COMPONENT_MEANS[comp_id][None, :]
            + z_shared @ DGP_SHARED_LOADINGS
            + eps * t_scale * (DGP_COMPONENT_SCALES[comp_id][None, :] * hetero)
        )
        x_comp += 0.22 * np.sin(x_comp[:, [0]]) * DGP_COMPONENT_DRIFTS[comp_id][None, :]

        x_comp[:, 2] += 0.28 * (x_comp[:, 0] ** 2 - 1.0)
        x_comp[:, 4] += 0.16 * x_comp[:, 1] * x_comp[:, 3]
        x_comp[:, 6] += 0.14 * x_comp[:, 2] * x_comp[:, 5]
        x_comp[:, 7] += 0.18 * np.sin(x_comp[:, 0] * x_comp[:, 2])
        x_comp[:, 8] += 0.15 * x_comp[:, 4] * x_comp[:, 6]
        x_comp[:, 9] += 0.12 * (x_comp[:, 7] ** 2 - 1.0)
        x[mask] = x_comp

    x += DGP_COMPONENT_OFFSETS[component_ids]
    x[:, 5] += 0.20 * np.sign(x[:, 0]) * np.sqrt(np.abs(x[:, 0]) + 1e-6)
    x[:, 1] -= 0.15 * np.tanh(x[:, 6] - x[:, 4])
    x[:, 8] += 0.18 * np.sign(x[:, 7]) * np.sqrt(np.abs(x[:, 7]) + 1e-6)
    x[:, 9] -= 0.12 * np.tanh(x[:, 8] - x[:, 5])
    return x


def population_interactions(x: np.ndarray) -> np.ndarray:
    """Build interaction features used by the population DGP."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    return np.column_stack(
        [
            x[:, 0] * x[:, 1],
            x[:, 1] * x[:, 2],
            x[:, 2] * x[:, 3],
            x[:, 3] * x[:, 4],
            x[:, 4] * x[:, 5],
            x[:, 5] * x[:, 6],
            x[:, 6] * x[:, 7],
            x[:, 7] * x[:, 8],
            x[:, 8] * x[:, 9],
            x[:, 0] * x[:, 9],
        ]
    )


def population_nonlinear_basis(x: np.ndarray) -> np.ndarray:
    """Build nonlinear basis features used by the population DGP."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    return np.column_stack(
        [
            np.sin(0.9 * x[:, 0]),
            np.cos(0.7 * x[:, 2]),
            np.tanh(x[:, 1] * x[:, 4]),
            np.maximum(x[:, 3], 0.0),
            np.sign(x[:, 5]) * np.sqrt(np.abs(x[:, 5]) + 1e-6),
            np.sin(x[:, 6] * x[:, 0]),
            np.exp(-0.3 * x[:, 2] ** 2),
            np.cos(0.6 * x[:, 7]),
            np.tanh(x[:, 8] - x[:, 9]),
            np.exp(-0.25 * x[:, 7] ** 2),
        ]
    )


def population_rbf_basis(x: np.ndarray) -> np.ndarray:
    """Evaluate radial-basis features around the population prototypes."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    rbf_sq_dist = ((x[:, None, :] - RBF_CENTERS[None, :, :]) ** 2 * RBF_INV_SCALES[None, :, None]).sum(axis=2)
    return np.exp(-rbf_sq_dist)


def population_context_shift(x: np.ndarray) -> np.ndarray:
    """Compute the context-dependent class-logit shift."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    context_basis = np.column_stack(
        [
            np.linalg.norm(x[:, :5], axis=1),
            np.linalg.norm(x[:, 5:], axis=1),
            np.arctan2(x[:, 1], 1.0 + np.abs(x[:, 0])),
            np.arctan2(x[:, 8], 1.0 + np.abs(x[:, 9])),
        ]
    )
    return context_basis @ TRUE_CONTEXT_WEIGHTS


def population_temperature(x: np.ndarray) -> np.ndarray:
    """Compute the feature-dependent softmax temperature."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    return np.clip(
        0.86 + 0.22 * np.tanh(0.60 * x[:, 0] - 0.35 * x[:, 3] + 0.20 * x[:, 8]),
        0.65,
        1.40,
    )


def population_linear_raw_logits(x: np.ndarray) -> np.ndarray:
    """Compute the linear component of the population DGP logits."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    return x @ TRUE_LINEAR_WEIGHTS


def population_h_raw_logits(x: np.ndarray) -> np.ndarray:
    """Compute the higher-order component of the population DGP logits."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    return (
        (x**2) @ TRUE_QUADRATIC_WEIGHTS
        + population_interactions(x) @ TRUE_INTERACTION_WEIGHTS
        + population_nonlinear_basis(x) @ TRUE_NONLINEAR_WEIGHTS
        + population_rbf_basis(x) @ TRUE_RBF_WEIGHTS
        + population_context_shift(x)
    )


def population_nonlinear_probabilities(x: np.ndarray) -> np.ndarray:
    """Evaluate probabilities using only the nonlinear population component."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    logits = (population_linear_raw_logits(x) + TRUE_BIASES[None, :]) / population_temperature(x)[:, None]
    return softmax(logits)


def population_h_only_probabilities(x: np.ndarray) -> np.ndarray:
    """Evaluate probabilities using only the higher-order logit component."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    logits = (population_h_raw_logits(x) + TRUE_BIASES[None, :]) / population_temperature(x)[:, None]
    return softmax(logits)


def population_full_probabilities(x: np.ndarray) -> np.ndarray:
    """Evaluate probabilities under the complete population DGP."""
    x = ensure_2d_array(x, expected_features=POPULATION_N_FEATURES)
    logits = (
        population_linear_raw_logits(x)
        + population_h_raw_logits(x)
        + TRUE_BIASES[None, :]
    ) / population_temperature(x)[:, None]
    return softmax(logits)


TRUE_LINEAR_GAUSSIAN_D10_W = np.array(
    [
        [1.4, -1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.7, 0.0, 1.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.0, 0.0, -1.2, 1.0, 0.0, 0.0, 0.0],
        [-0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, -0.9, 1.2],
    ],
    dtype=float,
)
TRUE_LINEAR_GAUSSIAN_D10_B = np.array([0.25, -0.20, 0.15, -0.20], dtype=float)


def sample_true_linear_gaussian_d10_features(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw ten-dimensional Gaussian features for the true-linear DGP."""
    return rng.normal(size=(n, 10))


def true_linear_gaussian_d10_probabilities(x: np.ndarray) -> np.ndarray:
    """Evaluate probabilities for the ten-dimensional true-linear DGP."""
    x = ensure_2d_array(x, expected_features=10)
    logits = x @ TRUE_LINEAR_GAUSSIAN_D10_W.T + TRUE_LINEAR_GAUSSIAN_D10_B
    return softmax(logits)


DGP_REGISTRY: dict[str, DGPDefinition] = {
    "true_linear_gaussian_d10": DGPDefinition(
        name="true_linear_gaussian_d10",
        title="Linear-Gaussian DGP",
        n_features=10,
        description="Simple Gaussian 10d feature generator with a linear 4-class softmax DGP.",
        feature_distribution="X ~ N(0, I_10)",
        logit_description="logits(X) = X W + b",
        sample_features=sample_true_linear_gaussian_d10_features,
        true_class_probabilities=true_linear_gaussian_d10_probabilities,
        metadata={
            "structure": "true_linear",
            "weight_matrix": TRUE_LINEAR_GAUSSIAN_D10_W.tolist(),
            "biases": TRUE_LINEAR_GAUSSIAN_D10_B.tolist(),
            "temperature": "none",
        },
    ),
    "population_nonlinear": DGPDefinition(
        name="population_nonlinear",
        title="Nonlinear-Mixture DGP",
        n_features=POPULATION_N_FEATURES,
        description=
            "Linear raw logits (10d mixture feature generator), nonlinear x-dependent scaling denominator.",
        feature_distribution="10-dimensional four-component latent-factor mixture",
        logit_description="logits(X) = [X W_linear + b] / temperature(X)",
        sample_features=sample_population_prototype_features,
        true_class_probabilities=population_nonlinear_probabilities,
        metadata={
            "structure": "nonlinear",
            "component_weights": DGP_COMPONENT_WEIGHTS.tolist(),
            "component_means": DGP_COMPONENT_MEANS.tolist(),
            "component_scales": DGP_COMPONENT_SCALES.tolist(),
            "shared_loadings": DGP_SHARED_LOADINGS.tolist(),
            "component_drifts": DGP_COMPONENT_DRIFTS.tolist(),
            "component_offsets": DGP_COMPONENT_OFFSETS.tolist(),
            "linear_weights": TRUE_LINEAR_WEIGHTS.tolist(),
            "biases": TRUE_BIASES.tolist(),
            "temperature": "x-dependent",
        },
    ),
}

DGP_DISPLAY_ORDER = tuple(DGP_REGISTRY.keys())


def sample_multiclass_labels(rng: np.random.Generator, q: np.ndarray) -> np.ndarray:
    """Draw one class label per row of class probabilities."""
    q = np.asarray(q, dtype=float)
    cumulative = np.cumsum(q, axis=1)
    uniforms = rng.random(size=(q.shape[0], 1))
    return np.minimum((uniforms > cumulative).sum(axis=1), q.shape[1] - 1).astype(int)


def generate_split_data(cfg: SweepConfig, dgp: DGPDefinition, seed: int) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Generate training, calibration, and evaluation folds for one DGP."""
    rng = np.random.default_rng(seed)

    def _fold(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw features, labels, and true probabilities for one fold."""
        x = dgp.sample_features(rng, n)
        q = dgp.true_class_probabilities(x)
        y = sample_multiclass_labels(rng, q)
        return x, y, q

    return _fold(cfg.n_train), _fold(cfg.n_cal), _fold(cfg.n_eval)


def align_multiclass_probabilities(model: LogisticRegression, x: np.ndarray, n_classes: int) -> np.ndarray:
    """Align fitted probability columns with the complete class index."""
    probabilities = model.predict_proba(x)
    aligned = np.zeros((len(x), n_classes), dtype=float)
    for idx, class_id in enumerate(np.asarray(model.classes_, dtype=int)):
        aligned[:, class_id] = probabilities[:, idx]
    return aligned


def predict_binary_positive_proba(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    """Return positive-outcome probabilities, including constant-label fits."""
    x = ensure_2d_array(x)
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return np.zeros(x.shape[0], dtype=float)
    positive_index = int(np.where(classes == 1)[0][0])
    return np.asarray(probabilities[:, positive_index], dtype=float)


def fit_binary_ci_helper(x_train: np.ndarray, y_train_binary: np.ndarray) -> dict[str, object]:
    """Fit one binary model and retain quantities needed for confidence bands."""
    x_train = ensure_2d_array(x_train)
    y_train_binary = np.asarray(y_train_binary, dtype=int)
    unique_labels = np.unique(y_train_binary)
    if unique_labels.size < 2:
        return {
            "kind": "constant",
            "constant_prob": float(unique_labels[0]) if unique_labels.size else 0.0,
        }

    model = LogisticRegression(solver="lbfgs", max_iter=2000)
    model.fit(x_train, y_train_binary)
    p_hat_train = predict_binary_positive_proba(model, x_train)
    x_design = np.column_stack([np.ones(x_train.shape[0]), x_train])
    weights = p_hat_train * (1.0 - p_hat_train)
    fisher = x_design.T @ (x_design * weights[:, None])
    cov_beta = np.linalg.pinv(fisher)
    beta_hat = np.concatenate([model.intercept_.ravel(), model.coef_.ravel()])
    return {
        "kind": "logistic",
        "beta_hat": beta_hat,
        "cov_beta": cov_beta,
    }


def simultaneous_ci_intervals_from_helper(helper: dict[str, object], x_eval: np.ndarray, alpha: float) -> np.ndarray:
    """Construct simultaneous intervals from a fitted binary-model helper."""
    x_eval = ensure_2d_array(x_eval)
    if helper["kind"] == "constant":
        constant_prob = float(helper["constant_prob"])
        return np.column_stack(
            [
                np.full(x_eval.shape[0], constant_prob, dtype=float),
                np.full(x_eval.shape[0], constant_prob, dtype=float),
            ]
        )

    beta_hat = np.asarray(helper["beta_hat"], dtype=float)
    cov_beta = np.asarray(helper["cov_beta"], dtype=float)
    x_eval_design = np.column_stack([np.ones(x_eval.shape[0]), x_eval])
    eta_eval = x_eval_design @ beta_hat
    se_eta = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x_eval_design, cov_beta, x_eval_design), 0.0))
    c = np.sqrt(chi2.ppf(1.0 - float(alpha), df=x_eval_design.shape[1]))
    return np.column_stack([expit(eta_eval - c * se_eta), expit(eta_eval + c * se_eta)])


def fit_shared_model(cfg: SweepConfig, dgp: DGPDefinition, seed: int) -> dict[str, object]:
    """Fit the models and cache shared predictions for one seeded DGP split."""
    (x_train, y_train, q_train), (x_cal, y_cal, q_cal), (x_eval, y_eval, q_eval) = generate_split_data(cfg, dgp, seed=seed)
    x_train = ensure_2d_array(x_train, expected_features=dgp.n_features)
    x_cal = ensure_2d_array(x_cal, expected_features=dgp.n_features)
    x_eval = ensure_2d_array(x_eval, expected_features=dgp.n_features)

    model = LogisticRegression(solver="lbfgs", max_iter=2000)
    model.fit(x_train, y_train)

    binary_ci_helpers = {
        class_id: fit_binary_ci_helper(x_train, (y_train == class_id).astype(int))
        for class_id in range(cfg.n_classes)
    }

    return {
        "seed": seed,
        "dgp_name": dgp.name,
        "model": model,
        "binary_ci_helpers": binary_ci_helpers,
        "x_train": x_train,
        "y_train": np.asarray(y_train, dtype=int),
        "q_train": np.asarray(q_train, dtype=float),
        "x_cal": x_cal,
        "y_cal": np.asarray(y_cal, dtype=int),
        "q_cal": np.asarray(q_cal, dtype=float),
        "x_eval": x_eval,
        "y_eval": np.asarray(y_eval, dtype=int),
        "q_eval": np.asarray(q_eval, dtype=float),
        "p_hat_cal": align_multiclass_probabilities(model, x_cal, cfg.n_classes),
        "p_hat_eval": align_multiclass_probabilities(model, x_eval, cfg.n_classes),
    }


def _limit_worker_threads() -> None:
    """Worker-process initializer that pins native math libraries to one thread.

    The environment variables already propagate from the parent (set before the
    NumPy/SciPy import), but we re-apply them and also use ``threadpoolctl`` as a
    belt-and-suspenders runtime limit in case a backend ignored the env vars.
    """
    for var in THREAD_LIMIT_ENV_VARS:
        os.environ.setdefault(var, "1")
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def _workers_for_attempt(attempt: int, n_workers: int, n_pending: int) -> int:
    """Pick the worker count for a (re)attempt, degrading concurrency on retries.

    The only crash we recover from is the concurrency-induced HiGHS MILP segfault,
    so each successive retry lowers concurrency. The final attempts use a single
    isolated worker, which never reproduces the crash, guaranteeing termination.
    """
    if attempt >= MAX_POOL_ATTEMPTS - 1:
        workers = 1
    elif attempt >= MAX_POOL_ATTEMPTS - 3:
        workers = 2
    else:
        workers = n_workers
    return max(1, min(workers, n_pending))


def _run_with_process_pool(
    func: Callable[[int], pd.DataFrame],
    seeds: Sequence[int],
    desc: str,
    n_workers: int,
) -> dict[int, pd.DataFrame]:
    """Run per-seed jobs on a spawn-based process pool that survives crashes.

    ``concurrent.futures.ProcessPoolExecutor`` raises ``BrokenProcessPool`` (rather
    than deadlocking like ``multiprocessing.Pool.imap_unordered``) when a worker is
    killed -- e.g. by the SIGSEGV that HiGHS's MILP backend can hit under
    concurrency on this stack. We collect whatever finished, then rebuild the pool
    for the missing seeds with progressively fewer workers, ending on a single
    isolated worker so the run always completes.
    """
    ctx = mp.get_context("spawn")
    results: dict[int, pd.DataFrame] = {}
    with tqdm(total=len(seeds), desc=desc) as progress_bar:
        for attempt in range(1, MAX_POOL_ATTEMPTS + 1):
            pending = [seed for seed in seeds if seed not in results]
            if not pending:
                break
            workers = _workers_for_attempt(attempt, n_workers, len(pending))
            pool_broke = False
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=ctx,
                    initializer=_limit_worker_threads,
                ) as executor:
                    future_to_seed = {executor.submit(func, seed): seed for seed in pending}
                    for future in as_completed(future_to_seed):
                        seed = future_to_seed[future]
                        try:
                            results[seed] = future.result()
                        except BrokenProcessPool:
                            pool_broke = True
                            break
                        progress_bar.update(1)
            except BrokenProcessPool:
                pool_broke = True

            if pool_broke:
                still_pending = len([seed for seed in seeds if seed not in results])
                warnings.warn(
                    f"{desc}: a worker process died (likely the HiGHS MILP segfault); "
                    f"retrying {still_pending} remaining seed(s) with reduced "
                    f"concurrency (attempt {attempt}/{MAX_POOL_ATTEMPTS}).",
                    RuntimeWarning,
                )

    missing = [seed for seed in seeds if seed not in results]
    if missing:
        raise RuntimeError(
            f"{desc}: {len(missing)} seed(s) still failed after {MAX_POOL_ATTEMPTS} "
            f"pool attempts (including single-worker isolation): {missing[:10]}..."
        )
    return results


def run_parallel_seed_jobs(
    func: Callable[[int], pd.DataFrame],
    seeds: Sequence[int],
    desc: str,
    n_workers: int,
    *,
    backend: str,
) -> list[pd.DataFrame]:
    """Run seeded jobs with the requested backend while preserving seed order."""
    seeds = list(seeds)
    if len(seeds) == 0:
        return []
    backend = effective_parallel_backend(str(backend), n_workers)

    if backend == "serial":
        return [func(seed) for seed in tqdm(seeds, desc=desc)]

    if backend == "threads":
        results: dict[int, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_seed = {executor.submit(func, seed): seed for seed in seeds}
            for future in tqdm(as_completed(future_to_seed), total=len(seeds), desc=desc):
                results[future_to_seed[future]] = future.result()
        return [results[seed] for seed in seeds]

    results = _run_with_process_pool(func, seeds, desc, n_workers)
    return [results[seed] for seed in seeds]


def lower_tail_conformal_threshold(scores: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Return the finite-sample lower-tail conformal order statistic."""
    scores = np.sort(np.asarray(scores, dtype=float))
    k = int(np.ceil((len(scores) + 1) * float(alpha)))
    k = min(max(k, 1), len(scores))
    return float(scores[k - 1])


def score_in_interval(scores: np.ndarray, left: float, right: float) -> np.ndarray:
    """Test score membership in a right-closed probability interval."""
    scores = np.asarray(scores, dtype=float)
    return ((scores > left) & (scores <= right)) | ((left == 0.0) & (scores == 0.0))


def intervals_are_disjoint(intervals: Sequence[tuple[float, float]]) -> bool:
    """Check whether a collection of intervals has no interior overlap."""
    if len(intervals) <= 1:
        return True
    ordered = sorted(intervals)
    for (_, right_prev), (left_next, _) in zip(ordered[:-1], ordered[1:]):
        if right_prev > left_next:
            return False
    return True


def weighted_interval_scheduling(intervals: Sequence[tuple[float, float]], weights: np.ndarray) -> list[int]:
    """Select a maximum-weight nonoverlapping subset of intervals."""
    if len(intervals) == 0:
        return []

    order = np.argsort([right for _, right in intervals])
    sorted_intervals = [intervals[idx] for idx in order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    starts = np.array([left for left, _ in sorted_intervals], dtype=float)
    ends = np.array([right for _, right in sorted_intervals], dtype=float)

    previous = np.full(len(sorted_intervals), -1, dtype=int)
    for idx in range(len(sorted_intervals)):
        previous[idx] = np.searchsorted(ends, starts[idx], side="right") - 1

    best = np.zeros(len(sorted_intervals), dtype=float)
    choose = np.zeros(len(sorted_intervals), dtype=bool)
    for idx in range(len(sorted_intervals)):
        take = sorted_weights[idx] + (best[previous[idx]] if previous[idx] >= 0 else 0.0)
        skip = best[idx - 1] if idx > 0 else 0.0
        if take > skip:
            best[idx] = take
            choose[idx] = True
        else:
            best[idx] = skip

    selected = []
    idx = len(sorted_intervals) - 1
    while idx >= 0:
        if choose[idx]:
            selected.append(idx)
            idx = previous[idx]
        else:
            idx -= 1
    return [order[idx] for idx in reversed(selected)]


def select_disjoint_intervals(raw_intervals: np.ndarray | Sequence[tuple[float, float]], reference_scores: np.ndarray) -> list[tuple[float, float]]:
    """Deduplicate intervals and retain a high-mass disjoint subset."""
    intervals = sorted({(float(left), float(right)) for left, right in np.asarray(raw_intervals, dtype=float)})
    if len(intervals) == 0:
        return []

    weights = np.array(
        [float(score_in_interval(reference_scores, left, right).mean()) for left, right in intervals],
        dtype=float,
    )
    positive = weights > 0.0
    intervals = [intervals[idx] for idx in np.where(positive)[0]]
    weights = weights[positive]
    if len(intervals) == 0:
        return []
    if intervals_are_disjoint(intervals):
        return intervals

    selected_idx = weighted_interval_scheduling(intervals, weights)
    return [intervals[idx] for idx in selected_idx]


def calibration_tau_from_upper_matrix(upper_cal: np.ndarray, y_cal: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Calibrate an inclusion threshold from true-label upper endpoints."""
    true_label_scores = upper_cal[np.arange(len(y_cal)), y_cal]
    return lower_tail_conformal_threshold(true_label_scores, alpha=alpha)


def build_empirical_interval_stats(scores: np.ndarray, labels_binary: np.ndarray, num_bins: int) -> dict[str, object]:
    """Construct empirical PICPIs and their classwise calibration statistics."""
    scores = np.asarray(scores, dtype=float)
    labels_binary = np.asarray(labels_binary, dtype=int)
    raw_intervals = calibration(
        list(zip(scores.tolist(), labels_binary.tolist())),
        scores.tolist(),
        num_bin=num_bins,
        mode="empirical",
    )
    intervals = [(float(left), float(right)) for left, right in raw_intervals]
    a = np.array([left for left, _ in intervals], dtype=float)
    b = np.array([right for _, right in intervals], dtype=float)
    m_hat = np.array(
        [float(score_in_interval(scores, left, right).mean()) for left, right in intervals],
        dtype=float,
    )
    pi_hat = float(labels_binary.mean())
    t_hat = float(np.clip(1.0 - m_hat.sum(), 0.0, 1.0))
    return {
        "intervals": intervals,
        "a": a,
        "b": b,
        "m_hat": m_hat,
        "pi_hat": pi_hat,
        "t_hat": t_hat,
        "intervals_disjoint": intervals_are_disjoint(intervals),
    }


def build_fixed_bin_interval_stats(scores: np.ndarray, labels_binary: np.ndarray, num_bins: int) -> dict[str, object]:
    """Construct equal-width bins and their classwise calibration statistics."""
    scores = np.asarray(scores, dtype=float)
    labels_binary = np.asarray(labels_binary, dtype=int)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    intervals = [(float(edges[k]), float(edges[k + 1])) for k in range(num_bins)]
    a = np.array([left for left, _ in intervals], dtype=float)
    b = np.array([right for _, right in intervals], dtype=float)
    m_hat = np.array(
        [float(score_in_interval(scores, left, right).mean()) for left, right in intervals],
        dtype=float,
    )
    pi_hat = float(labels_binary.mean())
    t_hat = float(np.clip(1.0 - m_hat.sum(), 0.0, 1.0))
    return {
        "intervals": intervals,
        "a": a,
        "b": b,
        "m_hat": m_hat,
        "pi_hat": pi_hat,
        "t_hat": t_hat,
        "intervals_disjoint": intervals_are_disjoint(intervals),
        "interval_strategy": "Fixed-width binning",
    }


def upper_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample upper conformal quantile."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n_scores = len(scores)
    if n_scores == 0:
        return np.inf
    k = int(np.ceil((n_scores + 1) * (1.0 - float(alpha))))
    if k <= 0:
        return 0.0
    if k > n_scores:
        return np.inf
    return float(scores[k - 1])


def build_interval_stats_from_intervals(
    scores: np.ndarray,
    labels_binary: np.ndarray,
    intervals: Sequence[tuple[float, float]],
    strategy_name: str,
) -> dict[str, object]:
    """Compute calibration statistics for an explicit interval collection."""
    scores = np.asarray(scores, dtype=float)
    labels_binary = np.asarray(labels_binary, dtype=int)
    intervals = sorted({(float(left), float(right)) for left, right in intervals})
    a = np.array([left for left, _ in intervals], dtype=float)
    b = np.array([right for _, right in intervals], dtype=float)

    if len(intervals) == 0:
        m_hat = np.zeros(0, dtype=float)
        positive_rate = np.zeros(0, dtype=float)
        calibration_error = np.zeros(0, dtype=float)
    else:
        m_values: list[float] = []
        positive_rates: list[float] = []
        calibration_errors: list[float] = []
        for left, right in intervals:
            mask = score_in_interval(scores, left, right)
            m_values.append(float(mask.mean()))
            if mask.any():
                rate = float(labels_binary[mask].mean())
                midpoint = 0.5 * (float(left) + float(right))
                positive_rates.append(rate)
                calibration_errors.append(abs(rate - midpoint))
            else:
                positive_rates.append(np.nan)
                calibration_errors.append(np.inf)
        m_hat = np.array(m_values, dtype=float)
        positive_rate = np.array(positive_rates, dtype=float)
        calibration_error = np.array(calibration_errors, dtype=float)

    pi_hat = float(labels_binary.mean())
    t_hat = float(np.clip(1.0 - m_hat.sum(), 0.0, 1.0))
    finite_errors = np.where(np.isfinite(calibration_error), calibration_error, 0.0)
    interval_ece = float(np.sum(m_hat * finite_errors))
    return {
        "intervals": intervals,
        "a": a,
        "b": b,
        "m_hat": m_hat,
        "positive_rate": positive_rate,
        "calibration_error": calibration_error,
        "interval_ece": interval_ece,
        "pi_hat": pi_hat,
        "t_hat": t_hat,
        "intervals_disjoint": intervals_are_disjoint(intervals),
        "interval_strategy": strategy_name,
    }


def build_simultaneous_ci_interval_stats(state: dict[str, object], cfg: SweepConfig, alpha: float) -> dict[int, dict[str, object]]:
    """Build classwise statistics from simultaneous confidence intervals."""
    x_cal = np.asarray(state["x_cal"], dtype=float)
    y_cal = np.asarray(state["y_cal"], dtype=int)
    p_hat_cal = np.asarray(state["p_hat_cal"], dtype=float)
    binary_ci_helpers: dict[int, dict[str, object]] = state["binary_ci_helpers"]  # type: ignore[assignment]

    classwise_stats: dict[int, dict[str, object]] = {}
    for class_id in range(cfg.n_classes):
        labels_binary = (y_cal == class_id).astype(int)
        raw_intervals = simultaneous_ci_intervals_from_helper(binary_ci_helpers[class_id], x_cal, alpha=alpha)
        intervals = select_disjoint_intervals(raw_intervals, p_hat_cal[:, class_id])
        classwise_stats[class_id] = build_interval_stats_from_intervals(
            p_hat_cal[:, class_id],
            labels_binary,
            intervals,
            "Simultaneous CI",
        )
    return classwise_stats


def classwise_interval_ece_threshold(classwise_stats: dict[int, dict[str, object]]) -> float:
    """Average finite interval-ECE values across classes."""
    ece_values = [float(stats.get("interval_ece", 0.0)) for stats in classwise_stats.values()]
    finite_values = [value for value in ece_values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else np.inf


def choose_intervals_by_average_ece(errors: np.ndarray, masses: np.ndarray, target_ece: float) -> np.ndarray:
    """Select low-error intervals nearest a target average ECE."""
    errors = np.asarray(errors, dtype=float)
    masses = np.asarray(masses, dtype=float)
    eligible = np.isfinite(errors) & (masses > 0.0)
    selected = np.zeros(errors.shape, dtype=bool)
    if not eligible.any():
        return selected

    eligible_idx = np.where(eligible)[0]
    order = eligible_idx[np.argsort(errors[eligible_idx], kind="stable")]
    cumulative_mass = np.cumsum(masses[order])
    cumulative_error = np.cumsum(masses[order] * errors[order])
    average_ece = cumulative_error / cumulative_mass
    if np.isfinite(target_ece):
        best_pos = int(np.argmin(np.abs(average_ece - target_ece)))
    else:
        best_pos = len(order) - 1
    selected[order[: best_pos + 1]] = True
    return selected


def build_split_conformal_neighborhood_stats(
    state: dict[str, object],
    cfg: SweepConfig,
    alpha: float,
) -> dict[int, dict[str, object]]:
    """Build classwise neighborhoods using split-conformal residual widths."""
    p_hat_cal = np.asarray(state["p_hat_cal"], dtype=float)
    y_cal = np.asarray(state["y_cal"], dtype=int)
    seed = int(state["seed"])
    classwise_stats: dict[int, dict[str, object]] = {}

    for class_id in range(cfg.n_classes):
        scores = p_hat_cal[:, class_id]
        labels_binary = (y_cal == class_id).astype(int)
        residuals = np.abs(labels_binary - scores)
        width = upper_conformal_quantile(residuals, alpha=alpha)

        if np.isinf(width):
            intervals = [(0.0, 1.0)]
            center_count = 1
        elif width <= 0.0:
            intervals = []
            center_count = 0
        else:
            rng = np.random.default_rng(seed + 1009 * class_id)
            n_centers = min(cfg.split_conformal_max_intervals_per_class, len(scores))
            center_indices = rng.choice(len(scores), size=n_centers, replace=False)
            center_scores = scores[center_indices]
            intervals = [
                (max(0.0, float(score) - width), min(1.0, float(score) + width))
                for score in center_scores
            ]
            center_count = len(intervals)

        stats = build_interval_stats_from_intervals(
            scores,
            labels_binary,
            intervals,
            "Split conformal prediction",
        )
        stats["conformal_width"] = float(width)
        stats["alpha_for_width"] = float(alpha)
        stats["candidate_center_count"] = int(center_count)
        stats["candidate_center_cap"] = int(cfg.split_conformal_max_intervals_per_class)
        classwise_stats[class_id] = stats

    return classwise_stats


def build_calibration_based_interval_stats(
    state: dict[str, object],
    cfg: SweepConfig,
    num_bins: int | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    picpi_stats: dict[int, dict[str, object]] | None = None,
) -> dict[int, dict[str, object]]:
    """Build fixed-bin classwise intervals matched to PICPI calibration error."""
    del alpha  # kept for interface parity with the notebook implementation
    p_hat_cal = np.asarray(state["p_hat_cal"], dtype=float)
    y_cal = np.asarray(state["y_cal"], dtype=int)
    num_bins = cfg.baseline_num_bins if num_bins is None else int(num_bins)

    if picpi_stats is None:
        picpi_stats = {}
        for class_id in range(p_hat_cal.shape[1]):
            labels_binary = (y_cal == class_id).astype(int)
            raw_stats = build_empirical_interval_stats(
                p_hat_cal[:, class_id],
                labels_binary,
                num_bins=cfg.num_bins_picpi,
            )
            enriched = build_interval_stats_from_intervals(
                p_hat_cal[:, class_id],
                labels_binary,
                raw_stats["intervals"],
                "PICPI",
            )
            raw_stats.update(
                positive_rate=enriched["positive_rate"],
                calibration_error=enriched["calibration_error"],
                interval_ece=enriched["interval_ece"],
                interval_strategy="PICPI",
            )
            picpi_stats[class_id] = raw_stats

    target_ece = classwise_interval_ece_threshold(picpi_stats)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    candidate_intervals = [(float(edges[idx]), float(edges[idx + 1])) for idx in range(num_bins)]
    classwise_stats: dict[int, dict[str, object]] = {}

    for class_id in range(p_hat_cal.shape[1]):
        labels_binary = (y_cal == class_id).astype(int)
        candidate_stats = build_interval_stats_from_intervals(
            p_hat_cal[:, class_id],
            labels_binary,
            candidate_intervals,
            "Calibration-based interval",
        )
        errors = np.asarray(candidate_stats["calibration_error"], dtype=float)
        masses = np.asarray(candidate_stats["m_hat"], dtype=float)
        selected = choose_intervals_by_average_ece(errors, masses, target_ece)
        intervals = [candidate_intervals[idx] for idx in np.where(selected)[0]]
        stats = build_interval_stats_from_intervals(
            p_hat_cal[:, class_id],
            labels_binary,
            intervals,
            "Calibration-based interval",
        )
        selected_errors = np.asarray(stats["calibration_error"], dtype=float)
        selected_masses = np.asarray(stats["m_hat"], dtype=float)
        finite_selected = np.isfinite(selected_errors) & (selected_masses > 0.0)
        if finite_selected.any():
            selected_average_ece = float(
                np.sum(selected_masses[finite_selected] * selected_errors[finite_selected])
                / np.sum(selected_masses[finite_selected])
            )
        else:
            selected_average_ece = np.nan
        stats["target_interval_ece"] = float(target_ece)
        stats["selected_average_ece"] = selected_average_ece
        stats["reference_interval_ece"] = float(picpi_stats[class_id].get("interval_ece", np.nan))
        classwise_stats[class_id] = stats

    return classwise_stats


def build_classwise_interval_stats(
    state: dict[str, object],
    cfg: SweepConfig,
    strategy: str = "picpi",
    num_bins: int | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[int, dict[str, object]]:
    """Dispatch construction of classwise statistics for one interval strategy."""
    p_hat_cal = np.asarray(state["p_hat_cal"], dtype=float)
    y_cal = np.asarray(state["y_cal"], dtype=int)

    if strategy == "simultaneous_ci":
        return build_simultaneous_ci_interval_stats(state, cfg, alpha=alpha)
    if strategy == "split_conformal_neighborhood":
        return build_split_conformal_neighborhood_stats(state, cfg, alpha=alpha)
    if strategy == "calibration_based_interval":
        return build_calibration_based_interval_stats(state, cfg, num_bins=num_bins, alpha=alpha)

    classwise_stats: dict[int, dict[str, object]] = {}
    for class_id in range(p_hat_cal.shape[1]):
        labels_binary = (y_cal == class_id).astype(int)
        if strategy == "picpi":
            stats = build_empirical_interval_stats(
                p_hat_cal[:, class_id],
                labels_binary,
                num_bins=cfg.num_bins_picpi if num_bins is None else int(num_bins),
            )
            enriched = build_interval_stats_from_intervals(
                p_hat_cal[:, class_id],
                labels_binary,
                stats["intervals"],
                "PICPI",
            )
            stats.update(
                positive_rate=enriched["positive_rate"],
                calibration_error=enriched["calibration_error"],
                interval_ece=enriched["interval_ece"],
                interval_strategy="PICPI",
            )
        elif strategy == "fixed_bins":
            stats = build_fixed_bin_interval_stats(
                p_hat_cal[:, class_id],
                labels_binary,
                num_bins=cfg.baseline_num_bins if num_bins is None else int(num_bins),
            )
            enriched = build_interval_stats_from_intervals(
                p_hat_cal[:, class_id],
                labels_binary,
                stats["intervals"],
                "Fixed-width binning",
            )
            stats.update(
                positive_rate=enriched["positive_rate"],
                calibration_error=enriched["calibration_error"],
                interval_ece=enriched["interval_ece"],
                interval_strategy="Fixed-width binning",
            )
        else:
            raise ValueError(f"Unknown interval strategy: {strategy}")
        classwise_stats[class_id] = stats
    return classwise_stats


def solve_cover_min_mass_dp(
    contributions: np.ndarray,
    masses: np.ndarray,
    threshold: float,
    scale: int | None = None,
) -> tuple[np.ndarray, float, bool, str]:
    """Exact min-mass interval cover via a vectorised 0/1 knapsack-cover DP.

    Solves ``min sum(masses[i] * x[i])`` subject to
    ``sum(contributions[i] * x[i]) >= threshold`` with ``x[i] in {0, 1}``.

    The coverage axis is discretised into ``DP_TARGET_STATES`` buckets relative to
    ``threshold`` (states at/above the threshold collapse to a single "covered"
    state), so the DP arrays stay small no matter the raw magnitudes. Every step
    is a NumPy vector operation, making this fast enough to replace the HiGHS MILP
    backend while being pure Python/NumPy (and therefore crash-free). ``scale`` is
    accepted for backwards compatibility and ignored.
    """
    del scale  # legacy parameter; resolution is set by DP_TARGET_STATES
    contributions = np.asarray(contributions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n_items = len(masses)

    if threshold <= 1e-12:
        return np.zeros(n_items, dtype=bool), 0.0, True, "dp-trivial"
    if n_items == 0:
        return np.zeros(0, dtype=bool), np.inf, False, "dp-empty"
    if float(contributions.sum()) + 1e-12 < float(threshold):
        return np.zeros(n_items, dtype=bool), np.inf, False, "dp-infeasible"

    cap = int(DP_TARGET_STATES)
    unit = float(threshold) / cap
    # Integer coverage value of each item, saturated at the "covered" state.
    values = np.minimum(cap, np.maximum(0, np.rint(contributions / unit).astype(np.int64)))
    costs = masses.astype(float)

    dp = np.full(cap + 1, np.inf, dtype=float)
    dp[0] = 0.0
    taken = np.zeros((n_items, cap + 1), dtype=bool)
    # Source state chosen whenever an item lands on the saturated "covered" state.
    src_at_cap = np.zeros(n_items, dtype=np.int64)

    for i in range(n_items):
        v = int(values[i])
        if v <= 0:
            # Zero coverage: taking the item only adds mass, never helps a
            # minimum-mass cover, so it is never selected.
            continue
        cost = costs[i]
        take_cost = np.full(cap + 1, np.inf, dtype=float)
        if v >= cap:
            j_best = int(np.argmin(dp))
            take_cost[cap] = dp[j_best] + cost
            src_at_cap[i] = j_best
        else:
            # Non-saturating sources s in [0, cap - v] map to s + v.
            take_cost[v : cap + 1] = dp[0 : cap + 1 - v] + cost
            # Saturating sources s in [cap - v, cap] all land on `cap`; keep best.
            seg = dp[cap - v : cap + 1]
            j_rel = int(np.argmin(seg))
            j_best = (cap - v) + j_rel
            sat_cost = dp[j_best] + cost
            if sat_cost <= take_cost[cap]:
                take_cost[cap] = sat_cost
                src_at_cap[i] = j_best
            else:
                src_at_cap[i] = cap - v
        better = take_cost < dp - 1e-15
        taken[i] = better
        dp = np.where(better, take_cost, dp)

    if not np.isfinite(dp[cap]):
        return np.zeros(n_items, dtype=bool), np.inf, False, "dp-infeasible"

    selected = np.zeros(n_items, dtype=bool)
    cur = cap
    for i in range(n_items - 1, -1, -1):
        if taken[i, cur]:
            selected[i] = True
            cur = int(src_at_cap[i]) if cur == cap else cur - int(values[i])
    return selected, float(masses[selected].sum()), True, "dp"


def solve_cover_min_mass_milp(
    contributions: np.ndarray,
    masses: np.ndarray,
    threshold: float,
    scale: int,
) -> tuple[np.ndarray, float, bool, str]:
    """Solve minimum-mass coverage with MILP, falling back to dynamic programming."""
    contributions = np.asarray(contributions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n_items = len(masses)
    if SCIPY_HAS_MILP and n_items > 0:
        constraint = LinearConstraint(contributions.reshape(1, -1), lb=np.array([threshold]), ub=np.array([np.inf]))
        # WARNING: HiGHS's native MIP backend segfaults when several solver
        # processes run concurrently on this macOS arm64 stack. Use the MILP
        # solver only with --n-workers 1 (or the thread backend, serialised by
        # this lock). For parallel runs prefer the DP solver.
        with MILP_SOLVE_LOCK:
            result = milp(
                c=masses,
                constraints=constraint,
                bounds=Bounds(lb=np.zeros(n_items), ub=np.ones(n_items)),
                integrality=np.ones(n_items, dtype=int),
                options={"time_limit": MILP_TIME_LIMIT_SECONDS},
            )
        if result.success and result.x is not None:
            selected = np.asarray(result.x > 0.5, dtype=bool)
            return selected, float(masses[selected].sum()), True, "milp"

    # Fall back to the DP if MILP is unavailable or failed.
    return solve_cover_min_mass_dp(contributions, masses, threshold, scale)


def solve_cover_min_mass(contributions: np.ndarray, masses: np.ndarray, threshold: float, scale: int) -> tuple[np.ndarray, float, bool, str]:
    """Solve the configured minimum-mass interval-cover problem."""
    contributions = np.asarray(contributions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n_items = len(masses)

    if threshold <= 1e-12:
        return np.zeros(n_items, dtype=bool), 0.0, True, "trivial"
    if contributions.sum() + 1e-12 < threshold:
        return np.zeros(n_items, dtype=bool), np.inf, False, "infeasible"

    if _ACTIVE_SOLVER == "milp":
        return solve_cover_min_mass_milp(contributions, masses, threshold, scale)
    return solve_cover_min_mass_dp(contributions, masses, threshold, scale)


def gamma_values(stats: dict[str, object], selected_mask: np.ndarray) -> dict[str, float]:
    """Compute upper, lower, and combined gamma values for a subset."""
    selected_mask = np.asarray(selected_mask, dtype=bool)
    pi_hat = float(stats["pi_hat"])
    if pi_hat <= 0.0:
        return {"gamma_up": np.inf, "gamma_low": np.inf, "gamma": np.inf}

    m_hat = np.asarray(stats["m_hat"], dtype=float)
    a = np.asarray(stats["a"], dtype=float)
    b = np.asarray(stats["b"], dtype=float)
    t_hat = float(stats["t_hat"])
    gamma_up = float((t_hat + np.sum(m_hat[~selected_mask] * b[~selected_mask])) / pi_hat)
    gamma_low = float(1.0 - np.sum(m_hat[selected_mask] * a[selected_mask]) / pi_hat)
    return {"gamma_up": gamma_up, "gamma_low": gamma_low, "gamma": min(gamma_up, gamma_low)}


def solve_population_subset_for_class(stats: dict[str, object], alpha: float, scale: int) -> dict[str, object]:
    """Choose the minimum-mass feasible interval subset for one class."""
    m_hat = np.asarray(stats["m_hat"], dtype=float)
    a = np.asarray(stats["a"], dtype=float)
    b = np.asarray(stats["b"], dtype=float)
    pi_hat = float(stats["pi_hat"])
    t_hat = float(stats["t_hat"])

    keep_target = max(0.0, (1.0 - alpha) * pi_hat)
    keep_selected, keep_obj, keep_feasible, keep_solver = solve_cover_min_mass(
        a * m_hat,
        m_hat,
        keep_target,
        scale=scale,
    )

    drop_target = float(t_hat + np.sum(b * m_hat) - alpha * pi_hat)
    drop_selected, drop_obj, drop_feasible, drop_solver = solve_cover_min_mass(
        b * m_hat,
        m_hat,
        drop_target,
        scale=scale,
    )

    if keep_feasible and (not drop_feasible or keep_obj <= drop_obj):
        chosen_problem = "keep"
        selected = keep_selected
        objective_mass = keep_obj
        solver_backend = keep_solver
        feasible = True
    elif drop_feasible:
        chosen_problem = "drop"
        selected = drop_selected
        objective_mass = drop_obj
        solver_backend = drop_solver
        feasible = True
    else:
        chosen_problem = "fallback-all"
        selected = np.ones(len(m_hat), dtype=bool)
        objective_mass = float(m_hat.sum())
        solver_backend = "fallback"
        feasible = False

    gamma = gamma_values(stats, selected)
    return {
        "selected": selected,
        "objective_mass": objective_mass,
        "feasible": feasible,
        "chosen_problem": chosen_problem,
        "solver_backend": solver_backend,
        "keep_feasible": keep_feasible,
        "drop_feasible": drop_feasible,
        "keep_target": keep_target,
        "drop_target": drop_target,
        **gamma,
    }


def solve_population_plugin(classwise_stats: dict[int, dict[str, object]], alpha_vector: np.ndarray, scale: int) -> dict[str, object]:
    """Solve the population plug-in subset problem independently by class."""
    alpha_vector = np.asarray(alpha_vector, dtype=float)
    class_solutions = {}
    selected_masks = {}
    for class_id, alpha in enumerate(alpha_vector):
        solution = solve_population_subset_for_class(classwise_stats[class_id], float(alpha), scale=scale)
        class_solutions[class_id] = solution
        selected_masks[class_id] = solution["selected"]
    return {"class_solutions": class_solutions, "selected_masks": selected_masks}


def build_inclusion_matrix(
    p_hat: np.ndarray,
    classwise_stats: dict[int, dict[str, object]],
    selected_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Mark each evaluation row and class included by selected intervals."""
    p_hat = np.asarray(p_hat, dtype=float)
    included = np.zeros_like(p_hat, dtype=bool)
    for class_id, stats in classwise_stats.items():
        selected = np.asarray(selected_masks[class_id], dtype=bool)
        for use_interval, (left, right) in zip(selected, stats["intervals"]):
            if use_interval:
                included[:, class_id] |= score_in_interval(p_hat[:, class_id], left, right)
    return included


def evaluate_inclusion_matrix(
    name: str,
    included: np.ndarray,
    q_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    alpha_target: float = np.nan,
    tau: float = np.nan,
) -> pd.DataFrame:
    """Evaluate classwise coverage and set size from boolean inclusions."""
    included = np.asarray(included, dtype=bool)
    prediction_set_mean_size = float(included.sum(axis=1).mean())
    prediction_set_nonempty_rate = float(included.any(axis=1).mean())

    rows = []
    for class_id in range(q_eval.shape[1]):
        prevalence_true = float(q_eval[:, class_id].mean())
        miss = ~included[:, class_id]
        lhs_weighted = float(np.mean(q_eval[:, class_id] * miss) / prevalence_true)
        lhs_label = float(np.mean(miss[y_eval == class_id])) if np.any(y_eval == class_id) else np.nan
        rows.append(
            {
                "method": name,
                "tau": tau,
                "alpha_target": alpha_target,
                "class": class_id,
                "lhs_weighted": lhs_weighted,
                "lhs_label": lhs_label,
                "prevalence_true": prevalence_true,
                "prediction_set_mean_size": prediction_set_mean_size,
                "prediction_set_nonempty_rate": prediction_set_nonempty_rate,
            }
        )
    return pd.DataFrame(rows)


def evaluate_upper_matrix(
    name: str,
    upper_eval: np.ndarray,
    q_eval: np.ndarray,
    y_eval: np.ndarray,
    tau: float,
    *,
    alpha_target: float = np.nan,
) -> pd.DataFrame:
    """Threshold upper endpoints and evaluate the resulting label sets."""
    included = np.asarray(upper_eval, dtype=float) >= float(tau)
    prediction_set_mean_size = float(included.sum(axis=1).mean())
    prediction_set_nonempty_rate = float(included.any(axis=1).mean())

    rows = []
    for class_id in range(q_eval.shape[1]):
        prevalence_true = float(q_eval[:, class_id].mean())
        miss = ~included[:, class_id]
        lhs_weighted = float(np.mean(q_eval[:, class_id] * miss) / prevalence_true)
        lhs_label = float(np.mean(miss[y_eval == class_id])) if np.any(y_eval == class_id) else np.nan
        rows.append(
            {
                "method": name,
                "tau": float(tau),
                "alpha_target": alpha_target,
                "class": class_id,
                "lhs_weighted": lhs_weighted,
                "lhs_label": lhs_label,
                "prevalence_true": prevalence_true,
                "prediction_set_mean_size": prediction_set_mean_size,
                "prediction_set_nonempty_rate": prediction_set_nonempty_rate,
            }
        )
    return pd.DataFrame(rows)


def evaluate_method_output(
    name: str,
    output: dict[str, object],
    q_eval: np.ndarray,
    y_eval: np.ndarray,
    tau_override: float | None = None,
) -> pd.DataFrame:
    """Evaluate either inclusion-based or upper-endpoint method output."""
    if output["output_type"] == "included":
        return evaluate_inclusion_matrix(
            name,
            np.asarray(output["included_eval"], dtype=bool),
            q_eval,
            y_eval,
            alpha_target=float(output.get("alpha_target", np.nan)),
            tau=np.nan if tau_override is None else float(tau_override),
        )
    if output["output_type"] == "upper":
        tau = float(output["tau"]) if tau_override is None else float(tau_override)
        return evaluate_upper_matrix(
            name,
            np.asarray(output["upper_eval"], dtype=float),
            q_eval,
            y_eval,
            tau,
            alpha_target=float(output.get("alpha_target", np.nan)),
        )
    raise ValueError(f"Unknown output_type: {output['output_type']}")


def _constant_group_metric(method_df: pd.DataFrame, column: str) -> float:
    """Extract a metric required to be constant within a method group."""
    values = pd.unique(method_df[column].to_numpy())
    if len(values) != 1:
        raise ValueError(f"{column} should be constant within each method summary group.")
    return float(values[0])


def _constant_or_nan(method_df: pd.DataFrame, column: str) -> float:
    """Extract one nonmissing group value, or return NaN when absent."""
    values = pd.unique(method_df[column].dropna().to_numpy())
    if len(values) == 0:
        return np.nan
    if len(values) != 1:
        raise ValueError(f"{column} should be constant within each method summary group.")
    return float(values[0])


def summarize_method_rows(method_df: pd.DataFrame) -> dict[str, float | str]:
    """Collapse class-level evaluation rows into one method summary."""
    return {
        "method": method_df["method"].iloc[0],
        "tau": _constant_or_nan(method_df, "tau"),
        "alpha_target": _constant_or_nan(method_df, "alpha_target"),
        "macro_lhs_weighted": float(method_df["lhs_weighted"].mean()),
        "worst_lhs_weighted": float(method_df["lhs_weighted"].max()),
        "best_lhs_weighted": float(method_df["lhs_weighted"].min()),
        "macro_lhs_label": float(method_df["lhs_label"].mean()),
        "worst_lhs_label": float(method_df["lhs_label"].max()),
        "avg_set_size": _constant_group_metric(method_df, "prediction_set_mean_size"),
        "nonempty_rate": _constant_group_metric(method_df, "prediction_set_nonempty_rate"),
    }


def summarize_coverage_curve_rows(method: str, method_rows: pd.DataFrame, seed: int, target_coverage: float) -> dict[str, float | str]:
    """Create one seeded coverage-curve record for a method and target."""
    summary = summarize_method_rows(method_rows)
    summary["method"] = method
    summary["seed"] = seed
    summary["target_coverage"] = float(target_coverage)
    summary["macro_coverage"] = 1.0 - float(summary["macro_lhs_weighted"])
    summary["worst_coverage"] = 1.0 - float(summary["worst_lhs_weighted"])
    return summary


def build_simultaneous_ci_output(
    state: dict[str, object],
    cfg: SweepConfig,
    *,
    x_target: np.ndarray | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, object]:
    """Build calibrated simultaneous-CI output for target feature rows."""
    x_cal = np.asarray(state["x_cal"], dtype=float)
    y_cal = np.asarray(state["y_cal"], dtype=int)
    if x_target is None:
        x_target = np.asarray(state["x_eval"], dtype=float)
    else:
        x_target = ensure_2d_array(x_target)

    upper_cal = np.zeros((len(x_cal), cfg.n_classes), dtype=float)
    upper_eval = np.zeros((len(x_target), cfg.n_classes), dtype=float)
    binary_ci_helpers: dict[int, dict[str, object]] = state["binary_ci_helpers"]  # type: ignore[assignment]

    for class_id in range(cfg.n_classes):
        intervals_cal = simultaneous_ci_intervals_from_helper(binary_ci_helpers[class_id], x_cal, alpha=alpha)
        intervals_eval = simultaneous_ci_intervals_from_helper(binary_ci_helpers[class_id], x_target, alpha=alpha)
        upper_cal[:, class_id] = np.asarray(intervals_cal, dtype=float)[:, 1]
        upper_eval[:, class_id] = np.asarray(intervals_eval, dtype=float)[:, 1]

    tau = calibration_tau_from_upper_matrix(upper_cal, y_cal, alpha=alpha)
    return {
        "output_type": "upper",
        "upper_eval": upper_eval,
        "upper_cal": upper_cal,
        "tau": tau,
        "alpha_target": alpha,
    }


def build_conformal_classification_output(
    state: dict[str, object],
    *,
    p_hat_target: np.ndarray | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, object]:
    """Build standard conformal-classification label-set output."""
    y_cal = np.asarray(state["y_cal"], dtype=int)
    p_hat_cal = np.asarray(state["p_hat_cal"], dtype=float)
    if p_hat_target is None:
        p_hat_target = np.asarray(state["p_hat_eval"], dtype=float)
    else:
        p_hat_target = np.asarray(p_hat_target, dtype=float)

    n_cal = len(y_cal)
    true_label_probabilities = p_hat_cal[np.arange(n_cal), y_cal]
    k = int(np.ceil(alpha * (n_cal + 1)))
    if k < 1:
        probability_threshold = -np.inf
    elif k > n_cal:
        probability_threshold = np.inf
    else:
        probability_threshold = float(np.sort(true_label_probabilities)[k - 1])
    included_eval = p_hat_target >= probability_threshold

    return {
        "output_type": "included",
        "included_eval": included_eval,
        "alpha_target": alpha,
        "probability_threshold": probability_threshold,
        "threshold_order_statistic": k,
    }


def build_plugin_stats_for_target(
    state: dict[str, object],
    cfg: SweepConfig,
    strategy_key: str,
    num_bins: int | None,
    alpha_target: float,
    *,
    simultaneous_ci_stats: dict[int, dict[str, object]] | None = None,
    picpi_stats: dict[int, dict[str, object]] | None = None,
) -> dict[int, dict[str, object]]:
    """Build or reuse interval statistics for one plug-in strategy."""
    if strategy_key == "simultaneous_ci":
        return (
            simultaneous_ci_stats
            if simultaneous_ci_stats is not None
            else build_simultaneous_ci_interval_stats(state, cfg, alpha=alpha_target)
        )
    if strategy_key == "calibration_based_interval":
        return build_calibration_based_interval_stats(
            state,
            cfg,
            num_bins=num_bins,
            alpha=alpha_target,
            picpi_stats=picpi_stats,
        )
    return build_classwise_interval_stats(state, cfg, strategy=strategy_key, num_bins=num_bins, alpha=alpha_target)


def run_method_coverage_sweep(seed: int, dgp: DGPDefinition, cfg: SweepConfig) -> pd.DataFrame:
    """Evaluate every Task 2 method and coverage target for one seed."""
    # Each (spawned) worker re-imports this module fresh, so set the process-local
    # solver from the config here, before any solve runs.
    global _ACTIVE_SOLVER
    _ACTIVE_SOLVER = cfg.solver
    state = fit_shared_model(cfg, dgp, seed=seed)
    target_coverages = np.asarray(cfg.coverage_sweep_targets, dtype=float)
    interval_configs = plugin_interval_configs(cfg)

    cached_stats = {
        (strategy_key, num_bins): build_plugin_stats_for_target(
            state,
            cfg,
            strategy_key,
            num_bins,
            alpha_target=cfg.alpha,
        )
        for _, strategy_key, num_bins in interval_configs
        if strategy_key not in {"calibration_based_interval", "split_conformal_neighborhood"}
    }
    cached_picpi_stats = cached_stats[("picpi", cfg.num_bins_picpi)]

    rows: list[dict[str, float | str]] = []
    for target_coverage in target_coverages:
        alpha_target = float(1.0 - target_coverage)
        alpha_vector = np.full(cfg.n_classes, alpha_target, dtype=float)

        simultaneous_ci_output = build_simultaneous_ci_output(
            state,
            cfg,
            x_target=np.asarray(state["x_eval"], dtype=float),
            alpha=alpha_target,
        )
        simultaneous_ci_rows = evaluate_method_output(
            "Simultaneous CI",
            simultaneous_ci_output,
            np.asarray(state["q_eval"], dtype=float),
            np.asarray(state["y_eval"], dtype=int),
        )
        rows.append(
            summarize_coverage_curve_rows(
                "Simultaneous CI",
                simultaneous_ci_rows,
                seed,
                float(target_coverage),
            )
        )

        for strategy_name, strategy_key, num_bins in interval_configs:
            if strategy_key in {"calibration_based_interval", "split_conformal_neighborhood"}:
                stats = build_plugin_stats_for_target(
                    state,
                    cfg,
                    strategy_key,
                    num_bins,
                    alpha_target=alpha_target,
                    picpi_stats=cached_picpi_stats,
                )
            else:
                stats = cached_stats[(strategy_key, num_bins)]

            max_intervals = max(len(class_stats["intervals"]) for class_stats in stats.values())
            scale = int(len(np.asarray(state["y_cal"], dtype=int)) * max(max_intervals, 1))
            plugin_solution = solve_population_plugin(stats, alpha_vector=alpha_vector, scale=scale)
            included = build_inclusion_matrix(
                np.asarray(state["p_hat_eval"], dtype=float),
                stats,
                plugin_solution["selected_masks"],
            )
            method_rows = evaluate_inclusion_matrix(
                strategy_name,
                included,
                np.asarray(state["q_eval"], dtype=float),
                np.asarray(state["y_eval"], dtype=int),
                alpha_target=alpha_target,
            )
            rows.append(summarize_coverage_curve_rows(strategy_name, method_rows, seed, float(target_coverage)))

        conformal_classification_output = build_conformal_classification_output(
            state,
            p_hat_target=np.asarray(state["p_hat_eval"], dtype=float),
            alpha=alpha_target,
        )
        conformal_classification_rows = evaluate_method_output(
            "Conformal classification",
            conformal_classification_output,
            np.asarray(state["q_eval"], dtype=float),
            np.asarray(state["y_eval"], dtype=int),
        )
        rows.append(
            summarize_coverage_curve_rows(
                "Conformal classification",
                conformal_classification_rows,
                seed,
                float(target_coverage),
            )
        )

    result = pd.DataFrame(rows)
    result["dgp_name"] = dgp.name
    result["dgp_title"] = dgp.title
    return result


def summarize_coverage_curve(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seeded coverage-curve results by method and target."""
    if long_df.empty:
        raise ValueError("Cannot summarize an empty long-form result table.")

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


def expected_rows_per_seed(cfg: SweepConfig) -> int:
    """Return the number of method-target rows expected from each seed."""
    return len(CURVE_METHOD_ORDER) * len(cfg.coverage_sweep_targets)


def build_seed_list(cfg: SweepConfig) -> list[int]:
    """Create the deterministic replication seeds for a sweep."""
    return [cfg.seed + SEED_STEP * idx for idx in range(cfg.mc_reps)]


def extract_completed_seeds(existing_long_df: pd.DataFrame, cfg: SweepConfig) -> set[int]:
    """Identify seeds with a complete set of saved method-target rows."""
    if existing_long_df.empty:
        return set()
    key_counts = (
        existing_long_df[["seed", "method", "target_coverage"]]
        .drop_duplicates()
        .groupby("seed")
        .size()
    )
    required = expected_rows_per_seed(cfg)
    return {int(seed) for seed, count in key_counts.items() if int(count) >= required}


def study_signature(cfg: SweepConfig, dgp: DGPDefinition) -> dict[str, object]:
    """Describe settings that must match when resuming a saved sweep."""
    return {
        "dgp_name": dgp.name,
        "dgp_title": dgp.title,
        "n_features": dgp.n_features,
        "description": dgp.description,
        "feature_distribution": dgp.feature_distribution,
        "logit_description": dgp.logit_description,
        "dgp_parameters": dgp.metadata,
        "n_classes": cfg.n_classes,
        "alpha": cfg.alpha,
        "n_train": cfg.n_train,
        "n_cal": cfg.n_cal,
        "n_eval": cfg.n_eval,
        "num_bins_picpi": cfg.num_bins_picpi,
        "baseline_num_bins": cfg.baseline_num_bins,
        "coverage_sweep_targets": [float(value) for value in cfg.coverage_sweep_targets],
        "curve_method_order": list(CURVE_METHOD_ORDER),
        "plugin_interval_configs": [
            {"strategy_name": name, "strategy_key": key, "num_bins": num_bins}
            for name, key, num_bins in plugin_interval_configs(cfg)
        ],
    }


def build_config_payload(
    cfg: SweepConfig,
    dgp: DGPDefinition,
    requested_seeds: Sequence[int],
    completed_seeds: Sequence[int],
    smoke_mode: bool,
) -> dict[str, object]:
    """Build the JSON-serializable configuration saved with one DGP."""
    resolved_backend = effective_parallel_backend(cfg.parallel_backend, cfg.n_workers)
    return {
        "study_signature": study_signature(cfg, dgp),
        "run": {
            "requested_mc_reps": int(cfg.mc_reps),
            "requested_seeds": [int(seed) for seed in requested_seeds],
            "completed_seeds": [int(seed) for seed in completed_seeds],
            "smoke_mode": bool(smoke_mode),
            "reference_coverage": float(1.0 - cfg.alpha),
            "requested_parallel_backend": cfg.parallel_backend,
            "effective_parallel_backend": resolved_backend,
            "parallel_backend": resolved_backend,
            "solver": cfg.solver,
            "multiprocessing_max_tasks_per_child": (
                MULTIPROCESSING_MAX_TASKS_PER_CHILD if resolved_backend == "multiprocessing" else None
            ),
        },
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    """Write a configuration dictionary as readable JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_saved_dgp_results(output_dir: Path | str = DEFAULT_RESULTS_DIR, dgp_names: Sequence[str] | None = None) -> dict[str, dict[str, object]]:
    """Load stored summaries and configurations for requested DGPs."""
    output_dir = Path(output_dir)
    names = list(dgp_names) if dgp_names is not None else list(DGP_DISPLAY_ORDER)
    bundles: dict[str, dict[str, object]] = {}
    for dgp_name in names:
        dgp_dir = output_dir / dgp_name
        summary_path = dgp_dir / "summary.csv"
        config_path = dgp_dir / "config.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file for {dgp_name}: {summary_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file for {dgp_name}: {config_path}")
        bundles[dgp_name] = {
            "summary": pd.read_csv(summary_path),
            "config": json.loads(config_path.read_text()),
            "directory": dgp_dir,
        }
    return bundles


def run_dgp_sweep(
    dgp: DGPDefinition,
    cfg: SweepConfig,
    output_dir: Path,
    *,
    overwrite: bool = False,
    smoke_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run, resume, and save the configured Monte Carlo sweep for one DGP."""
    dgp_dir = output_dir / dgp.name
    dgp_dir.mkdir(parents=True, exist_ok=True)
    long_path = dgp_dir / "long.csv"
    summary_path = dgp_dir / "summary.csv"
    config_path = dgp_dir / "config.json"

    requested_seeds = build_seed_list(cfg)
    signature = study_signature(cfg, dgp)
    existing_long_df = pd.DataFrame()
    if not overwrite and config_path.exists():
        existing_config = json.loads(config_path.read_text())
        existing_signature = existing_config.get("study_signature")
        if existing_signature != signature:
            raise ValueError(
                f"Existing config for {dgp.name} does not match the requested study signature. "
                "Use --overwrite to replace it."
            )
    if not overwrite and long_path.exists():
        existing_long_df = pd.read_csv(long_path)
        existing_long_df = existing_long_df[existing_long_df["seed"].isin(requested_seeds)].copy()

    completed_seeds = extract_completed_seeds(existing_long_df, cfg)
    pending_seeds = [seed for seed in requested_seeds if seed not in completed_seeds]

    if pending_seeds:
        worker = partial(run_method_coverage_sweep, dgp=dgp, cfg=cfg)
        backend = cfg.parallel_backend
        if (
            cfg.solver == "milp"
            and cfg.n_workers > 1
            and effective_parallel_backend(backend, cfg.n_workers) == "multiprocessing"
        ):
            # HiGHS segfaults when several solver processes run concurrently on
            # this stack. Run MILP on the thread pool instead, where MILP_SOLVE_LOCK
            # serialises the native solver so it never runs concurrently.
            warnings.warn(
                f"{dgp.title}: --solver milp is unsafe under process parallelism "
                "(HiGHS segfaults); using the thread backend so MILP solves are "
                "serialised. Use --solver dp for fast process parallelism.",
                RuntimeWarning,
            )
            backend = "threads"
        new_frames = run_parallel_seed_jobs(
            worker,
            pending_seeds,
            desc=f"{dgp.title} coverage sweep",
            n_workers=cfg.n_workers,
            backend=backend,
        )
        new_long_df = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
        combined_long_df = pd.concat([existing_long_df, new_long_df], ignore_index=True)
    else:
        combined_long_df = existing_long_df.copy()

    combined_long_df = (
        combined_long_df.sort_values(["seed", "target_coverage", "method"])
        .drop_duplicates(subset=["seed", "method", "target_coverage"], keep="last")
        .reset_index(drop=True)
    )
    completed_seeds = sorted(extract_completed_seeds(combined_long_df, cfg))
    combined_long_df.to_csv(long_path, index=False)

    summary_df = summarize_coverage_curve(combined_long_df)
    summary_df.to_csv(summary_path, index=False)

    config_payload = build_config_payload(cfg, dgp, requested_seeds, completed_seeds, smoke_mode)
    save_json(config_path, config_payload)
    return combined_long_df, summary_df


def apply_smoke_mode(cfg: SweepConfig, *, custom_coverages_provided: bool) -> SweepConfig:
    """Reduce sample sizes for a fast end-to-end verification run."""
    return replace(
        cfg,
        n_train=min(cfg.n_train, 500),
        n_cal=min(cfg.n_cal, 500),
        n_eval=min(cfg.n_eval, 500),
        coverage_sweep_targets=(
            cfg.coverage_sweep_targets if custom_coverages_provided else SMOKE_COVERAGE_SWEEP_TARGETS
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the Task 2 sweep."""
    parser = argparse.ArgumentParser(
        description="Run the multiclass DGP coverage sweep and save one result bundle per DGP.",
    )
    parser.add_argument(
        "--dgp",
        action="append",
        choices=[*DGP_DISPLAY_ORDER, "all"],
        help="Repeat to run a subset of DGPs. Defaults to all.",
    )
    parser.add_argument("--mc-reps", type=int, default=DEFAULT_CONFIG.mc_reps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)
    parser.add_argument("--n-train", type=int, default=DEFAULT_CONFIG.n_train)
    parser.add_argument("--n-cal", type=int, default=DEFAULT_CONFIG.n_cal)
    parser.add_argument("--n-eval", type=int, default=DEFAULT_CONFIG.n_eval)
    parser.add_argument("--num-bins-picpi", type=int, default=DEFAULT_CONFIG.num_bins_picpi)
    parser.add_argument("--baseline-num-bins", type=int, default=DEFAULT_CONFIG.baseline_num_bins)
    parser.add_argument("--n-workers", type=int, default=DEFAULT_CONFIG.n_workers)
    parser.add_argument(
        "--parallel-backend",
        choices=("threads", "processes", "multiprocessing"),
        default=DEFAULT_CONFIG.parallel_backend,
        help=(
            "Parallel backend for per-seed jobs. Defaults to multiprocessing (a "
            "crash-resilient, spawn-based process pool). 'processes' is an alias; "
            "'threads' uses a thread pool."
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("dp", "milp"),
        default=DEFAULT_CONFIG.solver,
        help=(
            "Min-mass interval-cover solver. 'dp' (default) is an exact pure-NumPy "
            "dynamic program that is safe under process parallelism. 'milp' uses "
            "SciPy/HiGHS (exact, but its native code can segfault when several "
            "solver processes run concurrently -- use only with --n-workers 1)."
        ),
    )
    parser.add_argument(
        "--coverage-targets",
        nargs="*",
        type=float,
        default=None,
        help="Optional custom target coverage grid.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run with reduced sample sizes while preserving the requested --mc-reps and --n-workers.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite incompatible or partial result bundles.")
    return parser.parse_args(argv)


def resolve_requested_dgps(raw_dgps: Sequence[str] | None) -> list[DGPDefinition]:
    """Resolve requested DGP names to registered definitions."""
    if not raw_dgps or "all" in raw_dgps:
        return [DGP_REGISTRY[name] for name in DGP_DISPLAY_ORDER]
    requested = []
    seen = set()
    for name in raw_dgps:
        if name in seen:
            continue
        requested.append(DGP_REGISTRY[name])
        seen.add(name)
    return requested


def build_config_from_args(args: argparse.Namespace) -> SweepConfig:
    """Translate parsed options into a sweep configuration."""
    coverage_targets = (
        tuple(float(value) for value in args.coverage_targets)
        if args.coverage_targets is not None and len(args.coverage_targets) > 0
        else DEFAULT_CONFIG.coverage_sweep_targets
    )
    cfg = SweepConfig(
        seed=int(args.seed),
        alpha=float(args.alpha),
        mc_reps=int(args.mc_reps),
        n_train=int(args.n_train),
        n_cal=int(args.n_cal),
        n_eval=int(args.n_eval),
        n_classes=N_CLASSES,
        num_bins_picpi=int(args.num_bins_picpi),
        baseline_num_bins=int(args.baseline_num_bins),
        n_workers=int(args.n_workers),
        parallel_backend=str(args.parallel_backend),
        solver=str(args.solver),
        coverage_sweep_targets=coverage_targets,
        split_conformal_max_intervals_per_class=SPLIT_CONFORMAL_MAX_INTERVALS_PER_CLASS,
    )
    if args.smoke:
        cfg = apply_smoke_mode(cfg, custom_coverages_provided=args.coverage_targets is not None)
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    """Run Task 2 from command-line options and report saved outputs."""
    args = parse_args(argv)
    cfg = build_config_from_args(args)
    requested_dgps = resolve_requested_dgps(args.dgp)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_backend = effective_parallel_backend(cfg.parallel_backend, cfg.n_workers)

    print("Running multiclass DGP sweep with:")
    print(
        json.dumps(
            {
                "seed": cfg.seed,
                "mc_reps": cfg.mc_reps,
                "alpha": cfg.alpha,
                "n_train": cfg.n_train,
                "n_cal": cfg.n_cal,
                "n_eval": cfg.n_eval,
                "n_workers": cfg.n_workers,
                "requested_parallel_backend": cfg.parallel_backend,
                "effective_parallel_backend": resolved_backend,
                "solver": cfg.solver,
                "multiprocessing_max_tasks_per_child": (
                    MULTIPROCESSING_MAX_TASKS_PER_CHILD if resolved_backend == "multiprocessing" else None
                ),
                "coverage_sweep_targets": [float(value) for value in cfg.coverage_sweep_targets],
                "dgps": [dgp.name for dgp in requested_dgps],
                "output_dir": str(output_dir),
                "smoke_mode": bool(args.smoke),
                "overwrite": bool(args.overwrite),
            },
            indent=2,
        )
    )

    for dgp in requested_dgps:
        print(f"\n[{dgp.name}] {dgp.title}")
        _, summary_df = run_dgp_sweep(
            dgp,
            cfg,
            output_dir,
            overwrite=bool(args.overwrite),
            smoke_mode=bool(args.smoke),
        )
        completed = int(summary_df["n_completed_seeds"].max())
        print(f"Saved summary to {output_dir / dgp.name / 'summary.csv'}")
        print(f"Completed seeds: {completed}")


if __name__ == "__main__":
    main()
