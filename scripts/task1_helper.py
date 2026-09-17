"""Reusable Task 1 computation helpers.

Original source: ``experiments/final_sub/task1_final_submission.ipynb``.
Includes the univariate visualization, misspecified tree, and multivariate table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from picpi.calibration import calibration
from scripts.paths import cached

TASK1_METHOD_ORDER = [
    "Simultaneous confidence interval",
    "PICPI",
    "Split conformal prediction",
    "Calibration-based interval",
    "Fixed-width binning",
]

ALPHA = 0.1
NUM_BINS_PICPI = 50
PICPI_EPSILON = 0.02
NUM_BINS = 50
N_MC_TABLE = 100

UNIVARIATE_SEED = 42
UNIVARIATE_N_TRAIN = 2000
UNIVARIATE_N_CAL = 2000
UNIVARIATE_N_EVAL = 10000
UNIVARIATE_BETA = 2.0
UNIVARIATE_EPSILON = 0.1
X_GRID = np.linspace(-2, 2, 1000)

MISSPECIFIED_SEED = 7
MISSPECIFIED_N_TRAIN = 20000
MISSPECIFIED_N_CAL = 20000

MULTIVARIATE_N_TRAIN = 2000
MULTIVARIATE_N_CAL = 2000
MULTIVARIATE_N_EVAL = 10000
MULTIVARIATE_DIM = 20
MULTIVARIATE_BETA0 = 0.0
MULTIVARIATE_BETA = np.concatenate(
    [np.array([1.6, -1.1, 0.9], dtype=float), np.zeros(MULTIVARIATE_DIM - 3, dtype=float)]
)
MULTIVARIATE_EPSILON = 0.10


def ensure_2d(x):
    """Convert a feature array to the two-dimensional shape models expect."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def predict_binary_proba(model, x):
    """Return the fitted model's probability for outcome one."""
    return model.predict_proba(ensure_2d(x))[:, 1]


def fit_shared_binary_model(x_train, y_train):
    """Fit the logistic model shared by the Task 1 comparisons."""
    return LogisticRegression(solver="lbfgs", max_iter=2000).fit(
        ensure_2d(x_train), y_train
    )


def p_true_univariate(x, beta: float = UNIVARIATE_BETA):
    """Evaluate the univariate DGP's conditional outcome probability."""
    return expit(beta * np.asarray(x, dtype=float))


def generate_univariate_split(
    n_train, n_cal, n_eval, seed=UNIVARIATE_SEED
):
    """Generate independent training, calibration, and evaluation samples."""
    rng = np.random.default_rng(seed)

    def _fold(n):
        """Draw one noisy univariate sample of the requested size."""
        x = rng.choice(X_GRID, size=n)
        noise = rng.normal(0.0, UNIVARIATE_EPSILON, size=n)
        p = expit(UNIVARIATE_BETA * x + noise)
        y = rng.binomial(1, p)
        return x, y

    return _fold(n_train), _fold(n_cal), _fold(n_eval)


def p_true_multivariate(x, beta0=MULTIVARIATE_BETA0, beta=MULTIVARIATE_BETA):
    """Evaluate the multivariate DGP's conditional outcome probability."""
    x = ensure_2d(x)
    return expit(beta0 + x @ beta)


def generate_multivariate_split(n_train, n_cal, n_eval, seed=0):
    """Generate the three samples used by one multivariate replication."""
    rng = np.random.default_rng(seed)

    def _fold(n):
        """Draw one noisy multivariate sample of the requested size."""
        x = rng.normal(0.0, 1.0, size=(n, MULTIVARIATE_DIM))
        noise = rng.normal(0.0, MULTIVARIATE_EPSILON, size=n)
        p = expit(MULTIVARIATE_BETA0 + x @ MULTIVARIATE_BETA + noise)
        y = rng.binomial(1, p)
        return x, y

    return _fold(n_train), _fold(n_cal), _fold(n_eval)


def p_true_tree(x):
    """Evaluate the nonlinear truth used in the tree misspecification example."""
    x = np.asarray(x, dtype=float)
    z = 1.2 * np.sin(2.5 * x) + 0.8 * x
    return 1.0 / (1.0 + np.exp(-z))


def sample_tree_data(n, rng):
    """Draw features and outcomes for the tree misspecification example."""
    x = rng.uniform(-2.0, 2.0, size=n)
    p = p_true_tree(x)
    y = rng.binomial(1, p)
    return x, y


def method_simultaneous_band(model, x_train, x_eval, alpha=ALPHA):
    """Construct simultaneous logistic-regression confidence bands."""
    x_train_2d = ensure_2d(x_train)
    x_eval_2d = ensure_2d(x_eval)
    x_design = np.column_stack([np.ones(len(x_train_2d)), x_train_2d])
    p_hat_train = model.predict_proba(x_train_2d)[:, 1]
    c = np.sqrt(chi2.ppf(1 - alpha, df=x_design.shape[1]))
    fisher = x_design.T @ (x_design * (p_hat_train * (1 - p_hat_train))[:, None])
    cov_beta = np.linalg.pinv(fisher)
    coef = model.coef_.reshape(-1)
    intercept = float(model.intercept_[0])
    intervals = np.zeros((len(x_eval_2d), 2), dtype=float)
    for idx, x_row in enumerate(x_eval_2d):
        x_d = np.concatenate(([1.0], x_row))
        eta = intercept + float(x_row @ coef)
        se_eta = np.sqrt(max(float(x_d @ cov_beta @ x_d), 0.0))
        intervals[idx] = [expit(eta - c * se_eta), expit(eta + c * se_eta)]
    return intervals


def build_picpi_partition(
    p_hat_cal, y_cal, num_bins=NUM_BINS_PICPI, mode="empirical", delta=None
):
    """Construct a PICPI partition from held-out calibration predictions."""
    partition = calibration(
        list(zip(range(len(y_cal)), np.asarray(y_cal, dtype=int).tolist())),
        np.asarray(p_hat_cal, dtype=float).tolist(),
        num_bin=num_bins,
        mode=mode,
        delta=delta,
    )
    if partition and partition[-1][1] < 1.0:
        partition.append((partition[-1][1], 1.0))
    return partition


def assign_partition_interval(scores, partition, expand=0.0):
    """Assign each predicted score to its partition interval."""
    scores = np.asarray(scores, dtype=float)
    intervals = np.zeros((len(scores), 2), dtype=float)
    for idx, score in enumerate(scores):
        chosen = None
        for left, right in partition:
            if (left < score <= right) or (left == 0.0 and score == 0.0):
                chosen = (left, right)
                break
        if chosen is None:
            if partition:
                distances = [
                    min(abs(score - left), abs(score - right))
                    for left, right in partition
                ]
                chosen = partition[int(np.argmin(distances))]
            else:
                chosen = (0.0, 1.0)
        intervals[idx] = [max(0.0, chosen[0] - expand), min(1.0, chosen[1] + expand)]
    return intervals


def score_in_interval(scores, left, right):
    """Test membership in a right-closed calibration interval."""
    scores = np.asarray(scores, dtype=float)
    return ((scores > left) & (scores <= right)) | ((left == 0.0) & (scores == 0.0))


def compute_calibration_interval_stats(scores, labels, intervals):
    """Compute interval masses, calibration errors, and aggregate ECE."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    masses = []
    cal_errors = []
    for left, right in intervals:
        mask = score_in_interval(scores, left, right)
        masses.append(float(mask.mean()))
        if mask.any():
            midpoint = 0.5 * (float(left) + float(right))
            cal_errors.append(abs(float(labels[mask].mean()) - midpoint))
        else:
            cal_errors.append(np.inf)
    masses = np.asarray(masses, dtype=float)
    cal_errors = np.asarray(cal_errors, dtype=float)
    finite_errors = np.where(np.isfinite(cal_errors), cal_errors, 0.0)
    interval_ece = float(np.sum(masses * finite_errors))
    return masses, cal_errors, interval_ece


def choose_intervals_by_average_ece(errors, masses, target_ece):
    """Select the lowest-error intervals nearest a target average ECE."""
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


def method_picpi(
    model,
    x_cal,
    y_cal,
    x_eval,
    num_bins=NUM_BINS_PICPI,
    eps=PICPI_EPSILON,
    mode="empirical",
    delta=None,
):
    """Fit the PICPI method and assign intervals to evaluation inputs."""
    p_hat_cal = predict_binary_proba(model, x_cal)
    p_hat_eval = predict_binary_proba(model, x_eval)
    partition = build_picpi_partition(
        p_hat_cal, y_cal, num_bins=num_bins, mode=mode, delta=delta
    )
    intervals = assign_partition_interval(p_hat_eval, partition, expand=eps)
    return intervals, partition


def method_split_conformal(model, x_cal, y_cal, x_eval, alpha=ALPHA):
    """Construct split-conformal probability intervals."""
    p_hat_cal = predict_binary_proba(model, x_cal)
    p_hat_eval = predict_binary_proba(model, x_eval)
    scores = np.abs(np.asarray(y_cal, dtype=float) - p_hat_cal)
    q_level = np.ceil((len(y_cal) + 1) * (1 - alpha)) / len(y_cal)
    q_hat = float(np.quantile(scores, min(q_level, 1.0)))
    intervals = np.column_stack(
        [
            np.clip(p_hat_eval - q_hat, 0.0, 1.0),
            np.clip(p_hat_eval + q_hat, 0.0, 1.0),
        ]
    )
    return intervals, q_hat


def method_calibration_based_interval(
    model,
    x_cal,
    y_cal,
    x_eval,
    num_bins=NUM_BINS,
    picpi_partition=None,
    target_ece=None,
):
    """Construct fixed-bin intervals selected to match a target ECE."""
    p_hat_cal = predict_binary_proba(model, x_cal)
    p_hat_eval = predict_binary_proba(model, x_eval)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    candidate_intervals = [
        (float(edges[k]), float(edges[k + 1])) for k in range(num_bins)
    ]
    masses, cal_errors, _ = compute_calibration_interval_stats(
        p_hat_cal, y_cal, candidate_intervals
    )
    if target_ece is None and picpi_partition is not None:
        _, _, target_ece = compute_calibration_interval_stats(
            p_hat_cal, y_cal, picpi_partition
        )
    if target_ece is None:
        target_ece = np.inf
    selected = choose_intervals_by_average_ece(cal_errors, masses, target_ece)
    sel_idx = np.where(selected)[0]
    intervals = np.zeros((len(p_hat_eval), 2), dtype=float)
    for idx, score in enumerate(p_hat_eval):
        k = min(int(score * num_bins), num_bins - 1)
        if selected[k]:
            chosen = k
        elif len(sel_idx):
            chosen = int(sel_idx[np.argmin(np.abs(mids[sel_idx] - score))])
        else:
            intervals[idx] = [0.0, 1.0]
            continue
        intervals[idx] = [edges[chosen], edges[chosen + 1]]
    return intervals, selected, cal_errors


def method_fixed_width_binning(model, x_eval, num_bins=NUM_BINS):
    """Assign predictions to equal-width probability bins."""
    p_hat_eval = predict_binary_proba(model, x_eval)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    intervals = np.zeros((len(p_hat_eval), 2), dtype=float)
    for idx, score in enumerate(p_hat_eval):
        k = min(int(score * num_bins), num_bins - 1)
        intervals[idx] = [edges[k], edges[k + 1]]
    return intervals


def compute_interval_metrics(intervals, p_star, p_hat=None):
    """Evaluate interval width, calibration error, and coverage metrics.

    Two ECE definitions are available. Both average unweighted over the
    distinct reported intervals ``I_j = [ell_j, u_j]`` with index sets
    ``A_j = {i : C(X_i) = I_j}``.

    Midpoint ECE (paper / ``ece``):
        (1/M) sum_j |mean_{i in A_j} p*(X_i) - midpoint(I_j)|

    Mean-score ECE (``ece_mean_phat``; requires ``p_hat``):
        (1/M) sum_j |mean_{i in A_j} p*(X_i) - mean_{i in A_j} p-hat(X_i)|
    """
    intervals = np.asarray(intervals, dtype=float)
    p_star = np.asarray(p_star, dtype=float)
    p_hat = None if p_hat is None else np.asarray(p_hat, dtype=float)
    lower = intervals[:, 0]
    upper = intervals[:, 1]
    lengths = upper - lower
    covered = (p_star >= lower) & (p_star <= upper)
    rounded = np.round(intervals, 12)
    unique_intervals = np.unique(rounded, axis=0)
    ece_midpoint_terms = []
    ece_mean_phat_terms = []
    for left, right in unique_intervals:
        mask = np.isclose(lower, left) & np.isclose(upper, right)
        if not mask.any():
            continue
        mean_pstar = float(np.mean(p_star[mask]))
        midpoint = 0.5 * (float(left) + float(right))
        ece_midpoint_terms.append(abs(mean_pstar - midpoint))
        if p_hat is not None:
            ece_mean_phat_terms.append(abs(mean_pstar - float(np.mean(p_hat[mask]))))
    grid = np.linspace(0.0, 1.0, 10001)
    covered_grid = np.zeros_like(grid, dtype=bool)
    for left, right in unique_intervals:
        covered_grid |= (grid >= left) & (grid <= right)
    metrics = {
        "avg_length": float(np.mean(lengths)),
        "ece": float(np.mean(ece_midpoint_terms)) if ece_midpoint_terms else 0.0,
        "coverage_01_union": float(np.mean(covered_grid)),
        "coverage_pstar": float(np.mean(covered)),
    }
    if p_hat is not None:
        metrics["ece_mean_phat"] = (
            float(np.mean(ece_mean_phat_terms)) if ece_mean_phat_terms else 0.0
        )
    return metrics


def build_task1_results(
    model, x_train, x_cal, y_cal, x_eval, picpi_mode="empirical", picpi_delta=None
):
    """Run every Task 1 method on the same fitted model and data split."""
    intervals_band = method_simultaneous_band(model, x_train, x_eval)
    intervals_picpi, picpi_partition = method_picpi(
        model,
        x_cal,
        y_cal,
        x_eval,
        num_bins=NUM_BINS_PICPI,
        eps=PICPI_EPSILON,
        mode=picpi_mode,
        delta=picpi_delta,
    )
    intervals_split, split_quantile = method_split_conformal(
        model, x_cal, y_cal, x_eval, alpha=ALPHA
    )
    p_hat_cal = predict_binary_proba(model, x_cal)
    _, _, reference_ece = compute_calibration_interval_stats(
        p_hat_cal, y_cal, picpi_partition
    )
    intervals_calib, selected_bins, cal_errors = method_calibration_based_interval(
        model,
        x_cal,
        y_cal,
        x_eval,
        num_bins=NUM_BINS,
        picpi_partition=picpi_partition,
        target_ece=reference_ece,
    )
    intervals_fixed = method_fixed_width_binning(model, x_eval, num_bins=NUM_BINS)
    intervals_by_method = {
        "Simultaneous confidence interval": intervals_band,
        "PICPI": intervals_picpi,
        "Split conformal prediction": intervals_split,
        "Calibration-based interval": intervals_calib,
        "Fixed-width binning": intervals_fixed,
    }
    metadata = {
        "picpi_partition": picpi_partition,
        "split_quantile": split_quantile,
        "selected_bins": selected_bins,
        "calibration_errors": cal_errors,
    }
    return intervals_by_method, metadata


def summarise_task1_results(
    intervals_by_method, p_star_eval, selected_bins, p_hat_eval=None
):
    """Create one comparison row per Task 1 interval method."""
    rows = []
    for method in TASK1_METHOD_ORDER:
        metrics = compute_interval_metrics(
            intervals_by_method[method], p_star_eval, p_hat=p_hat_eval
        )
        if method == "Simultaneous confidence interval":
            coverage_01 = np.nan
        elif method == "PICPI":
            coverage_01 = metrics["coverage_01_union"]
        elif method == "Split conformal prediction":
            coverage_01 = np.nan
        elif method == "Calibration-based interval":
            coverage_01 = float(np.mean(selected_bins))
        else:
            coverage_01 = 1.0
        row = {
            "Method": method,
            "Average length": metrics["avg_length"],
            "ECE": metrics["ece"],
        }
        if "ece_mean_phat" in metrics:
            row["ECE mean p-hat"] = metrics["ece_mean_phat"]
        row["Coverage [0,1]"] = coverage_01
        row["Coverage p*"] = metrics["coverage_pstar"]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_univariate_visualization():
    """Compute all arrays and metrics for the univariate comparison figure."""
    (x_train, y_train), (x_cal, y_cal), (x_eval, _y_eval) = generate_univariate_split(
        UNIVARIATE_N_TRAIN,
        UNIVARIATE_N_CAL,
        UNIVARIATE_N_EVAL,
        seed=UNIVARIATE_SEED,
    )
    model = fit_shared_binary_model(x_train, y_train)
    p_star_eval = p_true_univariate(x_eval)
    p_hat_eval = predict_binary_proba(model, x_eval)
    intervals_by_method, metadata = build_task1_results(
        model, x_train, x_cal, y_cal, x_eval
    )
    summary = summarise_task1_results(
        intervals_by_method,
        p_star_eval,
        metadata["selected_bins"],
        p_hat_eval=p_hat_eval,
    )
    sort_idx = np.argsort(x_eval)
    return {
        "x": x_eval[sort_idx],
        "p_star": p_true_univariate(x_eval[sort_idx]),
        "p_hat": predict_binary_proba(model, x_eval[sort_idx]),
        "intervals": {name: arr[sort_idx] for name, arr in intervals_by_method.items()},
        "summary": summary,
    }


def compute_misspecified_tree():
    """Compute the decision-tree misspecification illustration."""
    rng = np.random.default_rng(MISSPECIFIED_SEED)
    x_train, y_train = sample_tree_data(MISSPECIFIED_N_TRAIN, rng)
    x_cal, y_cal = sample_tree_data(MISSPECIFIED_N_CAL, rng)
    model = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=25, random_state=MISSPECIFIED_SEED
    )
    model.fit(ensure_2d(x_train), y_train)
    partition = build_picpi_partition(
        predict_binary_proba(model, x_cal), y_cal, num_bins=100, mode="empirical"
    )
    x_grid = np.linspace(-2.0, 2.0, 600)
    return {
        "x": x_grid,
        "p_star": p_true_tree(x_grid),
        "p_hat": predict_binary_proba(model, x_grid),
        "intervals": assign_partition_interval(
            predict_binary_proba(model, x_grid), partition, expand=PICPI_EPSILON
        ),
    }


def run_multivariate_rep(seed):
    """Run one seeded replication of the multivariate comparison."""
    (x_train, y_train), (x_cal, y_cal), (x_eval, _y_eval) = generate_multivariate_split(
        MULTIVARIATE_N_TRAIN,
        MULTIVARIATE_N_CAL,
        MULTIVARIATE_N_EVAL,
        seed=seed,
    )
    model = fit_shared_binary_model(x_train, y_train)
    p_star_eval = p_true_multivariate(x_eval)
    p_hat_eval = predict_binary_proba(model, x_eval)
    intervals_by_method, metadata = build_task1_results(
        model, x_train, x_cal, y_cal, x_eval, picpi_mode="empirical", picpi_delta=None
    )
    return summarise_task1_results(
        intervals_by_method,
        p_star_eval,
        metadata["selected_bins"],
        p_hat_eval=p_hat_eval,
    )


def compute_multivariate_table(n_mc: int = N_MC_TABLE):
    """Run and combine the requested multivariate Monte Carlo replications."""
    frames = []
    for seed in tqdm(range(n_mc), desc="Task 1 multivariate replications"):
        rep_df = run_multivariate_rep(seed)
        rep_df["seed"] = seed
        frames.append(rep_df)
    return pd.concat(frames, ignore_index=True)


def aggregate_multivariate_table(mc_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize Task 1 Monte Carlo metrics by method."""
    return (
        mc_results.groupby("Method", sort=False)
        .agg(
            average_length_mean=("Average length", "mean"),
            average_length_sd=("Average length", "std"),
            ece_mean=("ECE", "mean"),
            ece_sd=("ECE", "std"),
            coverage_01_mean=("Coverage [0,1]", "mean"),
            coverage_01_sd=("Coverage [0,1]", "std"),
            coverage_pstar_mean=("Coverage p*", "mean"),
            coverage_pstar_sd=("Coverage p*", "std"),
        )
        .reindex(TASK1_METHOD_ORDER)
        .reset_index()
    )


def aggregate_ece_compare_table(mc_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize midpoint ECE and mean-score ECE on the same replications."""
    if "ECE mean p-hat" not in mc_results.columns:
        raise ValueError("Monte Carlo results do not include ECE mean p-hat")
    summary = (
        mc_results.groupby("Method", sort=False)
        .agg(
            ece_midpoint_mean=("ECE", "mean"),
            ece_midpoint_sd=("ECE", "std"),
            ece_mean_phat_mean=("ECE mean p-hat", "mean"),
            ece_mean_phat_sd=("ECE mean p-hat", "std"),
        )
        .reindex(TASK1_METHOD_ORDER)
        .reset_index()
    )
    min_mid = float(summary["ece_midpoint_mean"].min())
    min_phat = float(summary["ece_mean_phat_mean"].min())
    summary["relative_ece_midpoint"] = summary["ece_midpoint_mean"] / min_mid
    summary["relative_ece_mean_phat"] = summary["ece_mean_phat_mean"] / min_phat
    return summary


def save_task1(
    univariate: dict[str, object],
    tree: dict[str, object],
    mc_results: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Save all Task 1 arrays and tabular results."""
    output_dir = Path(output_dir) if output_dir is not None else cached("task1")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "univariate_visualization.npz",
        x=univariate["x"],
        p_star=univariate["p_star"],
        p_hat=univariate["p_hat"],
        **{
            f"interval_{idx}": univariate["intervals"][name]
            for idx, name in enumerate(TASK1_METHOD_ORDER)
        },
    )
    univariate["summary"].to_csv(
        output_dir / "univariate_summary.csv", index=False
    )
    np.savez(
        output_dir / "misspecified_tree.npz",
        x=tree["x"],
        p_star=tree["p_star"],
        p_hat=tree["p_hat"],
        intervals=tree["intervals"],
    )
    mc_results.to_csv(output_dir / "multivariate_mc_reps.csv", index=False)
    aggregate_multivariate_table(mc_results).to_csv(
        output_dir / "multivariate_mc_summary.csv", index=False
    )
    if "ECE mean p-hat" in mc_results.columns:
        aggregate_ece_compare_table(mc_results).to_csv(
            output_dir / "multivariate_ece_compare_summary.csv", index=False
        )
    method_names = np.array(TASK1_METHOD_ORDER, dtype=object)
    np.save(output_dir / "method_order.npy", method_names, allow_pickle=True)
    return output_dir


def load_task1(output_dir: Path | None = None) -> dict[str, object]:
    """Load the stored Task 1 arrays and tables used for plotting."""
    output_dir = Path(output_dir) if output_dir is not None else cached("task1")
    uni = np.load(output_dir / "univariate_visualization.npz")
    methods = np.load(output_dir / "method_order.npy", allow_pickle=True).tolist()
    tree = np.load(output_dir / "misspecified_tree.npz")
    return {
        "univariate": {
            "x": uni["x"],
            "p_star": uni["p_star"],
            "p_hat": uni["p_hat"],
            "intervals": {
                name: uni[f"interval_{idx}"] for idx, name in enumerate(methods)
            },
            "summary": pd.read_csv(output_dir / "univariate_summary.csv"),
        },
        "tree": {
            "x": tree["x"],
            "p_star": tree["p_star"],
            "p_hat": tree["p_hat"],
            "intervals": tree["intervals"],
        },
        "mc_results": pd.read_csv(output_dir / "multivariate_mc_reps.csv"),
        "mc_summary": pd.read_csv(output_dir / "multivariate_mc_summary.csv"),
        "ece_compare_summary": (
            pd.read_csv(output_dir / "multivariate_ece_compare_summary.csv")
            if (output_dir / "multivariate_ece_compare_summary.csv").exists()
            else None
        ),
    }
