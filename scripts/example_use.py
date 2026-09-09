#!/usr/bin/env python3
"""Minimal example of constructing empirical- and population-mode PICPIs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picpi.calibration import calibration
from picpi.inference import intervals_containing
from scripts.example_use_helper import make_teaser_example_data, save_example_figure


def main() -> None:
    # The helper supplies example arrays from the Teaser DGP. Replace the first
    # two arrays with predictions and outcomes from your calibration sample.
    (
        model,
        predicted_probabilities,
        observed_outcomes,
        evaluation_probabilities,
        true_evaluation_probabilities,
    ) = make_teaser_example_data()

    # PICPI uses predicted probabilities and observed outcomes from the same
    # held-out calibration sample. For your model, obtain probabilities with
    # model.predict_proba(X_calib)[:, 1] and replace these two arrays.
    calibration_data = list(
        zip(predicted_probabilities.tolist(), observed_outcomes.tolist())
    )

    empirical_picpis = calibration(
        data=calibration_data,
        p=predicted_probabilities.tolist(),
        mode="empirical",
        num_bin=10,
    )
    population_picpis = calibration(
        data=calibration_data,
        p=predicted_probabilities.tolist(),
        mode="population",
        num_bin=10,
        delta=0.1,  # Population mode requires a failure probability.
    )

    print("All PICPIs the algorithm finds:")
    print(
        "  PICPI (population mode):",
        [
            (round(float(left), 3), round(float(right), 3))
            for left, right in population_picpis
        ],
    )
    print(
        "  PICPI (empirical mode):",
        [
            (round(float(left), 3), round(float(right), 3))
            for left, right in empirical_picpis
        ],
    )

    # Inference for new feature rows x:
    #   1. Use the same fitted model to compute p_hat(x).
    #   2. Find the calibrated PICPI interval(s) containing p_hat(x).
    # Replace this array with your own test design matrix X_test. Its number of
    # columns must match the features used to fit your model.
    X_test = np.array([[-1.0], [0.0], [1.0]])
    test_probabilities = model.predict_proba(X_test)[:, 1]

    print("PICPI inference for test inputs")
    for x, probability in zip(X_test, test_probabilities):
        empirical_matches = intervals_containing(
            float(probability), empirical_picpis, mode="empirical"
        )
        population_matches = intervals_containing(
            float(probability), population_picpis, mode="population"
        )
        shortest_population_match = min(
            population_matches,
            key=lambda interval: interval[1] - interval[0],
            default=None,
        )
        print(f"x={x.tolist()}, predicted probability={probability:.3f}")
        if shortest_population_match is None:
            print("  PICPI (population mode): None")
        else:
            left, right = shortest_population_match
            print(
                "  PICPI (population mode):",
                (round(float(left), 3), round(float(right), 3)),
            )
        print(
            "  PICPI (empirical mode):",
            [
                (round(float(left), 3), round(float(right), 3))
                for left, right in empirical_matches
            ],
        )

    output_dir = ROOT / "figures" / "example_use"
    for mode, intervals in (
        ("population", population_picpis),
        ("empirical", empirical_picpis),
    ):
        pdf_path, png_path = save_example_figure(
            mode=mode,
            intervals=intervals,
            predicted_probabilities=predicted_probabilities,
            observed_outcomes=observed_outcomes,
            evaluation_probabilities=evaluation_probabilities,
            true_evaluation_probabilities=true_evaluation_probabilities,
            output_dir=output_dir,
        )
        print(f"PICPI ({mode} mode): {len(intervals)} intervals")
        print(f"Saved {pdf_path}")
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
