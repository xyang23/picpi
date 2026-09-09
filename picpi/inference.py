"""Assign predicted probabilities to calibrated PICPI intervals."""

from __future__ import annotations

from typing import Literal, Sequence


def intervals_containing(
    probability: float,
    intervals: Sequence[tuple[float, float]],
    *,
    mode: Literal["empirical", "population"],
) -> list[tuple[float, float]]:
    """Return the calibrated intervals containing a predicted probability.

    Empirical-mode intervals have the form ``(left, right]``, except that zero
    is included in an interval starting at zero. Population-mode intervals are
    closed and may overlap, so more than one interval can be returned.
    """
    if mode not in ("empirical", "population"):
        raise ValueError("mode must be 'empirical' or 'population'")

    if mode == "empirical":
        return [
            (float(left), float(right))
            for left, right in intervals
            if left < probability <= right
            or (left == 0.0 and probability == 0.0)
        ]

    return [
        (float(left), float(right))
        for left, right in intervals
        if left <= probability <= right
    ]
