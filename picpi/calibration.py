"""PICPI interval construction (empirical and population modes).

Copied from the original repository file
``src/conformal_bandit/calibration.py``. The empirical branch is
Algorithm 1 in the paper; the population branch is Algorithm 2.
"""

from typing import List, Optional, Tuple

import numpy as np


def criteria(x: float, l: float, r: float) -> bool:
    """Check if x is in the interval (l, r]."""
    return l < x <= r


def calibration(
    data: List[Tuple[float, int]],
    p: List[float],
    num_bin: int = 5,
    mode: str = "empirical",
    delta: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Construct PICPI intervals from calibration scores and labels.

    Args:
        data: list of tuples (x_i, y_i). Only the label y_i is used.
        p: predicted probabilities p(x_i)
        num_bin: number of candidate endpoints on [0, 1]
        mode: ``empirical`` (Alg. 1) or ``population`` (Alg. 2)
        delta: failure probability for the population-mode gap
    """
    assert len(data) == len(p), "length of data and p should be the same"
    assert mode in ("empirical", "population"), "undefined mode arg"
    intervals = []

    if mode == "empirical":
        sorted_indices = np.argsort(p)
        p_sorted = [p[i] for i in sorted_indices]
        y_sorted = [data[i][1] for i in sorted_indices]
        t_prev = 0

        for t in np.linspace(0, 1, num=num_bin):
            denom1 = np.sum(
                [1 for i in range(len(data)) if criteria(p_sorted[i], t_prev, t)]
            )
            denom2 = np.sum(
                [1 for i in range(len(data)) if criteria(p_sorted[i], t, 1)]
            )
            numerator1 = np.sum(
                [
                    y_sorted[i]
                    for i in range(len(data))
                    if criteria(p_sorted[i], t_prev, t)
                ]
            )
            numerator2 = np.sum(
                [y_sorted[i] for i in range(len(data)) if criteria(p_sorted[i], t, 1)]
            )

            if denom1 != 0:
                condition1 = criteria(numerator1 / denom1, t_prev, t)
            else:
                condition1 = criteria(0, t_prev, t)

            if denom2 != 0:
                condition2 = criteria(numerator2 / denom2, t, 1)
            else:
                condition2 = True

            if condition1 and condition2:
                intervals.append((t_prev, t))
                t_prev = t

    else:
        assert delta, "please specify delta"
        bin_size = 1 / num_bin
        for i in range(num_bin):
            for j in range(i + 1, num_bin + 1):
                (a, b) = (i * bin_size, j * bin_size)
                denom = np.sum([1 for i in range(len(data)) if a <= p[i] <= b])
                numerator = np.sum(
                    [data[i][1] for i in range(len(data)) if a <= p[i] <= b]
                )
                if denom == 0:
                    condition = False
                else:
                    gen_gap = 2 * np.sqrt(np.log(num_bin**2 / delta) / denom)
                    condition = (a + gen_gap) <= (numerator / denom) <= (b - gen_gap)
                if condition:
                    intervals.append((a, b))

    return intervals
