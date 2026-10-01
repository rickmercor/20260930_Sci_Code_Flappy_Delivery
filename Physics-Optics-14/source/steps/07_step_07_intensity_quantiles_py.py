"""
Gaussian-closure intensity quantiles

For each row, (U,V) has the stated bivariate Gaussian law, including degenerate positive-semidefinite laws, and intensity is U^2+V^2. Return the generalized inverse CDF inf{x:Pr(I<=x)>=p} for every supplied probability. This is numerical inversion of the intensity law underlying the source Eq. (1), with absolute probability error at most 1e-7 for nonsingular inputs. A zero-covariance law has its deterministic intensity as every quantile.

Returns
-------
float ndarray (M,Q). Intensity quantiles, with row order from statistics and column order from probabilities; same intensity units as the squared field.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def intensity_quantiles(
    statistics: "np.ndarray", probabilities: "np.ndarray"
) -> "np.ndarray":
    """Gaussian-closure intensity quantiles.

    For each row, (U,V) has the stated bivariate Gaussian law, including
    degenerate positive-semidefinite laws, and intensity is U^2+V^2. Return
    the generalized inverse CDF inf{x:Pr(I<=x)>=p} for every supplied
    probability. This is numerical inversion of the intensity law
    underlying the source Eq. (1), with absolute probability error at most
    1e-7 for nonsingular inputs. A zero-covariance law has its
    deterministic intensity as every quantile.

    Parameters
    statistics : float ndarray (M,5): rows [mean U,mean V,var
    U,cov(U,V),var V]; finite, positive-semidefinite covariances.
    probabilities : float ndarray (Q,): probabilities strictly between 0
    and 1, output order preserved.

    Returns
    float ndarray (M,Q). Intensity quantiles, with row order from
    statistics and column order from probabilities; same intensity units as
    the squared field.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.optimize import brentq


def _intensity_cdf(threshold, row):
    if threshold < 0:
        return 0.0
    covariance = np.array([[row[2], row[3]], [row[3], row[4]]])
    eigenvalues, basis = np.linalg.eigh(covariance)
    mean = basis.T @ np.asarray(row[:2])
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if eigenvalues[1] == 0:
        return float(mean @ mean <= threshold)
    # Only genuinely degenerate laws use the rank-one limit. An eigenvalue
    # ratio cannot bound the effect of a narrow direction on a displaced tail.
    if eigenvalues[0] == 0:
        remaining = threshold - mean[0] ** 2
        if remaining <= 0:
            return 0.0
        radius = np.sqrt(remaining)
        scale = np.sqrt(eigenvalues[1])
        return float(
            ndtr((radius - mean[1]) / scale)
            - ndtr((-radius - mean[1]) / scale)
        )
    scales = np.sqrt(eigenvalues)
    # Condition on the most strongly displaced standardized coordinate.
    # This resolves narrow displaced tails over an O(1) normal integration
    # range.
    axis = int(np.argmax(np.abs(mean) / scales))
    other = 1 - axis
    center, scale = mean[axis], scales[axis]
    radius = np.sqrt(threshold)
    difference = threshold - center**2
    upper = (
        difference / (scale * (radius + center))
        if center >= 0 and radius + center
        else (radius - center) / scale
    )
    lower = (
        -difference / (scale * (radius - center))
        if center < 0
        else (-radius - center) / scale
    )
    lower, upper = max(-11.0, lower), min(11.0, upper)
    if lower >= upper:
        return 0.0

    def _density(z):
        # Expand the square to avoid subtracting two nearly equal full squares.
        remaining = max(
            difference - 2 * center * scale * z - scale**2 * z**2, 0.0
        )
        half_width = np.sqrt(remaining)
        conditional = ndtr((half_width - mean[other]) / scales[other]) - ndtr(
            (-half_width - mean[other]) / scales[other]
        )
        return np.exp(-z * z / 2) / np.sqrt(2 * np.pi) * conditional

    points = [
        z
        for z in [-8.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 8.0]
        if lower < z < upper
    ]
    return float(
        quad(
            _density,
            lower,
            upper,
            points=points,
            epsabs=2e-11,
            epsrel=2e-11,
            limit=250,
        )[0]
    )


def _oracle_intensity_quantiles(
    statistics: "np.ndarray", probabilities: "np.ndarray"
) -> "np.ndarray":
    statistics = np.asarray(statistics, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        statistics.ndim != 2
        or statistics.shape[1] != 5
        or statistics.shape[0] == 0
        or probabilities.ndim != 1
        or probabilities.size == 0
        or not np.isfinite(statistics).all()
        or not np.isfinite(probabilities).all()
        or np.any((probabilities <= 0) | (probabilities >= 1))
    ):
        raise ValueError("Invalid Gaussian statistics or probabilities.")
    result = np.zeros((len(statistics), len(probabilities)))
    for j, row in enumerate(statistics):
        sigma = np.array([[row[2], row[3]], [row[3], row[4]]])
        ev = np.linalg.eigvalsh(sigma)
        if ev[0] < -1e-12 * max(1.0, ev[1]):
            raise ValueError("Covariance must be positive semidefinite.")
        mean = row[0] ** 2 + row[1] ** 2 + row[2] + row[4]
        if ev[1] <= 0:
            result[j] = max(mean, 0.0)
            continue
        # Scale to a mean-one intensity law, retaining the noncircular
        # covariance.
        scale = max(mean, np.finfo(float).tiny)
        normalized = np.array(
            [
                row[0] / np.sqrt(scale),
                row[1] / np.sqrt(scale),
                row[2] / scale,
                row[3] / scale,
                row[4] / scale,
            ]
        )
        for k, prob in enumerate(probabilities):
            hi = 2.0
            while _intensity_cdf(hi, normalized) < prob:
                hi *= 2
            result[j, k] = scale * brentq(
                lambda q: _intensity_cdf(q, normalized) - prob,
                0.0,
                hi,
                xtol=2e-14,
                rtol=2e-14,
            )
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independently initialized differential cases."""
    return [
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.0, 0.0, 0.5, 0.0, 0.5]])
p = np.array([0.01, 0.5, 0.99])
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "intensity_quantiles(*candidate_args)",
            "gold_call": "_oracle_intensity_quantiles(*oracle_args)",
            "tol": 2e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.4, -0.3, 0.12, 0.0, 0.12]])
p = np.array([0.05, 0.5, 0.95])
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "intensity_quantiles(*candidate_args)",
            "gold_call": "_oracle_intensity_quantiles(*oracle_args)",
            "tol": 2e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.0, 0.0, 0.03, 0.0, 0.7]])
p = np.array([0.1, 0.9])
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "intensity_quantiles(*candidate_args)",
            "gold_call": "_oracle_intensity_quantiles(*oracle_args)",
            "tol": 2e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.2, 0.8, 0.3, -0.11, 0.2]])
p = np.array([0.02, 0.8, 0.97])
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "intensity_quantiles(*candidate_args)",
            "gold_call": "_oracle_intensity_quantiles(*oracle_args)",
            "tol": 2e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.4, 0.0, 0.3, 0.0, 0.0]])
p = np.array([0.05, 0.95])
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "intensity_quantiles(*candidate_args)",
            "gold_call": "_oracle_intensity_quantiles(*oracle_args)",
            "tol": 2e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.2, 0.7, 0.1, 0.2, 0.4]])
p = np.array([0.05, 0.95])
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "intensity_quantiles(*candidate_args)",
            "gold_call": "_oracle_intensity_quantiles(*oracle_args)",
            "tol": 2e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 0.4, 0.0, 0.0, 0.0]])
p = np.array([0.1, 0.9])
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "intensity_quantiles(*candidate_args)",
            "gold_call": "_oracle_intensity_quantiles(*oracle_args)",
            "tol": 2e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array(
    [
        [0.0001, -0.0002, 3e-09, 1e-09, 4e-09],
        [1000.0, 2000.0, 200000.0, -50000.0, 100000.0],
    ]
)
p = np.array([0.9, 0.1, 0.5])
scale = np.array([1e-08, 1000000.0])[:, None]
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "intensity_quantiles(*candidate_args) / scale",
            "gold_call": "_oracle_intensity_quantiles(*oracle_args) / scale",
            "tol": 2e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[1.0, 0.0, 1e-16, 0.0, 0.1]])
p = np.array([0.0001])
candidate_args = deepcopy((s, p))
oracle_args = deepcopy((s, p))
"""
            ),
            "call": "(intensity_quantiles(*candidate_args) - 1.0) / 1e-08",
            "gold_call": (
                "(_oracle_intensity_quantiles(*oracle_args) - 1.0) / 1e-08"
            ),
            "tol": 0.002,
        },
    ]
