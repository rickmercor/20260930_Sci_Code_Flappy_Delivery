"""
Longitudinal exposure certificate

A normalized exposure t prints a location when t*I>=1. Each bright location must print with probability at least 1-p; each dark location must remain below threshold with probability at least 1-p, separately at every declared longitudinal plane. quantiles[...,0] and [...,1] contain the p and 1-p intensity quantiles. Intersect these marginal exposure intervals with t<=dose_cap. Efficiency at a plane uses the ensemble mean bright-region intensity divided by incident power. Return the intersection endpoints and the minimum plane efficiency. This marginal certificate does not assume different image pixels are independent. Exposure endpoints denote the infimum and supremum; deterministic dark thresholds can make the upper endpoint open.

Returns
-------
float ndarray (3,). [lower exposure,upper exposure,minimum plane efficiency], all dimensionless. lower is +inf when a bright p-quantile is zero; an empty interval may have upper<=lower.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exposure_window(
    statistics: "np.ndarray",
    quantiles: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    dose_cap: float,
) -> "np.ndarray":
    """Longitudinal exposure certificate.

    A normalized exposure t prints a location when t*I>=1. Each bright
    location must print with probability at least 1-p; each dark location
    must remain below threshold with probability at least 1-p, separately
    at every declared longitudinal plane. quantiles[...,0] and [...,1]
    contain the p and 1-p intensity quantiles. Intersect these marginal
    exposure intervals with t<=dose_cap. Efficiency at a plane uses the
    ensemble mean bright-region intensity divided by incident power. Return
    the intersection endpoints and the minimum plane efficiency. This
    marginal certificate uses per-location probabilities. Exposure
    endpoints denote the infimum and supremum; deterministic dark
    thresholds can make the upper endpoint open.

    Parameters
    statistics : float ndarray (Z,M,5): per-plane rows [mean U,mean V,var
    U,cov(U,V),var V].
    quantiles : float ndarray (Z,M,2): nonnegative intensity quantiles at
    [p,1-p], with p<1/2.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    At least one dark observation is also required.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    incident_power : positive float: source-area-weighted sum of incident
    intensity.
    dose_cap : positive float: maximum dimensionless normalized exposure.

    Returns
    float ndarray (3,). [lower exposure,upper exposure,minimum plane
    efficiency], all dimensionless. lower is +inf when a bright p-quantile
    is zero; an empty interval may have upper<=lower.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exposure_window(
    statistics: "np.ndarray",
    quantiles: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    dose_cap: float,
) -> "np.ndarray":
    statistics = np.asarray(statistics, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    bright = np.asarray(bright, dtype=bool)
    if (
        statistics.ndim != 3
        or statistics.shape[2] != 5
        or quantiles.shape != statistics.shape[:2] + (2,)
        or bright.shape != (statistics.shape[1],)
        or not bright.any()
        or bright.all()
    ):
        raise ValueError(
            "Incompatible plane statistics, quantiles or bright/dark mask."
        )
    if (
        not all(
            np.isfinite(x).all()
            for x in [
                statistics,
                quantiles,
                pixel_area,
                incident_power,
                dose_cap,
            ]
        )
        or min(pixel_area, incident_power, dose_cap) <= 0
        or np.any(quantiles < 0)
        or np.any(quantiles[:, :, 0] > quantiles[:, :, 1])
    ):
        raise ValueError("Invalid physical scales or ordered quantiles.")
    qb = quantiles[:, bright, 0].min()
    qd = quantiles[:, ~bright, 1].max()
    lower = 1 / qb if qb > 0 else np.inf
    upper = min(dose_cap, 1 / qd) if qd > 0 else dose_cap
    means = (
        np.sum(statistics[:, :, :2] ** 2, axis=2)
        + statistics[:, :, 2]
        + statistics[:, :, 4]
    )
    eta = np.min(
        pixel_area * np.sum(means[:, bright], axis=1) / incident_power
    )
    return np.array([lower, upper, eta])

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

s = np.zeros((2, 4, 5))
s[:, :, 0] = np.array([[1.1, 0.3, 1.2, 0.2], [1.2, 0.2, 1.0, 0.3]])
s[:, :, 2] = 0.01
s[:, :, 4] = 0.02
B = np.array([1, 0, 1, 0], bool)
q = np.array(
    [
        [[0.8, 1.5], [0.03, 0.2], [0.9, 1.6], [0.02, 0.15]],
        [[0.85, 1.4], [0.04, 0.18], [0.7, 1.3], [0.03, 0.22]],
    ]
)
cap = 20.0
s = s[:1]
q = q[:1]
candidate_args = deepcopy((s, q, B, 0.25, 2.0, cap))
oracle_args = deepcopy((s, q, B, 0.25, 2.0, cap))
"""
            ),
            "call": "exposure_window(*candidate_args)",
            "gold_call": "_oracle_exposure_window(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.zeros((2, 4, 5))
s[:, :, 0] = np.array([[1.1, 0.3, 1.2, 0.2], [1.2, 0.2, 1.0, 0.3]])
s[:, :, 2] = 0.01
s[:, :, 4] = 0.02
B = np.array([1, 0, 1, 0], bool)
q = np.array(
    [
        [[0.8, 1.5], [0.03, 0.2], [0.9, 1.6], [0.02, 0.15]],
        [[0.85, 1.4], [0.04, 0.18], [0.7, 1.3], [0.03, 0.22]],
    ]
)
cap = 20.0
q[1, 2, 0] = 0.25
candidate_args = deepcopy((s, q, B, 0.25, 2.0, cap))
oracle_args = deepcopy((s, q, B, 0.25, 2.0, cap))
"""
            ),
            "call": "exposure_window(*candidate_args)",
            "gold_call": "_oracle_exposure_window(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.zeros((2, 4, 5))
s[:, :, 0] = np.array([[1.1, 0.3, 1.2, 0.2], [1.2, 0.2, 1.0, 0.3]])
s[:, :, 2] = 0.01
s[:, :, 4] = 0.02
B = np.array([1, 0, 1, 0], bool)
q = np.array(
    [
        [[0.8, 1.5], [0.03, 0.2], [0.9, 1.6], [0.02, 0.15]],
        [[0.85, 1.4], [0.04, 0.18], [0.7, 1.3], [0.03, 0.22]],
    ]
)
cap = 20.0
q[1, 3, 1] = 0.6
candidate_args = deepcopy((s, q, B, 0.25, 2.0, cap))
oracle_args = deepcopy((s, q, B, 0.25, 2.0, cap))
"""
            ),
            "call": "exposure_window(*candidate_args)",
            "gold_call": "_oracle_exposure_window(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.zeros((2, 4, 5))
s[:, :, 0] = np.array([[1.1, 0.3, 1.2, 0.2], [1.2, 0.2, 1.0, 0.3]])
s[:, :, 2] = 0.01
s[:, :, 4] = 0.02
B = np.array([1, 0, 1, 0], bool)
q = np.array(
    [
        [[0.8, 1.5], [0.03, 0.2], [0.9, 1.6], [0.02, 0.15]],
        [[0.85, 1.4], [0.04, 0.18], [0.7, 1.3], [0.03, 0.22]],
    ]
)
cap = 20.0
cap = 2.0
candidate_args = deepcopy((s, q, B, 0.25, 2.0, cap))
oracle_args = deepcopy((s, q, B, 0.25, 2.0, cap))
"""
            ),
            "call": "exposure_window(*candidate_args)",
            "gold_call": "_oracle_exposure_window(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.zeros((2, 4, 5))
s[:, :, 0] = np.array([[1.1, 0.3, 1.2, 0.2], [1.2, 0.2, 1.0, 0.3]])
s[:, :, 2] = 0.01
s[:, :, 4] = 0.02
B = np.array([1, 0, 1, 0], bool)
q = np.array(
    [
        [[0.8, 1.5], [0.03, 0.2], [0.9, 1.6], [0.02, 0.15]],
        [[0.85, 1.4], [0.04, 0.18], [0.7, 1.3], [0.03, 0.22]],
    ]
)
cap = 20.0
q[:, ~B, :] = 0.0
candidate_args = deepcopy((s, q, B, 0.25, 2.0, cap))
oracle_args = deepcopy((s, q, B, 0.25, 2.0, cap))
"""
            ),
            "call": "exposure_window(*candidate_args)",
            "gold_call": "_oracle_exposure_window(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.zeros((2, 4, 5))
s[:, :, 0] = np.array([[1.1, 0.3, 1.2, 0.2], [1.2, 0.2, 1.0, 0.3]])
s[:, :, 2] = 0.01
s[:, :, 4] = 0.02
B = np.array([1, 0, 1, 0], bool)
q = np.array(
    [
        [[0.8, 1.5], [0.03, 0.2], [0.9, 1.6], [0.02, 0.15]],
        [[0.85, 1.4], [0.04, 0.18], [0.7, 1.3], [0.03, 0.22]],
    ]
)
cap = 20.0
q[:, ~B, 1] = 2.0
candidate_args = deepcopy((s, q, B, 0.25, 2.0, cap))
oracle_args = deepcopy((s, q, B, 0.25, 2.0, cap))
"""
            ),
            "call": "exposure_window(*candidate_args)",
            "gold_call": "_oracle_exposure_window(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.zeros((2, 4, 5))
s[:, :, 0] = np.array([[1.1, 0.3, 1.2, 0.2], [1.2, 0.2, 1.0, 0.3]])
s[:, :, 2] = 0.01
s[:, :, 4] = 0.02
B = np.array([1, 0, 1, 0], bool)
q = np.array(
    [
        [[0.8, 1.5], [0.03, 0.2], [0.9, 1.6], [0.02, 0.15]],
        [[0.85, 1.4], [0.04, 0.18], [0.7, 1.3], [0.03, 0.22]],
    ]
)
cap = 20.0
q[0, 0, 0] = 0.0
candidate_args = deepcopy((s, q, B, 0.25, 2.0, cap))
oracle_args = deepcopy((s, q, B, 0.25, 2.0, cap))
"""
            ),
            "call": "exposure_window(*candidate_args)",
            "gold_call": "_oracle_exposure_window(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.zeros((2, 4, 5))
s[:, :, 0] = np.array([[1.1, 0.3, 1.2, 0.2], [1.2, 0.2, 1.0, 0.3]])
s[:, :, 2] = 0.01
s[:, :, 4] = 0.02
B = np.array([1, 0, 1, 0], bool)
q = np.array(
    [
        [[0.8, 1.5], [0.03, 0.2], [0.9, 1.6], [0.02, 0.15]],
        [[0.85, 1.4], [0.04, 0.18], [0.7, 1.3], [0.03, 0.22]],
    ]
)
cap = 20.0
s[1, B, 0] *= 0.2
candidate_args = deepcopy((s, q, B, 0.25, 2.0, cap))
oracle_args = deepcopy((s, q, B, 0.25, 2.0, cap))
"""
            ),
            "call": "exposure_window(*candidate_args)",
            "gold_call": "_oracle_exposure_window(*oracle_args)",
            "tol": 1e-06,
        },
    ]
