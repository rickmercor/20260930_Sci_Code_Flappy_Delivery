"""
Integrate the squared gain between two independently partitioned robustness profiles.

For nested measurement families $\mathcal N\subseteq\mathcal M$, projection relaxes the affine robustness constraints, so $R_{\mathcal M}(t)\ge R_{\mathcal N}(t)$. The profiles generally change affine pieces at different mixture parameters. The observable gain is

$$

J=\int_0^1[R_{\mathcal M}(t)-R_{\mathcal N}(t)]^2\,dt.

$$

Integrate on the common refinement of both breakpoint partitions; align pieces by their intervals rather than row index. The integrand is a quadratic on each common interval. Equality over an entire interval contributes zero even when the common resource value is above one. Use local interval coordinates to avoid cancellation when integrating narrow pieces near an endpoint.

Returns
-------
Dimensionless integral of the squared nonnegative difference between the two profiles.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_measurement_gain(
    full_profile: "np.ndarray", reduced_profile: "np.ndarray"
) -> float:
    r"""Integrate the squared gain between two independently partitioned robustness profiles.

    Parameters
    ----------
    full_profile : np.ndarray
        Finite real $(K,4)$ rows $(\ell,r,a,s)$ for the larger-family profile.
    reduced_profile : np.ndarray
        Finite real $(L,4)$ rows of the same form for the smaller-family profile. Both
        inputs must cover $[0,1]$ continuously, have positive widths, nondecreasing slopes
        and values at least one, within absolute $10^{-7}$. Redundant collinear splits are
        allowed.

    Returns
    -------
    gain : float
        Dimensionless integral of the squared nonnegative difference between the two
        profiles.

    Raises
    ------
    ValueError
        If profiles have invalid shapes or finite-real data, invalid coverage or continuity,
        decreasing slopes, values below one, or reduced values exceed full values by more
        than 1e-7.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _checked_segments(profile):
    raw = np.asarray(profile)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] != 4 or np.iscomplexobj(raw):
        raise ValueError("profile must be a nonempty real (K,4) array")
    segments = raw.astype(float)
    if not np.all(np.isfinite(segments)) or np.any(segments[:, 1] <= segments[:, 0]):
        raise ValueError(
            "profile must have finite coefficients and positive interval widths"
        )
    tolerance = 1e-7
    if abs(segments[0, 0]) > tolerance or abs(segments[-1, 1] - 1.0) > tolerance:
        raise ValueError("profile must span [0,1]")
    if np.any(np.abs(segments[1:, 0] - segments[:-1, 1]) > tolerance):
        raise ValueError("profile intervals must meet")
    left_values = segments[:, 2] + segments[:, 3] * segments[:, 0]
    right_values = segments[:, 2] + segments[:, 3] * segments[:, 1]
    if np.any(np.abs(left_values[1:] - right_values[:-1]) > tolerance):
        raise ValueError("profile must be continuous")
    if (
        np.any(np.diff(segments[:, 3]) < -tolerance)
        or min(left_values.min(), right_values.min()) < 1.0 - tolerance
    ):
        raise ValueError("profile must be convex and at least one")
    return segments


def _oracle_integrate_measurement_gain(
    full_profile: "np.ndarray", reduced_profile: "np.ndarray"
) -> float:
    full = _checked_segments(full_profile)
    reduced = _checked_segments(reduced_profile)
    i = j = 0
    left = total = 0.0
    while i < len(full) and j < len(reduced):
        right = min(full[i, 1], reduced[j, 1])
        width = right - left
        offset, slope = full[i, 2:] - reduced[j, 2:]
        value_left = offset + slope * left
        value_right = offset + slope * right
        if min(value_left, value_right) < -1e-7:
            raise ValueError("reduced profile exceeds the full profile")
        total += (
            width
            * (
                value_left * value_left
                + value_left * value_right
                + value_right * value_right
            )
            / 3.0
        )
        left = right
        if full[i, 1] <= right:
            i += 1
        if reduced[j, 1] <= right:
            j += 1
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent scientific cases for this numerical contract."""
    return [
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
full = np.array([[0.,.2,1.4,-2.],[.2,.7,1.,0.],[.7,1.,-.4,2.]])
reduced = np.array([[0.,.1,1.1,-1.],[.1,.9,1.,0.],[.9,1.,.1,1.]])
""",
            "call": "integrate_measurement_gain(full.copy(), reduced.copy())",
            "gold_call": "_oracle_integrate_measurement_gain(full.copy(), reduced.copy())",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
full = np.array([[0.,1.,1.2,.3]])
reduced = np.array([[0.,.3,1.2,.3],[.3,1.,1.2,.3]])
""",
            "call": "integrate_measurement_gain(full.copy(), reduced.copy())",
            "gold_call": "_oracle_integrate_measurement_gain(full.copy(), reduced.copy())",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
full = np.array([[0.,.999,1.,0.],[.999,1.,-998.,1000.]])
reduced = np.array([[0.,1.,1.,0.]])
""",
            "call": "integrate_measurement_gain(full.copy(), reduced.copy())",
            "gold_call": "_oracle_integrate_measurement_gain(full.copy(), reduced.copy())",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
full = np.array([[0.,1.,1.,0.]])
reduced = np.array([[0.,1.,1.1,0.]])

def _reject(fn):
    try:
        fn(full.copy(), reduced.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_reject(integrate_measurement_gain)",
            "gold_call": "_reject(_oracle_integrate_measurement_gain)",
            "tol": 0.0,
        },
    ]
