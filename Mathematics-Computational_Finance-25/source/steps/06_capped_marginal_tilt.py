"""
Apply the exponential tilt that drives one entry of one marginal of a coupling towards its prescribed mass, with the log-tilt magnitude limited by a cap supplied by the caller. `axis` names the axis that is SUMMED OVER to form the marginal being corrected, exactly as in numpy: axis=1 corrects an entry of P.sum(axis=1), so it rescales the row P[index, :]; axis=0 corrects an entry of P.sum(axis=0), so it rescales the column P[:, index]. A slice carrying no mass is returned unchanged.

A marginal constraint row has an indicator coefficient vector, so the exponential tilt that meets it exactly is a single scalar rescaling of one slice of the coupling, available in closed form with no iteration. The cap makes the correction partial. The axis convention follows numpy's: the axis argument is the one passed to P.sum(), so P.sum(axis=1) is the vector of row sums and axis=1 therefore addresses a row.

Returns
-------
ndarray with the same shape as P, float64: the coupling after the tilt. With axis=1 only P[index, :] changes; with axis=0 only P[:, index] changes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def capped_marginal_tilt(P, axis, index, target, cap):
    """Apply the exponential tilt that drives one entry of one marginal of a coupling towards its prescribed mass, with the log-tilt magnitude limited by a cap supplied by the caller. `axis` names the axis that is SUMMED OVER to form the marginal being corrected, exactly as in numpy: axis=1 corrects an entry of P.sum(axis=1), so it rescales the row P[index, :]; axis=0 corrects an entry of P.sum(axis=0), so it rescales the column P[:, index]. A slice carrying no mass is returned unchanged.

    Returns
    -------
    ndarray with the same shape as P, float64: the coupling after the tilt. With axis=1 only P[index, :] changes; with axis=0 only P[:, index] changes.
    """
    return np.array(P, dtype=float, copy=True)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_capped_marginal_tilt(P, axis, index, target, cap):
    P = np.array(P, dtype=float, copy=True)
    if P.ndim != 2:
        raise ValueError("P must be 2-D")
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    index = int(index)
    if not (0 <= index < P.shape[1 - axis]):
        raise ValueError("index out of range for the requested axis")
    cap = float(cap)
    if not (cap > 0.0) or not np.isfinite(cap):
        raise ValueError("cap must be finite and strictly positive")
    target = float(target)
    if not (target >= 0.0) or not np.isfinite(target):
        raise ValueError("target must be finite and non-negative")
    sl = P[index, :] if axis == 1 else P[:, index]
    s = float(sl.sum())
    if s <= 0.0:
        return P
    theta = float(np.clip(np.log(max(target, 1e-300) / s), -cap, cap))
    sl *= float(np.exp(theta))
    return P

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nrng = np.random.default_rng(0)\nP = np.full((4, 4), 1.0/16.0)",
            "call": "capped_marginal_tilt(P, 1, 2, 0.4, 0.02)",
            "gold_call": "_oracle_capped_marginal_tilt(P, 1, 2, 0.4, 0.02)",
        },
        {
            "setup": "import numpy as np\nP = np.full((4, 4), 1.0/16.0)",
            "call": "capped_marginal_tilt(P, 0, 0, 0.25, 50.0)",
            "gold_call": "_oracle_capped_marginal_tilt(P, 0, 0, 0.25, 50.0)",
        },
        {
            "setup": "import numpy as np\nP = np.zeros((3, 3)); P[0, 0] = 1.0",
            "call": "capped_marginal_tilt(P, 1, 2, 0.5, 0.02)",
            "gold_call": "_oracle_capped_marginal_tilt(P, 1, 2, 0.5, 0.02)",
        },
        {
            "setup": "import numpy as np\nP = np.arange(1.0, 13.0).reshape(3, 4); P = P/P.sum()",
            "call": "capped_marginal_tilt(P, 1, 0, 0.9, 5.0)",
            "gold_call": "_oracle_capped_marginal_tilt(P, 1, 0, 0.9, 5.0)",
        },
        {
            "setup": "import numpy as np\nP = np.arange(1.0, 13.0).reshape(3, 4); P = P/P.sum()",
            "call": "capped_marginal_tilt(P, 0, 3, 0.9, 5.0)",
            "gold_call": "_oracle_capped_marginal_tilt(P, 0, 3, 0.9, 5.0)",
        },
    ]
