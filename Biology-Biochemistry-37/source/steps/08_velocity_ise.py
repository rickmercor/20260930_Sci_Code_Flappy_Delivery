"""
Compute the velocity integral square error of a speed series over the post-transient window: the trapezoidal integral over time of the squared deviation of the speed from its mean over the window. speeds[j] is the speed on the sampling interval that starts at times[j]; the window consists of the intervals whose starting sample has the wave centre, given as a fraction of the pathway length in centres[j], in [s_lo, s_hi). The mean is the plain average of the speeds in the window.

A single summary statistic of how much the propagation speed fluctuates along the pathway makes it possible to compare the original and the rescaled description of the same signal, and to sweep over degrees of kinetic heterogeneity.

Returns
-------
float, the velocity integral square error over the window.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def velocity_ise(speeds: "np.ndarray", times: "np.ndarray", centres: "np.ndarray",
                         s_lo: float, s_hi: float) -> float:
    """Compute the velocity integral square error of a speed series over the post-transient window: the trapezoidal integral over time of the squared deviation of the speed from its mean over the window. speeds[j] is the speed on the sampling interval that starts at times[j]; the window consists of the intervals whose starting sample has the wave centre, given as a fraction of the pathway length in centres[j], in [s_lo, s_hi). The mean is the plain average of the speeds in the window.

    Parameters
    ----------
    speeds : np.ndarray
        Speed on each sampling interval.
    times : np.ndarray
        Strictly increasing start time of each sampling interval.
    centres : np.ndarray
        Wave-centre position at each start time, as a fraction of the pathway length.
    s_lo : float
        Lower window bound in [0, 1] (inclusive).
    s_hi : float
        Upper window bound in [0, 1] (exclusive), greater than s_lo.

    Returns
    -------
    vise : float
        Velocity integral square error.

    Raises
    ------
    ValueError
        If the arrays differ in length, the times do not increase, the bounds are invalid, or fewer than two intervals fall in the window.
    """
    return vise

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _as_vector(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty 1-D array of finite numbers")
    return a


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def _oracle_velocity_ise(speeds: "np.ndarray", times: "np.ndarray", centres: "np.ndarray",
                         s_lo: float, s_hi: float) -> float:
    """Sec. 3.2 VISE: trapezoidal integral over the post-transient window of the squared deviation of
    the speed from its window mean. The window is the set of sampling intervals whose START time has
    the wave centre (a fraction of the pathway length) in [s_lo, s_hi); speeds[j] belongs to the
    interval starting at times[j]."""
    c = _as_vector(speeds, "speeds"); t = _as_vector(times, "times"); s = _as_vector(centres, "centres")
    if not (c.size == t.size == s.size) or np.any(np.diff(t) <= 0.0):
        raise ValueError("speeds, times and centres must have equal length and times must increase")
    lo = _check_scalar(s_lo, "s_lo", 0.0, 1.0); hi = _check_scalar(s_hi, "s_hi", 0.0, 1.0)
    if hi <= lo:
        raise ValueError("need s_lo < s_hi")
    mask = (s >= lo) & (s < hi)
    if mask.sum() < 2:
        raise ValueError("fewer than two sampling intervals fall in the window")
    cw = c[mask]; tw = t[mask]
    dev = cw - cw.mean()
    return float(np.trapezoid(dev ** 2, tw))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntimes = np.arange(10.0)\nspeeds = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.3, 0.7, 1.0, 1.0])\ncentres = np.linspace(0.0, 0.9, 10)\n",
            "call": "velocity_ise(speeds, times, centres, 0.15, 0.85)",
            "gold_call": "_oracle_velocity_ise(speeds, times, centres, 0.15, 0.85)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ntimes = np.arange(0.0, 20.0, 2.0)\nspeeds = np.sin(np.arange(10.0))\ncentres = np.linspace(0.05, 0.95, 10)\n",
            "call": "velocity_ise(speeds, times, centres, 0.15, 0.85)",
            "gold_call": "_oracle_velocity_ise(speeds, times, centres, 0.15, 0.85)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ntimes = np.arange(6.0)\nspeeds = np.array([2.0, 2.0, 2.0, 5.0, 2.0, 2.0])\ncentres = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])\n",
            "call": "velocity_ise(speeds, times, centres, 0.15, 0.85)",
            "gold_call": "_oracle_velocity_ise(speeds, times, centres, 0.15, 0.85)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ntimes = np.arange(6.0)\nspeeds = np.ones(6)\ncentres = np.linspace(0.0, 0.5, 6)\ndef run_model():\n    try:\n        velocity_ise(speeds, times, centres, 0.85, 0.15)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_velocity_ise(speeds, times, centres, 0.85, 0.15)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
