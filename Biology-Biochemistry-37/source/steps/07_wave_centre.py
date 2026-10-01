"""
Return the position of the wave centre of a sampled profile: the position at which the profile, interpolated piecewise-linearly between consecutive nodes with the upstream input counted as node 0 at position 0, first crosses zero coming from the input side. Return the terminal position if the profile never crosses zero.

The zero level of the activity variable is the balance point between the two molecular forms, so the location where a front crosses it is a natural marker of how far the signal has travelled.

Returns
-------
float, the wave-centre position in the units of the supplied positions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wave_centre(positions: "np.ndarray", profile: "np.ndarray", x_in: float) -> float:
    """Return the position of the wave centre of a sampled profile: the position at which the profile, interpolated piecewise-linearly between consecutive nodes with the upstream input counted as node 0 at position 0, first crosses zero coming from the input side. Return the terminal position if the profile never crosses zero.

    Parameters
    ----------
    positions : np.ndarray
        Strictly increasing positive node positions.
    profile : np.ndarray
        Node activities at one sample.
    x_in : float
        Constant upstream input in [-1, 1].

    Returns
    -------
    centre : float
        Wave-centre position.

    Raises
    ------
    ValueError
        If the arrays differ in length, the positions are not strictly increasing and positive, or x_in is outside [-1, 1].
    """
    return centre

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


def _oracle_wave_centre(positions: "np.ndarray", profile: "np.ndarray", x_in: float) -> float:
    """Position at which the linearly interpolated profile (input as node 0 at position 0) first
    crosses zero coming from the input side; the terminal position if it never does."""
    pos = _as_vector(positions, "positions"); prof = _as_vector(profile, "profile")
    if pos.size != prof.size or np.any(np.diff(pos) <= 0.0) or pos[0] <= 0.0:
        raise ValueError("positions must be increasing and positive with one entry per profile value")
    xin = _check_scalar(x_in, "x_in", -1.0, 1.0)
    P = np.concatenate(([0.0], pos)); V = np.concatenate(([xin], prof))
    idx = np.where((V[:-1] > 0.0) & (V[1:] <= 0.0))[0]
    if idx.size == 0:
        return float(P[-1])
    i = int(idx[0])
    return float(P[i] + (0.0 - V[i]) * (P[i + 1] - P[i]) / (V[i + 1] - V[i]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.full(N, 1.5)\nB = np.full(N, 3.0)\nphi = np.full(N, 0.1)\nprofiles = _oracle_simulate_cascade(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 100000)\npos = np.arange(1, 31, dtype=float)\nj = profiles.shape[0] // 2\n",
            "call": "wave_centre(pos, profiles[j], 1.0)",
            "gold_call": "_oracle_wave_centre(pos, profiles[j], 1.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\npos = np.arange(1, 31, dtype=float)\nprofile = np.tanh(np.linspace(2.0, -2.0, 30))\nj = 0\nprofiles = profile[None, :]\n",
            "call": "wave_centre(pos, profiles[j], 1.0)",
            "gold_call": "_oracle_wave_centre(pos, profiles[j], 1.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.full(N, 1.5)\nB = np.full(N, 3.0)\nphi = np.full(N, 0.1)\nprofiles = _oracle_simulate_cascade(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 100000)\npos = np.cumsum(np.linspace(0.5, 2.0, 30)) / np.sum(np.linspace(0.5, 2.0, 30))\nj = profiles.shape[0] - 1\n",
            "call": "wave_centre(pos, profiles[j], 1.0)",
            "gold_call": "_oracle_wave_centre(pos, profiles[j], 1.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\npos = np.arange(1, 31, dtype=float)\nprofiles = np.full((1, 30), 0.4)\nj = 0\n",
            "call": "wave_centre(pos, profiles[j], 1.0)",
            "gold_call": "_oracle_wave_centre(pos, profiles[j], 1.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\npos = np.arange(1, 31, dtype=float)\nprofile = np.zeros(30)\ndef run_model():\n    try:\n        wave_centre(pos, profile, 2.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_wave_centre(pos, profile, 2.0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
