"""
Return the rescaled node coordinate s_1..s_N of the source for a pathway whose edges have the given intrinsic speeds (edge_speeds[i-1] is the speed of the edge feeding node i), constructed so that the spacing between consecutive nodes is inversely proportional to the intrinsic speed of the connecting edge and the whole rescaled pathway has unit length, with node 0 at s = 0.

Stretching the pathway coordinate where the local kinetics are slow and compressing it where they are fast absorbs edge-to-edge variability, so that a front which stutters in the node index moves at a nearly constant speed in the rescaled coordinate.

Returns
-------
ndarray of float64, shape (N,), the rescaled positions s_1..s_N with s_N = 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rescaled_positions(edge_speeds: "np.ndarray") -> "np.ndarray":
    """Return the rescaled node coordinate s_1..s_N of the source for a pathway whose edges have the given intrinsic speeds (edge_speeds[i-1] is the speed of the edge feeding node i), constructed so that the spacing between consecutive nodes is inversely proportional to the intrinsic speed of the connecting edge and the whole rescaled pathway has unit length, with node 0 at s = 0.

    Parameters
    ----------
    edge_speeds : np.ndarray
        Positive intrinsic speeds, one per edge.

    Returns
    -------
    s : np.ndarray
        Rescaled node positions of length N.

    Raises
    ------
    ValueError
        If edge_speeds is empty, non-finite or contains a non-positive entry.
    """
    return s

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


def _oracle_rescaled_positions(edge_speeds: "np.ndarray") -> "np.ndarray":
    """Eq. (15): s_0 = 0, s_{i+1} = s_i + cbar / c_i with cbar = (sum_k 1/c_k)^-1, so that the
    rescaled pathway has unit length; returns s_1..s_N."""
    c = _as_vector(edge_speeds, "edge_speeds")
    if np.any(c <= 0.0):
        raise ValueError("edge speeds must be positive")
    inv = 1.0 / c
    return np.cumsum(inv) / inv.sum()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nedge_speeds = np.array([0.3, 0.3, 0.6, 1.2])\n",
            "call": "np.asarray(rescaled_positions(edge_speeds))",
            "gold_call": "np.asarray(_oracle_rescaled_positions(edge_speeds))",
        },
        {
            "setup": "import numpy as np\nN = 120\nalpha = 1.0 + 4.0 * np.arange(N) / (N - 1)\nB = np.where(np.arange(N) < N // 2, 3.0, 8.0)\nphi = np.zeros(N)\nedge_speeds = alpha * np.where(B == 3.0, 0.2977, 0.2692)\n",
            "call": "np.asarray(rescaled_positions(edge_speeds))",
            "gold_call": "np.asarray(_oracle_rescaled_positions(edge_speeds))",
        },
        {
            "setup": "import numpy as np\nedge_speeds = np.array([2.0])\n",
            "call": "np.asarray(rescaled_positions(edge_speeds))",
            "gold_call": "np.asarray(_oracle_rescaled_positions(edge_speeds))",
        },
        {
            "setup": "import numpy as np\nedge_speeds = np.array([0.3, 0.0, 0.6])\ndef run_model():\n    try:\n        rescaled_positions(edge_speeds)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_rescaled_positions(edge_speeds)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
