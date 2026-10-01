"""
Evaluate the right-hand side of the node equations of the cascade for the current activity vector x, returning dx_i/dt for every node. Node 1 is driven by the constant upstream input x_in and node i > 1 by node i - 1; the edge feeding node i has parameters alpha_i, B_i = 2 beta_i - 1 and phi_i. Use exactly the activation/inactivation form of the node equation given in the task.

Each transition in a canonical feed-forward pathway is a balance between an activating term driven by the active fraction of the upstream node and an inactivating term driven by its inactive fraction, both saturating in the downstream activity in a Michaelis-Menten fashion. The bias parameter sets the relative weight of the two drives.

Returns
-------
ndarray of float64, shape (N,), the time derivative of every node activity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cascade_rate(x: "np.ndarray", x_in: float, alpha: "np.ndarray", B: "np.ndarray",
                         phi: "np.ndarray") -> "np.ndarray":
    """Evaluate the right-hand side of the node equations of the cascade for the current activity vector x, returning dx_i/dt for every node. Node 1 is driven by the constant upstream input x_in and node i > 1 by node i - 1; the edge feeding node i has parameters alpha_i, B_i = 2 beta_i - 1 and phi_i. Use exactly the activation/inactivation form of the node equation given in the task.

    Parameters
    ----------
    x : np.ndarray
        Current node activities x_1..x_N, each with |x_i| < B_i.
    x_in : float
        Constant upstream input x_0 in [-1, 1].
    alpha : np.ndarray
        Edge timescale parameters alpha_i > 0, one per node.
    B : np.ndarray
        Edge saturation parameters B_i > 1, one per node.
    phi : np.ndarray
        Edge bias parameters phi_i in [-1, 1], one per node.

    Returns
    -------
    dxdt : np.ndarray
        Array of length N with dx_i/dt.

    Raises
    ------
    ValueError
        If the parameter arrays differ in length or violate alpha > 0, B > 1, |phi| <= 1; if x has the wrong length or |x_i| >= B_i; or if x_in is outside [-1, 1].
    """
    return dxdt

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


def _check_params(alpha, B, phi):
    a = _as_vector(alpha, "alpha"); b = _as_vector(B, "B"); p = _as_vector(phi, "phi")
    if not (a.size == b.size == p.size):
        raise ValueError("alpha, B and phi must have the same length")
    if np.any(a <= 0.0) or np.any(b <= 1.0) or np.any(np.abs(p) > 1.0):
        raise ValueError("need alpha > 0, B > 1 and |phi| <= 1 on every edge")
    return a, b, p


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def _oracle_cascade_rate(x: "np.ndarray", x_in: float, alpha: "np.ndarray", B: "np.ndarray",
                         phi: "np.ndarray") -> "np.ndarray":
    """Eq. (1) for every node, with the constant upstream input x_in feeding node 1 and
    beta_i = (B_i + 1) / 2. Activation is driven by (1 + x_{i-1}) and weighted (1 + phi_i)/4,
    inactivation by (1 - x_{i-1}) and weighted (1 - phi_i)/4; the saturating factors are
    alpha beta (1 -/+ x_i) / (2 beta - (1 +/- x_i))."""
    a, b, p = _check_params(alpha, B, phi)
    xv = _as_vector(x, "x")
    if xv.size != a.size or np.any(np.abs(xv) >= b):
        raise ValueError("x must have one entry per edge with |x_i| < B_i (the saturating denominators must stay positive)")
    xin = _check_scalar(x_in, "x_in", -1.0, 1.0)
    beta = (b + 1.0) / 2.0
    up = np.concatenate(([xin], xv[:-1]))
    act = (1.0 + p) / 4.0 * (1.0 + up) * a * beta * (1.0 - xv) / (2.0 * beta - (1.0 + xv))
    ina = (1.0 - p) / 4.0 * (1.0 - up) * a * beta * (1.0 + xv) / (2.0 * beta - (1.0 - xv))
    return act - ina

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.full(N, 1.5)\nB = np.full(N, 3.0)\nphi = np.full(N, 0.1)\nx = np.linspace(0.9, -0.9, N)\n",
            "call": "np.asarray(cascade_rate(x, 1.0, alpha, B, phi))",
            "gold_call": "np.asarray(_oracle_cascade_rate(x, 1.0, alpha, B, phi))",
        },
        {
            "setup": "import numpy as np\nN = 40\nalpha = 1.0 + 2.0 * np.arange(N) / (N - 1)\nB = np.where(np.arange(N) < 20, 2.5, 6.0)\nphi = np.full(N, -0.05)\nx = np.full(N, -1.0)\n",
            "call": "np.asarray(cascade_rate(x, 1.0, alpha, B, phi))",
            "gold_call": "np.asarray(_oracle_cascade_rate(x, 1.0, alpha, B, phi))",
        },
        {
            "setup": "import numpy as np\nN = 120\nalpha = 1.0 + 4.0 * np.arange(N) / (N - 1)\nB = np.where(np.arange(N) < N // 2, 3.0, 8.0)\nphi = np.zeros(N)\nx = np.tanh(np.linspace(3.0, -3.0, N))\n",
            "call": "np.asarray(cascade_rate(x, 1.0, alpha, B, phi))",
            "gold_call": "np.asarray(_oracle_cascade_rate(x, 1.0, alpha, B, phi))",
        },
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.full(N, 1.5)\nB = np.full(N, 3.0)\nphi = np.full(N, 0.1)\nx = np.zeros(N)\ndef run_model():\n    try:\n        cascade_rate(x, 1.5, alpha, B, phi)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_cascade_rate(x, 1.5, alpha, B, phi)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
