"""
Integrate the cascade from the uniform initial activity x_init under the sustained input x_in and return the sampled activity profiles at t_j = j dt for j = 0..J as a (J + 1, N) array. J is the first sample at which the terminal node has moved by more than threshold from x_init (the run stops there); if that never happens, stop at j = max_steps. Integrate accurately (relative tolerance 1e-9 or tighter) so that the sampled profiles are reproducible to about 1e-8.

A sustained boundary stimulus applied to a resting bistable pathway launches a front that switches nodes one after another. The finite pathway distorts the front near its far end, so sampling is stopped as soon as the terminal node responds.

Returns
-------
ndarray of float64, shape (J + 1, N), the sampled activity profiles.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_cascade(alpha: "np.ndarray", B: "np.ndarray", phi: "np.ndarray", x_in: float,
                             x_init: float, dt: float, threshold: float, max_steps: int) -> "np.ndarray":
    """Integrate the cascade from the uniform initial activity x_init under the sustained input x_in and return the sampled activity profiles at t_j = j dt for j = 0..J as a (J + 1, N) array. J is the first sample at which the terminal node has moved by more than threshold from x_init (the run stops there); if that never happens, stop at j = max_steps. Integrate accurately (relative tolerance 1e-9 or tighter) so that the sampled profiles are reproducible to about 1e-8.

    Parameters
    ----------
    alpha : np.ndarray
        Edge timescale parameters alpha_i > 0.
    B : np.ndarray
        Edge saturation parameters B_i > 1.
    phi : np.ndarray
        Edge bias parameters phi_i in [-1, 1].
    x_in : float
        Constant upstream input in [-1, 1].
    x_init : float
        Uniform initial activity of every node, in [-1, 1].
    dt : float
        Sampling interval, positive.
    threshold : float
        Positive deviation of the terminal node from x_init that ends the run.
    max_steps : int
        Positive cap on the number of sampling intervals.

    Returns
    -------
    profiles : np.ndarray
        Array of shape (J + 1, N); row j is the profile at time j dt.

    Raises
    ------
    ValueError
        If the parameter arrays are invalid, x_in or x_init is outside [-1, 1], dt or threshold is not positive, or max_steps is not a positive integer.
    """
    return profiles

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


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


def _oracle_simulate_cascade(alpha: "np.ndarray", B: "np.ndarray", phi: "np.ndarray", x_in: float,
                             x_init: float, dt: float, threshold: float, max_steps: int) -> "np.ndarray":
    """Integrate eq. (1) from the uniform state x_i(0) = x_init under the sustained input x_in and
    return the profiles at t_j = j dt, j = 0..J, as a (J + 1, N) array. J is the first sample at
    which the terminal node has moved by more than `threshold` from x_init (the source halts its
    velocity tracking the moment the front reaches the terminal node, Sec. 2.4); if that never
    happens within max_steps samples the last row is at j = max_steps."""
    a, b, p = _check_params(alpha, B, phi)
    xin = _check_scalar(x_in, "x_in", -1.0, 1.0)
    x0 = _check_scalar(x_init, "x_init", -1.0, 1.0)
    h = _check_scalar(dt, "dt", 0.0, strict_lo=True)
    thr = _check_scalar(threshold, "threshold", 0.0, strict_lo=True)
    if isinstance(max_steps, bool) or int(max_steps) != max_steps or int(max_steps) < 1:
        raise ValueError("max_steps must be a positive integer")
    x = np.full(a.size, x0, dtype=np.float64)
    rows = [x.copy()]
    for j in range(int(max_steps)):
        sol = solve_ivp(lambda t, y: _oracle_cascade_rate(y, xin, a, b, p), (j * h, (j + 1) * h), x,
                        method="RK45", rtol=1e-9, atol=1e-11)
        x = sol.y[:, -1]
        rows.append(x.copy())
        if abs(x[-1] - x0) > thr:
            break
    return np.array(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.full(N, 1.5)\nB = np.full(N, 3.0)\nphi = np.full(N, 0.1)\n",
            "call": "np.asarray(simulate_cascade(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 100000))",
            "gold_call": "np.asarray(_oracle_simulate_cascade(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 100000))",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nN = 40\nalpha = 1.0 + 2.0 * np.arange(N) / (N - 1)\nB = np.where(np.arange(N) < 20, 2.5, 6.0)\nphi = np.full(N, -0.05)\n",
            "call": "np.asarray(simulate_cascade(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 100000))",
            "gold_call": "np.asarray(_oracle_simulate_cascade(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 100000))",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.full(N, 1.5)\nB = np.full(N, 3.0)\nphi = np.full(N, -0.2)\n",
            "call": "np.asarray(simulate_cascade(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 100000))",
            "gold_call": "np.asarray(_oracle_simulate_cascade(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 100000))",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.full(N, 1.5)\nB = np.full(N, 3.0)\nphi = np.full(N, 0.1)\ndef run_model():\n    try:\n        simulate_cascade(alpha, B, phi, 1.0, -1.0, 0.0, 1e-4, 100000)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_simulate_cascade(alpha, B, phi, 1.0, -1.0, 0.0, 1e-4, 100000)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
