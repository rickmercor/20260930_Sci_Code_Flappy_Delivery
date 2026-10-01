"""
Orchestrate the whole pipeline for a heterogeneous pathway: check that every edge is bistable and that the uniform initial activity is a resting state, simulate the cascade, obtain the intrinsic edge speeds (the intrinsic speed of an edge with timescale alpha_i is alpha_i times the intrinsic speed of the unit-timescale edge with the same B_i and phi_i), build the rescaled coordinate, estimate the instantaneous speed on every sampling interval in both the node-index coordinate and the rescaled coordinate, express both as fractions of the respective pathway length per unit time, locate the wave centre at every sample in the rescaled coordinate, and return the ratio of the velocity integral square error in the rescaled coordinate to that in the node-index coordinate over the post-transient window. Call the earlier step functions rather than reimplementing them.

The ratio of velocity fluctuation in the rescaled and the original description measures how much of the apparent stuttering of a signal in a heterogeneous cascade is an artefact of the node coordinate rather than a property of the signal itself.

Returns
-------
float, VISE(rescaled) / VISE(node index) over the post-transient window.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fluctuation_suppression_ratio(alpha: "np.ndarray", B: "np.ndarray", phi: "np.ndarray",
                                          x_in: float, x_init: float, dt: float, threshold: float,
                                          speed_rel_tol: float, s_lo: float, s_hi: float) -> float:
    """Orchestrate the whole pipeline for a heterogeneous pathway: check that every edge is bistable and that the uniform initial activity is a resting state, simulate the cascade, obtain the intrinsic edge speeds (the intrinsic speed of an edge with timescale alpha_i is alpha_i times the intrinsic speed of the unit-timescale edge with the same B_i and phi_i), build the rescaled coordinate, estimate the instantaneous speed on every sampling interval in both the node-index coordinate and the rescaled coordinate, express both as fractions of the respective pathway length per unit time, locate the wave centre at every sample in the rescaled coordinate, and return the ratio of the velocity integral square error in the rescaled coordinate to that in the node-index coordinate over the post-transient window. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    alpha : np.ndarray
        Edge timescale parameters alpha_i > 0.
    B : np.ndarray
        Edge saturation parameters B_i > 1.
    phi : np.ndarray
        Edge bias parameters, each with |phi_i| < 1 / B_i.
    x_in : float
        Constant upstream input in [-1, 1].
    x_init : float
        Uniform initial activity in [-1, 1].
    dt : float
        Sampling interval, positive.
    threshold : float
        Positive terminal-node deviation that ends every run.
    speed_rel_tol : float
        Required relative accuracy of every intrinsic edge speed, in (0, 1e-6].
    s_lo : float
        Lower window bound in [0, 1].
    s_hi : float
        Upper window bound in [0, 1], greater than s_lo.

    Returns
    -------
    ratio : float
        VISE(rescaled) / VISE(node index).

    Raises
    ------
    ValueError
        If the parameters are invalid, an edge is not bistable, the uniform initial activity is not a resting state of the pathway, an intrinsic speed cannot be certified to speed_rel_tol, or the window contains fewer than two intervals.
    """
    return ratio

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


def _oracle_fluctuation_suppression_ratio(alpha: "np.ndarray", B: "np.ndarray", phi: "np.ndarray",
                                          x_in: float, x_init: float, dt: float, threshold: float,
                                          speed_rel_tol: float, s_lo: float, s_hi: float) -> float:
    """End-to-end: bistability guard (step 01), resting-state guard (02), heterogeneous simulation (03), intrinsic edge speeds
    (05: the asymptotic travelling-wave speed to speed_rel_tol, using c(alpha,B,phi) = alpha c(1,B,phi) since alpha
    only rescales time), rescaled
    coordinate (06), least-squares speeds in both coordinates (04), wave centres (07) and the two
    VISE values (08). Speeds are compared as fractions of the respective pathway length per unit
    time (c_j / N against c~_j / 1). Returns VISE(normalised) / VISE(original)."""
    a, b, p = _check_params(alpha, B, phi)
    for pi, bi in zip(p, b):
        xi_phic = _oracle_uniform_steady_states(pi, bi)
        if abs(pi) >= xi_phic[1]:
            raise ValueError("every edge must lie in the bistable regime |phi| < 1/B")
    h = _check_scalar(dt, "dt", 0.0, strict_lo=True)
    n = a.size
    # the pathway must start at rest: the uniform state x_init driven by an input equal to x_init is stationary
    rest = _oracle_cascade_rate(np.full(n, float(x_init)), float(x_init), a, b, p)
    if np.max(np.abs(rest)) > 1e-9:
        raise ValueError("x_init is not a uniform steady state of the pathway")
    profiles = _oracle_simulate_cascade(a, b, p, x_in, x_init, h, threshold, 100000)
    n_int = profiles.shape[0] - 1
    times = h * np.arange(n_int)
    base = {}
    for bi, pi in set(zip(b.tolist(), p.tolist())):
        base[(bi, pi)] = _oracle_intrinsic_edge_speed(1.0, bi, pi, x_in, x_init, speed_rel_tol)
    c_edge = np.array([a[i] * base[(b[i], p[i])] for i in range(n)])
    s = _oracle_rescaled_positions(c_edge)
    pos = np.arange(1, n + 1, dtype=np.float64)
    xin = float(x_in)
    c_orig = np.array([_oracle_instantaneous_speed(profiles[j], profiles[j + 1], pos, xin, h)
                       for j in range(n_int)]) / n
    c_norm = np.array([_oracle_instantaneous_speed(profiles[j], profiles[j + 1], s, xin, h)
                       for j in range(n_int)]) / s[-1]
    centres = np.array([_oracle_wave_centre(s, profiles[j], xin) for j in range(n_int)]) / s[-1]
    v_orig = _oracle_velocity_ise(c_orig, times, centres, s_lo, s_hi)
    v_norm = _oracle_velocity_ise(c_norm, times, centres, s_lo, s_hi)
    return v_norm / v_orig

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nN = 120\nalpha = 1.0 + 4.0 * np.arange(N) / (N - 1)\nB = np.where(np.arange(N) < N // 2, 3.0, 8.0)\nphi = np.zeros(N)\n",
            "call": "fluctuation_suppression_ratio(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 1e-9, 0.15, 0.85)",
            "gold_call": "_oracle_fluctuation_suppression_ratio(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 1e-9, 0.15, 0.85)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nN = 40\nalpha = 1.0 + 2.0 * np.arange(N) / (N - 1)\nB = np.where(np.arange(N) < 20, 2.5, 6.0)\nphi = np.full(N, -0.05)\n",
            "call": "fluctuation_suppression_ratio(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 1e-9, 0.15, 0.85)",
            "gold_call": "_oracle_fluctuation_suppression_ratio(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 1e-9, 0.15, 0.85)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.linspace(2.0, 1.0, N)\nB = np.full(N, 3.0)\nphi = np.full(N, -0.2)\n",
            "call": "fluctuation_suppression_ratio(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 1e-9, 0.15, 0.85)",
            "gold_call": "_oracle_fluctuation_suppression_ratio(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 1e-9, 0.15, 0.85)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nN = 30\nalpha = np.full(N, 1.5)\nB = np.full(N, 3.0)\nphi = np.full(N, 0.5)\ndef run_model():\n    try:\n        fluctuation_suppression_ratio(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 1e-9, 0.15, 0.85)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_fluctuation_suppression_ratio(alpha, B, phi, 1.0, -1.0, 1.0, 1e-4, 1e-9, 0.15, 0.85)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
