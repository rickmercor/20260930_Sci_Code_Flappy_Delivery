"""
Eccentric anomaly of the 1PN quasi-Keplerian orbit

The hybrid spin evolution is driven by the instantaneous orbital separation rather than its orbit average, so the pipeline must track the orbital phase itself. In the 1PN quasi-Keplerian parametrization the radial motion is $r = a_r(1 - e_r\cos u)$ with the eccentric anomaly $u$ obeying the Kepler equation $n\,(t - t_0) = u - e_t \sin u$, where $n$ is the 1PN mean motion and $e_t$ the *time* eccentricity -- distinct from the radial eccentricity $e_r$ and from the angular eccentricity $e_\theta$ used elsewhere in the pipeline. The solution $u(t)$ is continuous and strictly increasing; over the integration times of interest it advances through hundreds of orbits, and the returned anomaly must accumulate accordingly rather than being reduced to a single orbit.

Returns
-------
np.ndarray, same shape as t_grid, the continuous eccentric anomaly u(t) in radians
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def kepler_eccentric_anomaly(t_grid: np.ndarray, n: float, e_t: float,
                             t0: float = 0.0) -> np.ndarray:
    '''Continuous eccentric anomaly u(t) from the quasi-Keplerian
    Kepler equation n (t - t0) = u - e_t sin u.

    Parameters
    ----------
    t_grid : np.ndarray
        1-D array of reduced times, entries finite.
    n : float
        Mean motion in reduced units, must be > 0.
    e_t : float
        Time eccentricity, must satisfy 0 <= e_t < 1.
    t0 : float
        Epoch of periapsis passage, must be finite.

    Returns
    -------
    u : np.ndarray
        1-D array, same shape as t_grid, the continuous (unreduced)
        eccentric anomaly in radians. For the computed mean anomaly
        ell = n * (t - t0), the absolute error in u must be at most
        1e-12 + 8 * eps * max(1, abs(ell)) per entry, where eps is
        the float64 machine epsilon. This allows for the resolution
        of large unreduced angles. Any sufficiently accurate solver
        is acceptable.

    Raises
    ------
    ValueError
        If t_grid is not a nonempty 1-D array of finite times; if n is
        not a finite scalar > 0; if e_t is not a finite scalar in
        [0, 1); if t0 is not a finite scalar; or if computing
        ell = n * (t - t0) in float64 produces a nonfinite value.
    RuntimeError
        If the numerical solver cannot establish convergence.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_kepler_eccentric_anomaly(t_grid: np.ndarray, n: float, e_t: float,
                                     t0: float = 0.0) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("t_grid must be a nonempty 1-D array of finite times")
    for name, v in (("n", n), ("e_t", e_t), ("t0", t0)):
        if not (np.isscalar(v) and np.isfinite(float(v))):
            raise ValueError(f"{name} must be a finite scalar")
    n, e_t, t0 = float(n), float(e_t), float(t0)
    if n <= 0.0:
        raise ValueError("n must be > 0")
    if not (0.0 <= e_t < 1.0):
        raise ValueError("e_t must lie in [0, 1)")
    with np.errstate(over="ignore", invalid="ignore"):
        ell = n * (t - t0)
    if not np.all(np.isfinite(ell)):
        raise ValueError("computed mean anomaly must be finite in float64")
    if e_t == 0.0:
        return ell.copy()

    # Use the trigonometric functions' argument reduction. Reducing by
    # the rounded float64 value of 2*pi would spuriously treat its multiples
    # as exact periapsis passages and can amplify the phase error when e_t is
    # close to one. atan2(sin(ell), cos(ell)) also preserves tiny anomalies.
    reduced = np.arctan2(np.sin(ell), np.cos(ell))
    mean = np.abs(reduced)

    def residual(z):
        # Stable evaluation near periapsis, including e_t close to one.
        z2 = z * z
        z_minus_sin = np.where(
            z < 0.25,
            z * z2 * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 *
                (1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 *
                (1.0 / 39916800.0 - z2 / 6227020800.0))))),
            z - np.sin(z))
        return (1.0 - e_t) * z + e_t * z_minus_sin - mean

    # The positive reduced root lies in [mean, min(mean + e_t, pi)].
    lo = mean.copy()
    hi = np.minimum(mean + e_t, np.pi)
    hi = np.where(mean == 0.0, 0.0, hi)
    eps = np.finfo(float).eps
    for _ in range(80):
        mid = lo + 0.5 * (hi - lo)
        f_mid = residual(mid)
        lo = np.where(f_mid <= 0.0, mid, lo)
        hi = np.where(f_mid >= 0.0, mid, hi)
        if np.all(hi - lo <= 8.0 * eps * np.maximum(1.0, mid)):
            break
    else:
        raise RuntimeError("elliptic Kepler solver did not converge")
    root = lo + 0.5 * (hi - lo)
    if np.any(np.abs(residual(root)) > 32.0 * eps * np.maximum(1.0, mean)):
        raise RuntimeError("elliptic Kepler residual check failed")

    u = ell + (np.copysign(root, reduced) - reduced)
    error_limit = 1e-12 + 8.0 * eps * np.maximum(1.0, np.abs(ell))
    if (not np.all(np.isfinite(u)) or
            np.any(np.abs((u - ell) - e_t * np.sin(u)) > 2.0 * error_limit)):
        raise RuntimeError("unreduced Kepler residual check failed")
    return u

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: hundreds of orbits at the pipeline eccentricity ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 5.0e6, 512)\n"
                      "n, e_t = 6.80568578e-4, 0.61553846\n"),
            "call": "kepler_eccentric_anomaly(t, n, e_t)",
            "gold_call": "_oracle_kepler_eccentric_anomaly(t, n, e_t)",
        },
        # --- Normal: high eccentricity, nonzero epoch ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(-2.0e3, 8.0e3, 257)\n"
                      "n, e_t, t0 = 5.204e-3, 0.95, 750.0\n"),
            "call": "kepler_eccentric_anomaly(t, n, e_t, t0)",
            "gold_call": "_oracle_kepler_eccentric_anomaly(t, n, e_t, t0)",
        },
        # --- Boundary: circular orbit reduces to u = n t ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 1.0e4, 64)\n"
                      "n, e_t = 1.0e-3, 0.0\n"),
            "call": "kepler_eccentric_anomaly(t, n, e_t)",
            "gold_call": "_oracle_kepler_eccentric_anomaly(t, n, e_t)",
        },
        # --- Boundary: single sample exactly at periapsis passage ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.array([1250.0])\n"
                      "n, e_t, t0 = 2.0e-3, 0.6, 1250.0\n"),
            "call": "kepler_eccentric_anomaly(t, n, e_t, t0)",
            "gold_call": "_oracle_kepler_eccentric_anomaly(t, n, e_t, t0)",
        },
        # --- Edge: parabolic eccentricity must raise ValueError ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 10.0, 5)
def run_model():
    try:
        kepler_eccentric_anomaly(t, 1.0e-3, 1.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_kepler_eccentric_anomaly(t, 1.0e-3, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: nonpositive mean motion must raise ValueError ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 10.0, 5)
def run_model():
    try:
        kepler_eccentric_anomaly(t, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_kepler_eccentric_anomaly(t, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Regression: allowed e_t=0.99 with explicit absolute accuracy ---
        {
            "setup": """import numpy as np
t = np.array([0.2554114827368501])
expected = np.array([1.1650195950839073])
def check_accuracy(solver):
    got = np.asarray(solver(t, 1.0, 0.99, 0.0))
    limit = 1e-12 + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(t))
    return int(got.shape == t.shape and np.all(np.isfinite(got))
               and np.all(np.abs(got - expected) <= limit))
""",
            "call": "check_accuracy(kepler_eccentric_anomaly)",
            "gold_call": "check_accuracy(_oracle_kepler_eccentric_anomaly)",
        },
        # --- Regression: both signs and continuous orbit count at e_t=0.99 ---
        {
            "setup": """import numpy as np
turns = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
small_mean = 0.2554114827368501
small_root = 1.1650195950839073
t = np.concatenate((2.0 * np.pi * turns + small_mean,
                    2.0 * np.pi * turns - small_mean))
expected = np.concatenate((2.0 * np.pi * turns + small_root,
                           2.0 * np.pi * turns - small_root))
def check_accuracy(solver):
    got = np.asarray(solver(t, 1.0, 0.99, 0.0))
    limit = 1e-12 + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(t))
    return int(got.shape == t.shape and np.all(np.isfinite(got))
               and np.all(np.abs(got - expected) <= limit))
""",
            "call": "check_accuracy(kepler_eccentric_anomaly)",
            "gold_call": "check_accuracy(_oracle_kepler_eccentric_anomaly)",
        },
        # --- Regression: near-parabolic multi-orbit periapsis endpoint ---
        {
            "setup": """import numpy as np
t = np.array([100.0 * (2.0 * np.pi)])
expected = np.array([628.318531110835])
def check_accuracy(solver):
    got = np.asarray(solver(t, 1.0, 0.99999999, 0.0))
    limit = 1e-12 + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(t))
    return int(got.shape == t.shape and np.all(np.isfinite(got))
               and np.all(np.abs(got - expected) <= limit))
""",
            "call": "check_accuracy(kepler_eccentric_anomaly)",
            "gold_call": "check_accuracy(_oracle_kepler_eccentric_anomaly)",
        },
        # --- Edge: nonfinite computed mean anomaly ---
        {
            "setup": """import numpy as np
t = np.array([np.finfo(float).max])
def check_rejection(solver):
    try:
        solver(t, 2.0, 0.5)
        return 0
    except ValueError:
        return 1
""",
            "call": "check_rejection(kepler_eccentric_anomaly)",
            "gold_call": "check_rejection(_oracle_kepler_eccentric_anomaly)",
        },
    ]
