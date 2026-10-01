"""
Implement locate_stationary_points, which finds the two minima and the barrier top of the lower
adiabatic surface of the two-state diabatic model, together with the harmonic frequencies of the
two wells.

The two wells of an asymmetric double well play different roles in tunnelling. The energies of
their minima set the bias between the two localized configurations, their curvatures set the
harmonic vibrational frequencies and zero-point energies, and the barrier between them separates
the configurations. The wells are labelled by the potential energy of their minimum, not by their
position, so the labels do not change when the surface is reflected.

Returns
-------
np.ndarray of 8 floats [x_low, x_top, x_high, V_low, V_top, V_high, omega_low, omega_high]: the deeper minimum, barrier top and higher minimum with their energies, and the harmonic frequencies of the two wells
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_stationary_points(params: "np.ndarray") -> "np.ndarray":
    '''Minima, barrier top and well frequencies of the lower adiabatic surface.

    Parameters
    ----------
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.

    Returns
    -------
    points : np.ndarray
        Array of 8 floats [x_low, x_top, x_high, V_low, V_top, V_high, omega_low,
        omega_high]. x_low and x_high are the positions of the lower (deeper) and the higher
        minimum, and x_top is the position of the barrier maximum between them, each accurate
        to 1e-10. V_low, V_top and V_high are the potential energies at these three points,
        and omega_low and omega_high are the harmonic frequencies sqrt(V''/m) of the two wells
        at their minima (m = 1). When the surface is mirror-symmetric (omega_l == omega_r and
        eps == 0), the two minima are degenerate, and the minimum at negative x is labelled
        low.

    Raises
    ------
    ValueError
        If params is invalid (as in two_diabat_potential) or the surface does not have
        exactly two minima.
    '''
    return points

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _is_mirror_symmetric(params: "np.ndarray") -> bool:
    """True when the surface is symmetric under x -> -x (degenerate minima)."""
    omega_l, omega_r, _, eps, _ = _diabat_params(params)
    return omega_l == omega_r and eps == 0.0


def _oracle_locate_stationary_points(params: "np.ndarray") -> "np.ndarray":
    x0 = _diabat_params(params)[2]
    # every stationary point of the lower adiabat lies between the two diabatic minima
    grid = np.linspace(-abs(x0), abs(x0), 20001)
    slope = _oracle_two_diabat_potential(grid, params)[1]

    def _force(z):
        return float(_oracle_two_diabat_potential(z, params)[1, 0])

    def _roots(brackets):
        return [brentq(_force, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15, maxiter=200)
                for i in np.flatnonzero(brackets)]

    minima = _roots((slope[:-1] < 0.0) & (slope[1:] >= 0.0))
    maxima = _roots((slope[:-1] > 0.0) & (slope[1:] <= 0.0))
    if len(minima) != 2:
        raise ValueError("the surface does not have exactly two minima")
    x_top = [x for x in maxima if minima[0] < x < minima[1]][0]
    v_minima = _oracle_two_diabat_potential(np.array(minima), params)[0]
    if _is_mirror_symmetric(params) or v_minima[0] <= v_minima[1]:
        x_low, x_high = minima
    else:
        x_high, x_low = minima
    v, _, d2v = _oracle_two_diabat_potential(np.array([x_low, x_top, x_high]), params)
    return np.array([x_low, x_top, x_high, v[0], v[1], v[2], np.sqrt(d2v[0]), np.sqrt(d2v[2])])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: the asymmetric model; the narrow left well is the deeper one ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
""",
            "call": "locate_stationary_points(params.copy())",
            "gold_call": "_oracle_locate_stationary_points(params.copy())",
            "tol": 1e-9,
        },
        # --- Typical: the mirror image (x0 < 0), so the deeper well lies at positive x ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, -4.6, -0.19, 1.5])
""",
            "call": "locate_stationary_points(params.copy())",
            "gold_call": "_oracle_locate_stationary_points(params.copy())",
            "tol": 1e-9,
        },
        # --- Typical: larger displacement and coupling ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 5.0, -0.46, 2.0])
""",
            "call": "locate_stationary_points(params.copy())",
            "gold_call": "_oracle_locate_stationary_points(params.copy())",
            "tol": 1e-9,
        },
        # --- Edge: the narrow diabat raised so far that the wide right well becomes the deeper one ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -1.5, 1.5])
""",
            "call": "locate_stationary_points(params.copy())",
            "gold_call": "_oracle_locate_stationary_points(params.copy())",
            "tol": 1e-9,
        },
        # --- Boundary: mirror-symmetric surface with degenerate minima (negative-x minimum is low) ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 1.0, 3.0, 0.0, 1.0])
""",
            "call": "locate_stationary_points(params.copy())",
            "gold_call": "_oracle_locate_stationary_points(params.copy())",
            "tol": 1e-9,
        },
        # --- Invalid: coupling so strong that the two wells merge into one ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 30.0])
def run_model():
    try:
        locate_stationary_points(params.copy())
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_locate_stationary_points(params.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
