"""
Implement two_diabat_potential, which evaluates the lower adiabatic potential energy surface of a
two-state diabatic model, together with its first and second derivatives, at a set of positions.

Two displaced harmonic diabatic states of different curvature, mixed by a constant electronic
coupling, produce a lower adiabatic surface with two wells that generally differ in depth and in
width. Such surfaces are standard one-dimensional models of asymmetric proton-transfer and
conformational double wells, in which a light particle tunnels between two inequivalent
configurations. Positions, energies and frequencies are in reduced units in which the particle
mass is one.

Returns
-------
np.ndarray of shape (3, n): V(x), dV/dx and d2V/dx2 of the lower adiabatic surface at each of the n positions (reduced units)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_diabat_potential(x: "np.ndarray", params: "np.ndarray") -> "np.ndarray":
    '''Lower adiabatic potential of a two-state diabatic model and its x-derivatives.

    Parameters
    ----------
    x : np.ndarray
        Positions, a scalar or a 1-D array of n finite values.
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01], five finite values. The
        diabatic potentials are V00(x) = omega_l**2 (x + x0)**2 / 2 - eps and
        V11(x) = omega_r**2 (x - x0)**2 / 2, they are coupled by the constant V01, and the
        lower adiabatic surface is
        V(x) = (V00 + V11) / 2 - sqrt((V00 - V11)**2 + 4 V01**2) / 2.
        omega_l, omega_r and V01 must be positive and x0 must be nonzero.

    Returns
    -------
    values : np.ndarray
        Array of shape (3, n): row 0 holds V(x), row 1 dV/dx and row 2 d2V/dx2, one column
        per position in the order of x (n = 1 for a scalar x).

    Raises
    ------
    ValueError
        If params does not hold exactly five finite values, omega_l, omega_r or V01 is not
        positive, x0 is zero, or x contains a non-finite value.
    '''
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _diabat_params(params: "np.ndarray") -> tuple:
    """Validated model parameters (omega_l, omega_r, x0, eps, V01) as floats."""
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 5 or not np.all(np.isfinite(p)):
        raise ValueError("params must hold five finite values [omega_l, omega_r, x0, eps, V01]")
    omega_l, omega_r, x0, eps, v01 = (float(v) for v in p)
    if omega_l <= 0.0 or omega_r <= 0.0 or v01 <= 0.0:
        raise ValueError("omega_l, omega_r and V01 must be positive")
    if x0 == 0.0:
        raise ValueError("x0 must be nonzero")
    return omega_l, omega_r, x0, eps, v01


def _oracle_two_diabat_potential(x: "np.ndarray", params: "np.ndarray") -> "np.ndarray":
    omega_l, omega_r, x0, eps, v01 = _diabat_params(params)
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("positions must be finite")
    k_l, k_r = omega_l * omega_l, omega_r * omega_r
    v00 = 0.5 * k_l * (x + x0) ** 2 - eps
    v11 = 0.5 * k_r * (x - x0) ** 2
    gap = v00 - v11
    gap_slope = k_l * (x + x0) - k_r * (x - x0)
    root = np.sqrt(gap * gap + 4.0 * v01 * v01)
    v = 0.5 * (v00 + v11) - 0.5 * root
    dv = 0.5 * (k_l * (x + x0) + k_r * (x - x0)) - 0.5 * gap * gap_slope / root
    d2v = (0.5 * (k_l + k_r) - 0.5 * gap * (k_l - k_r) / root
           - 2.0 * v01 * v01 * gap_slope * gap_slope / root ** 3)
    return np.vstack([v, dv, d2v])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid = """
def run_model():
    try:
        two_diabat_potential(x.copy(), params.copy())
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_two_diabat_potential(x.copy(), params.copy())
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: the asymmetric model on a grid through both wells, the barrier and the
        #     outer walls ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
x = np.linspace(-7.0, 7.0, 29)
""",
            "call": "two_diabat_potential(x.copy(), params.copy())",
            "gold_call": "_oracle_two_diabat_potential(x.copy(), params.copy())",
            "tol": 1e-10,
        },
        # --- Boundary: a scalar position must still give a (3, 1) array ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 5.0, -0.46, 2.0])
x = 0.37
""",
            "call": "two_diabat_potential(x, params.copy())",
            "gold_call": "_oracle_two_diabat_potential(x, params.copy())",
            "tol": 1e-10,
        },
        # --- Typical: diabats displaced the other way (x0 < 0), so the narrow well is on the right ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, -4.6, -0.19, 1.5])
x = np.array([-5.2, -4.47, -1.65, 0.0, 1.65, 2.3, 4.49, 6.0])
""",
            "call": "two_diabat_potential(x.copy(), params.copy())",
            "gold_call": "_oracle_two_diabat_potential(x.copy(), params.copy())",
            "tol": 1e-10,
        },
        # --- Boundary: equal diabatic frequencies and no offset (mirror-symmetric surface),
        #     including the barrier top at x = 0 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 1.0, 3.0, 0.0, 1.0])
x = np.array([-3.5, -2.98, -1.0, -0.2, 0.0, 0.2, 1.0, 2.98, 3.5])
""",
            "call": "two_diabat_potential(x.copy(), params.copy())",
            "gold_call": "_oracle_two_diabat_potential(x.copy(), params.copy())",
            "tol": 1e-10,
        },
        # --- Edge: weak coupling, positions straddling the sharply avoided diabatic crossing ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, 0.4, 0.05])
x = np.linspace(-2.2, -1.2, 21)
""",
            "call": "two_diabat_potential(x.copy(), params.copy())",
            "gold_call": "_oracle_two_diabat_potential(x.copy(), params.copy())",
            "tol": 1e-10,
        },
        # --- Invalid: zero diabatic coupling ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 0.0])
x = np.array([0.0, 1.0])
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: four parameters instead of five ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19])
x = np.array([0.0, 1.0])
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
