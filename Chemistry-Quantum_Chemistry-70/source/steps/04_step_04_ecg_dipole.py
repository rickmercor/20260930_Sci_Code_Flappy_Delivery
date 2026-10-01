"""
Step 04: Electric-dipole coupling matrix between the two parities. Electric-dipole coupling matrix between even-parity and odd-parity antisymmetrized L = 1 correlated Gaussians, in the length or the velocity form.

A photon emitted in an electric-dipole transition between two L = 1 states of opposite parity couples through the total electron position r1_vec + r2_vec in the length form, or through the total momentum in the velocity form. For exact eigenstates the two are related by <f| r |i> = <f| nabla |i> / (E_i - E_f) in hartree atomic units, so the agreement of the rates they give is a standard measure of how well a variational basis describes both states.

The even and odd functions are the same unnormalized correlated Gaussians used for the overlap, and the component convention that fixes the returned matrix is given with the parameters below.

Returns
-------
numpy.ndarray of shape (n, m): length- or velocity-form dipole coupling between even and odd functions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ecg_dipole(exps_even: "np.ndarray", exps_odd: "np.ndarray", gauge: str) -> "np.ndarray":
    '''Dipole coupling matrix between even (1^+) and odd (1^-) antisymmetrized correlated Gaussians.

    Parameters
    ----------
    exps_even : np.ndarray
        Array of shape (n, 3), n >= 1, of exponents (a, b, c) in bohr^-2 for the even functions; finite,
        non-negative, a b + c (a + b) > 0.
    exps_odd : np.ndarray
        Array of shape (m, 3), m >= 1, with the same rules, for the odd functions.
    gauge : str
        "length" for D_ij = integral of chi_i (y1 + y2) phi_j, or "velocity" for
        D_ij = integral of chi_i (d/dy1 + d/dy2) phi_j, where chi_i = (1 - P12)[(r1_vec x r2_vec)_x g_i],
        phi_j = (1 - P12)[z1 g_j], g = exp(-a r1^2 - b r2^2 - c r12^2) and P12 exchanges the electrons.

    Returns
    -------
    result : np.ndarray
        Real array of shape (n, m) holding D_ij for the unnormalized functions. For normalized states the length form
        has units of bohr and the velocity form units of inverse bohr.

    Raises
    ------
    ValueError
        If an exponent array is malformed or non-integrable, or if gauge is not "length" or "velocity".
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _one_dipole(k, swap, gauge):
    _, w = _odd_vectors(swap)
    b00, b01, b11 = k["B"]
    if gauge == "length":
        moment = -(w[0] - w[1]) * k["dS"]
    else:
        moment = k["dS"] * (-b00 * w[1] + b01 * w[0] - b01 * w[1] + b11 * w[0])
    return k["N"] * moment


def _oracle_ecg_dipole(exps_even: "np.ndarray", exps_odd: "np.ndarray", gauge: str) -> "np.ndarray":
    if gauge not in ("length", "velocity"):
        raise ValueError("gauge must be 'length' or 'velocity'")
    ee, eo = _ecg_exponents(exps_even), _ecg_exponents(exps_odd)
    direct = _one_dipole(_ecg_blocks(ee, eo, "odd", False), False, gauge)
    exchanged = _one_dipole(_ecg_blocks(ee, eo, "odd", True), True, gauge)
    return np.asarray(2.0 * (direct - exchanged), dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
ee = np.array([[0.8, 0.3, 0.0], [0.5, 0.5, 0.2], [1.6, 0.1, 0.05]])
eo = np.array([[0.3, 0.9, 0.1], [2.0, 0.4, 0.0], [0.7, 0.7, 0.6], [0.2, 1.1, 0.3]])
"""
    return [
        # --- Normal: length form ---
        {"setup": base, "call": "ecg_dipole(ee.copy(), eo.copy(), 'length')",
         "gold_call": "_oracle_ecg_dipole(ee.copy(), eo.copy(), 'length')", "tol": 1e-9},
        # --- Normal: velocity form ---
        {"setup": base, "call": "ecg_dipole(ee.copy(), eo.copy(), 'velocity')",
         "gold_call": "_oracle_ecg_dipole(ee.copy(), eo.copy(), 'velocity')", "tol": 1e-9},
        # --- Boundary: uncorrelated odd functions, one with equal orbital exponents on both electrons ---
        {"setup": "import numpy as np\nee = np.array([[0.6, 0.2, 0.0], [0.3, 0.3, 0.3]])\neo = np.array([[0.9, 0.9, 0.0], [0.1, 2.0, 0.0]])\n",
         "call": "ecg_dipole(ee.copy(), eo.copy(), 'length')", "gold_call": "_oracle_ecg_dipole(ee.copy(), eo.copy(), 'length')", "tol": 1e-9},
        # --- Edge: compact core-like odd function against a diffuse correlated even function, velocity form ---
        {"setup": "import numpy as np\nee = np.array([[0.02, 0.05, 0.5]])\neo = np.array([[40.0, 0.01, 0.0], [0.0, 0.5, 0.7]])\n",
         "call": "ecg_dipole(ee.copy(), eo.copy(), 'velocity')", "gold_call": "_oracle_ecg_dipole(ee.copy(), eo.copy(), 'velocity')", "tol": 1e-9},
        # --- Error: unknown gauge ---
        {"setup": "import numpy as np\ne = np.array([[1.0, 0.5, 0.1]])\n"
                  "def _probe(fn):\n    try:\n        fn(e, e, 'acceleration')\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(ecg_dipole)", "gold_call": "_probe(_oracle_ecg_dipole)"},
    ]
