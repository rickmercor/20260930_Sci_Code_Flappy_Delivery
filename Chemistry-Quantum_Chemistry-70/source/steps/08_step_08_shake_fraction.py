"""
Step 08: Shake fraction of the radiative width (orchestrator). Orchestrator: fraction of the total electric-dipole radiative width of the lowest even-parity L = 1 triplet state that does not end on the lowest odd-parity L = 1 triplet level.

A doubly excited 2p^2 3Pe level of a helium-like ion lies above the first ionization threshold but has the wrong parity to autoionize into any 1s el continuum of total L = 1, so in the nonrelativistic limit it lives only as long as photon emission allows. Its strongest line is the one-electron jump 2p -> 1s that leaves the other electron in 2p, ending on 1s2p 3Po. Electron correlation and the sudden change of screening seen by the spectator electron send part of the width to the 1snp Rydberg levels with n >= 3 and part of it into the 1s ep continuum above the threshold, where the atom is left ionized rather than excited. The fraction of the total width outside the main line measures that shake and correlation contribution.

The two parts of the width are different objects. The lines are a discrete sum over bound final states, while the continuum part is an integral of a distribution over final-state energy that only a calculation able to represent continuum final states can supply. Adding them gives the total width, and the fraction is converged only when the bound basis supports the Rydberg series, the rotated basis supports the continuum, and the result no longer moves with the rotation angle.

Returns
-------
float, fraction 1 - A_1 / A_tot of the E1 decay width that does not end on the lowest odd-parity level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def shake_fraction(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, theta: float, cut: float,
                   n_grid: int) -> float:
    '''Fraction of the total E1 decay rate of the lowest 1^+ triplet state not carried by its strongest line.

    Parameters
    ----------
    exps_even : np.ndarray
        Array of shape (n, 3) of exponents (a, b, c) in bohr^-2 for the even basis (1 - P12)[(r1_vec x r2_vec) g];
        finite, non-negative, a b + c (a + b) > 0.
    exps_odd : np.ndarray
        Array of shape (m, 3), same rules, for the odd basis (1 - P12)[r1_vec g], g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge, Z > 0, of an infinitely heavy nucleus.
    theta : float
        Rotation angle in radians, 0 < theta < pi/4, used for the continuum part only.
    cut : float
        Relative threshold, 0 < cut < 1, on the overlap eigenvalues, applied separately to each basis.
    n_grid : int
        Number of points, n_grid >= 2, of the uniform grid of final-state energies spanning the closed interval from
        the one-electron threshold -Z^2/2 hartree to the lowest even-parity energy E_i, on which the continuum
        distribution is evaluated and integrated by the trapezoidal rule.

    Returns
    -------
    result : float
        1 - A_1 / A_tot, where A_1 is the rate of the line to the lowest bound odd-parity level and A_tot is the sum
        of the rates of every line to a bound odd-parity level and the integral over final-state energy of the
        continuum distribution.

    Raises
    ------
    ValueError
        If the inputs are invalid as for the line table and the continuum distribution, if n_grid is smaller than 2,
        or if no bound odd-parity level lies below the initial state.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_shake_fraction(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, theta: float, cut: float,
                           n_grid: int) -> float:
    points = int(n_grid)
    if points < 2:
        raise ValueError("n_grid must be at least 2")
    lines = _oracle_bound_line_rates(exps_even, exps_odd, Z, cut)
    if lines.shape[0] == 0:
        raise ValueError("no bound odd-parity level lies below the initial state")
    threshold = -0.5 * float(Z) ** 2
    e_init = lines[0, 0] + lines[0, 1]
    grid = np.linspace(threshold, e_init, points)
    spec = _oracle_dissociation_spectrum(exps_even, exps_odd, Z, theta, cut, grid)
    continuum = float(np.trapezoid(np.asarray(spec, dtype=float), grid))
    total = float(np.sum(lines[:, 2])) + continuum
    return float(1.0 - lines[0, 2] / total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
g = [0.08, 0.25, 0.8, 2.5]
ee = np.array([(g[i], g[j], c) for i in range(4) for j in range(i, 4) for c in (0.0, 0.15)])
inner, outer = [0.6, 2.4, 9.0, 36.0], [0.015, 0.05, 0.15, 0.5]
rows = [(a, b, 0.0) for a in inner for b in outer] + [(b, a, 0.0) for a in inner for b in outer]
rows += [(a, a, 0.2) for a in (0.1, 0.5)]
eo = np.array(rows)
"""
    return [
        # --- Normal: helium with a small basis at the paper's rotation angle ---
        {"setup": setup, "call": "shake_fraction(ee.copy(), eo.copy(), 2.0, 0.25, 1e-12, 200)",
         "gold_call": "_oracle_shake_fraction(ee.copy(), eo.copy(), 2.0, 0.25, 1e-12, 200)", "tol": 1e-7},
        # --- Normal: Li+ with the scaled basis and a different angle ---
        {"setup": setup, "call": "shake_fraction(ee * 2.25, eo * 2.25, 3.0, 0.18, 1e-12, 200)",
         "gold_call": "_oracle_shake_fraction(ee * 2.25, eo * 2.25, 3.0, 0.18, 1e-12, 200)", "tol": 1e-7},
        # --- Normal: stationarity against the rotation angle, the property that makes the rotated route meaningful ---
        {"setup": setup + "def spread(fn):\n    lo = fn(ee.copy(), eo.copy(), 2.0, 0.18, 1e-12, 200)\n"
                          "    hi = fn(ee.copy(), eo.copy(), 2.0, 0.38, 1e-12, 200)\n    return float(abs(lo - hi))\n",
         "call": "spread(shake_fraction)", "gold_call": "spread(_oracle_shake_fraction)", "tol": 1e-9},
        # --- Boundary: the coarsest grid the contract allows, where the quadrature is a single trapezoid ---
        {"setup": setup, "call": "shake_fraction(ee.copy(), eo.copy(), 2.0, 0.25, 1e-12, 2)",
         "gold_call": "_oracle_shake_fraction(ee.copy(), eo.copy(), 2.0, 0.25, 1e-12, 2)", "tol": 1e-7},
        # --- Boundary: an odd basis without diffuse functions, so the Rydberg series is barely represented ---
        {"setup": setup + "eo_c = eo[(eo[:, 0] > 0.06) & (eo[:, 1] > 0.06)]\n",
         "call": "shake_fraction(ee.copy(), eo_c.copy(), 2.0, 0.25, 1e-12, 200)",
         "gold_call": "_oracle_shake_fraction(ee.copy(), eo_c.copy(), 2.0, 0.25, 1e-12, 200)", "tol": 1e-7},
        # --- Error: no bound odd-parity level below the initial state ---
        {"setup": "import numpy as np\nee = np.array([[0.02, 0.02, 0.0]])\neo = np.array([[40.0, 40.0, 0.0]])\n"
                  "def _probe(fn):\n    try:\n        fn(ee, eo, 2.0, 0.25, 1e-10, 50)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(shake_fraction)", "gold_call": "_probe(_oracle_shake_fraction)"},
    ]
