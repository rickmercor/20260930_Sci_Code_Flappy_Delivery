"""
Step 06: E1 rates of the lines to the bound odd-parity levels. Partial electric-dipole rates of the lines from the lowest even-parity L = 1 triplet state to the bound odd-parity L = 1 triplet levels.

Spontaneous emission in the electric-dipole approximation connects the initial level to final states of opposite parity, and the rate carries the cube of the photon energy together with the square of the dipole matrix element between normalized states. For the two-electron atom treated here the initial level is the lowest state of the unnatural-parity set and the discrete final states are the 1snp levels of the natural-parity set, whose energies lie below the one-electron threshold -Z^2/2 hartree of the residual ion. The strongest of these is the one-electron jump that leaves the second electron a spectator.

The initial and final states come from separate variational calculations in bases of the two parities, each with its own overlap matrix, and the lines are the part of the radiative width that ends on a bound state. The rest of the width, which ends in the continuum above the threshold, is not described by these lines.

For states of total orbital angular momentum one the sum over photon polarizations and over the magnetic sublevels of the final state, for a fixed sublevel of the initial state, reduces to a single Cartesian matrix element with a fixed multiplicity, and the result does not depend on which sublevel of the initial state is chosen. The value of that multiplicity for the angular factors used here is given with the parameters below.

Returns
-------
numpy.ndarray of shape (k, 3): final energy, photon energy and E1 rate in s^-1 for every line to a bound odd-parity level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bound_line_rates(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, cut: float) -> "np.ndarray":
    '''Photon energies and E1 rates of the lines to the bound odd-parity levels below the lowest even-parity state.

    Parameters
    ----------
    exps_even : np.ndarray
        Array of shape (n, 3) of exponents (a, b, c) in bohr^-2 for the even basis (1 - P12)[(r1_vec x r2_vec) g];
        finite, non-negative, a b + c (a + b) > 0.
    exps_odd : np.ndarray
        Array of shape (m, 3), same rules, for the odd basis (1 - P12)[r1_vec g], g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge, Z > 0, of an infinitely heavy nucleus.
    cut : float
        Relative threshold, 0 < cut < 1, on the eigenvalues of the overlap matrix of the unnormalized basis functions,
        applied separately to each basis before the Hamiltonian -(nabla_1^2 + nabla_2^2)/2 - Z/r1 - Z/r2 + 1/r12 is
        diagonalized without rotation.

    Returns
    -------
    result : np.ndarray
        Real array of shape (k, 3) with one row per odd-parity eigenstate whose energy lies below both the one-electron
        threshold -Z^2/2 hartree and the lowest even-parity energy E_i, in ascending order of E_f. The columns are E_f
        in hartree, the photon energy omega = E_i - E_f in hartree, and the rate
        A_f = (4/3) alpha^3 omega^3 * 2 * D_f^2 / t_au in s^-1, where D_f is the length-form dipole matrix element
        between the two states normalized with their own overlap matrices, the factor 2 is the polarization and
        sublevel multiplicity, alpha = 7.2973525693e-3 and t_au = 2.4188843265857e-17 s. k may be zero.

    Raises
    ------
    ValueError
        If either exponent array is malformed or non-integrable, Z is not positive, or cut is not strictly between
        0 and 1.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bound_line_rates(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, cut: float) -> "np.ndarray":
    fine_structure = 7.2973525693e-3
    au_time_s = 2.4188843265857e-17
    even = _oracle_rotated_levels(exps_even, "even", Z, 0.0, cut)
    odd = _oracle_rotated_levels(exps_odd, "odd", Z, 0.0, cut)
    e_init = even[0, 0].real
    c_init = even[0, 1:].real
    e_final = odd[:, 0].real
    c_final = odd[:, 1:].real.T
    coupling = c_init @ _oracle_ecg_dipole(exps_even, exps_odd, "length") @ c_final
    omega = e_init - e_final
    below = (e_final < -0.5 * float(Z) ** 2) & (omega > 0.0)
    rate = ((4.0 / 3.0) * fine_structure ** 3 * omega[below] ** 3
            * 2.0 * coupling[below] ** 2 / au_time_s)
    return np.column_stack([e_final[below], omega[below], rate]).astype(float)

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
def logged(table):
    t = np.asarray(table, dtype=float).reshape(-1, 3)
    return np.column_stack([t[:, 0], t[:, 1], np.log(t[:, 2])])
"""
    return [
        # --- Normal: helium, energies, photon energies and log rates ---
        {"setup": setup, "call": "logged(bound_line_rates(ee.copy(), eo.copy(), 2.0, 1e-12))",
         "gold_call": "logged(_oracle_bound_line_rates(ee.copy(), eo.copy(), 2.0, 1e-12))", "tol": 1e-6},
        # --- Normal: Li+ with the same basis shape scaled by (Z/2)^2 ---
        {"setup": setup, "call": "logged(bound_line_rates(ee * 2.25, eo * 2.25, 3.0, 1e-12))",
         "gold_call": "logged(_oracle_bound_line_rates(ee * 2.25, eo * 2.25, 3.0, 1e-12))", "tol": 1e-6},
        # --- Boundary: how many final states count as bound, which fixes where the line list stops ---
        {"setup": setup, "call": "float(len(np.asarray(bound_line_rates(ee.copy(), eo.copy(), 2.0, 1e-6), dtype=float).reshape(-1, 3)))",
         "gold_call": "float(len(np.asarray(_oracle_bound_line_rates(ee.copy(), eo.copy(), 2.0, 1e-6), dtype=float).reshape(-1, 3)))"},
        # --- Edge: an odd basis so compact that no state is bound gives an empty table ---
        {"setup": "import numpy as np\nee = np.array([[0.02, 0.02, 0.0]])\neo = np.array([[40.0, 40.0, 0.0]])\n"
                  "def shape_code(table):\n    t = np.asarray(table, dtype=float).reshape(-1, 3)\n    return 10.0 * t.shape[0] + t.shape[1]\n",
         "call": "shape_code(bound_line_rates(ee.copy(), eo.copy(), 2.0, 1e-10))",
         "gold_call": "shape_code(_oracle_bound_line_rates(ee.copy(), eo.copy(), 2.0, 1e-10))"},
        # --- Error: non-positive nuclear charge ---
        {"setup": "import numpy as np\ne = np.array([[0.3, 0.3, 0.05]])\n"
                  "def _probe(fn):\n    try:\n        fn(e, e, -2.0, 1e-10)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(bound_line_rates)", "gold_call": "_probe(_oracle_bound_line_rates)"},
    ]
