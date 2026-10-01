"""
Step 07: Energy-resolved rate into the continuum final states. Energy-resolved rate of the radiative transitions that leave the atom in the continuum above the one-electron threshold.

Part of the radiative width of a level that sits above a break-up threshold but cannot break up ends on final states in the continuum rather than on a discrete level. That part is not a set of lines but a smooth distribution over the energy of the final state, and the quantity of interest is the rate per unit photon energy. Expressed through the resolvent of the Hamiltonian, the distribution at a chosen final energy is an imaginary part of a sum over states weighted by their dipole coupling to the initial state, which is why one calculation carried out at a single rotation angle yields the whole distribution on an arbitrarily fine energy grid.



Each contribution to that sum is a square of a matrix element rather than a squared modulus, because complex-rotated states pair with their transposes rather than their conjugate transposes. Bound final states sit on the real energy axis and contribute nothing away from their own energies, so the distribution evaluated between the threshold and the initial energy describes only the continuum. Its integral over the final-state energy is the part of the radiative width that ends in the continuum, and both the shape and that integral must be stationary against the rotation angle once the basis supports the rotated continuum.



The initial state is the lowest state of the unnatural-parity set at the same rotation angle. Under rotation the dipole operator in the length form carries one factor of the coordinate scaling, and the photon energy is measured from the real part of the initial energy. The multiplicity that turns one Cartesian matrix element into the sum over polarizations and final sublevels is the same as for the discrete lines.

Returns
-------
numpy.ndarray of shape (q,): rate per unit photon energy in s^-1 per hartree at each requested continuum final-state energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dissociation_spectrum(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, theta: float, cut: float,
                          energies: "np.ndarray") -> "np.ndarray":
    '''Rate per unit photon energy into continuum final states, at given final-state energies.

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
        Rotation angle in radians, 0 < theta < pi/4, applied to both bases as for rotated_levels.
    cut : float
        Relative threshold, 0 < cut < 1, on the overlap eigenvalues, applied separately to each basis.
    energies : np.ndarray
        One-dimensional array of q >= 1 finite final-state energies E_F in hartree at which the distribution is
        evaluated.

    Returns
    -------
    result : np.ndarray
        Real array of shape (q,): the energy-resolved electric-dipole continuum rate density in s^-1 per hartree
        at each supplied final-state energy. The polarization and sublevel multiplicity is 2.
        alpha = 7.2973525693e-3 and t_au = 2.4188843265857e-17 s.

    Raises
    ------
    ValueError
        If either exponent array is malformed or non-integrable, Z is not positive, theta is outside (0, pi/4), cut is
        not strictly between 0 and 1, or energies is not a one-dimensional array of at least one finite value.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dissociation_spectrum(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, theta: float, cut: float,
                                  energies: "np.ndarray") -> "np.ndarray":
    fine_structure = 7.2973525693e-3
    au_time_s = 2.4188843265857e-17
    grid = np.asarray(energies, dtype=float)
    if grid.ndim != 1 or grid.size < 1 or not np.all(np.isfinite(grid)):
        raise ValueError("energies must be a one-dimensional array of finite values")
    if not float(theta) > 0.0:
        raise ValueError("theta must be strictly positive")
    even = _oracle_rotated_levels(exps_even, "even", Z, theta, cut)
    odd = _oracle_rotated_levels(exps_odd, "odd", Z, theta, cut)
    e_init = even[0, 0]
    coupling = (even[0, 1:] @ (np.exp(1.0j * float(theta)) * _oracle_ecg_dipole(exps_even, exps_odd, "length"))
                @ odd[:, 1:].T)
    weight = coupling ** 2
    values = odd[:, 0]
    dens = np.array([np.sum(weight / (values - e)).imag for e in grid])
    omega = e_init.real - grid
    spec = (4.0 / 3.0) * fine_structure ** 3 * omega ** 3 * (2.0 / np.pi) * dens / au_time_s
    return np.asarray(spec, dtype=float)

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
grid = np.linspace(-1.95, -0.85, 12)
"""
    return [
        # --- Normal: helium at the paper's rotation angle ---
        {"setup": setup, "call": "dissociation_spectrum(ee.copy(), eo.copy(), 2.0, 0.25, 1e-12, grid.copy())",
         "gold_call": "_oracle_dissociation_spectrum(ee.copy(), eo.copy(), 2.0, 0.25, 1e-12, grid.copy())", "tol": 1e-4},
        # --- Normal: a different angle, which changes the rotated states but not the physics they encode ---
        {"setup": setup, "call": "dissociation_spectrum(ee.copy(), eo.copy(), 2.0, 0.14, 1e-12, grid.copy())",
         "gold_call": "_oracle_dissociation_spectrum(ee.copy(), eo.copy(), 2.0, 0.14, 1e-12, grid.copy())", "tol": 1e-4},
        # --- Normal: Li+ with the scaled basis on its own energy window ---
        {"setup": setup + "grid = np.linspace(-4.4, -2.0, 9)\n",
         "call": "dissociation_spectrum(ee * 2.25, eo * 2.25, 3.0, 0.2, 1e-12, grid.copy())",
         "gold_call": "_oracle_dissociation_spectrum(ee * 2.25, eo * 2.25, 3.0, 0.2, 1e-12, grid.copy())", "tol": 1e-4},
        # --- Edge: a single energy, given as a one-element array ---
        {"setup": setup, "call": "dissociation_spectrum(ee.copy(), eo.copy(), 2.0, 0.25, 1e-12, np.array([-1.5]))",
         "gold_call": "_oracle_dissociation_spectrum(ee.copy(), eo.copy(), 2.0, 0.25, 1e-12, np.array([-1.5]))",
         "tol": 1e-4},
        # --- Error: an unrotated calculation cannot give a continuum distribution ---
        {"setup": setup + "def _probe(fn):\n    try:\n        fn(ee, eo, 2.0, 0.0, 1e-12, grid)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(dissociation_spectrum)", "gold_call": "_probe(_oracle_dissociation_spectrum)"},
    ]
