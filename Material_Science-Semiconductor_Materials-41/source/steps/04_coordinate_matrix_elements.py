"""
Build the matrix of phonon matrix elements that the transition rate of the source is actually made of. The source writes the rate with one particular operator between the initial-state and final-state phonon states, and it is not the identity; the whole difference from the older model rests on which operator it is. Return the matrix in the mass-weighted coordinate measured from the centre of the initial-state surface, indexed with the initial quantum number first, on the same grid as the previous steps.

The older treatment factorises the electronic and vibrational parts of the transition matrix element and keeps the vibrational overlap alone. The source keeps the first derivative of the electronic coupling with respect to the coordinate instead, which leaves a different operator sandwiched between the phonon states and therefore a different weighting of the phonon sum. A solver who builds the wrong one gets a lineshape that differs from the right one by a factor that is nearly independent of temperature, which is exactly why the distinction has been easy to overlook.

Returns
-------
An (n_phonon + 1, n_phonon + 1) float64 array of phonon matrix elements in amu^(1/2) Angstrom, entry (m, n) pairing initial-state phonon state m with final-state phonon state n.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):
    """Build the matrix of phonon matrix elements that the transition rate of the source is
    actually made of. An (n_phonon + 1, n_phonon + 1) float64 array of phonon matrix
    elements in amu^(1/2) Angstrom, entry (m, n) pairing initial-state phonon state m with
    final-state phonon state n."""
    return np.zeros((n_phonon + 1, n_phonon + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


DELTA_E   = 1.02

def _oracle_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):
    if float(delta_q) <= 0.0 or float(hw) <= 0.0 or int(n_phonon) < 1:
        raise ValueError("delta_q, hw must be positive and n_phonon at least one")
    ell = float(_oracle_configuration_coordinate(delta_q, hw, DELTA_E)[0])
    tab = _oracle_oscillator_basis(n_phonon, u_lo, u_hi, n_quad)
    u, phi = tab[0], tab[1:]
    du = u[1]-u[0]
    shift = float(delta_q)/ell
    phi_f = _oracle_oscillator_basis(n_phonon, u_lo-shift, u_hi-shift, n_quad)[1:]
    return ((phi*u*du) @ phi_f.T)*ell

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndelta_q = 1.20\nhw = 0.038\nn_phonon = 60\nu_lo = -18.0\nu_hi = 22.0\nn_quad = 12001\n',
         'call': 'coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)'},
        {'setup': 'import numpy as np\ndelta_q = 4.90\nhw = 0.038\nn_phonon = 220\nu_lo = -35.0\nu_hi = 60.0\nn_quad = 47501\n',
         'call': 'coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)'},
        {'setup': 'import numpy as np\ndelta_q = 2.50\nhw = 0.045\nn_phonon = 70\nu_lo = -20.0\nu_hi = 26.0\nn_quad = 14001\n',
         'call': 'coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)'},
    ]
