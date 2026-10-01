"""
Build the matrix of overlaps between the phonon states of the two charge states, the initial-state oscillator centred at the origin of the mass-weighted coordinate and the final-state oscillator centred at the lattice relaxation. Integrate on the grid of the previous step and use the same grid spacing as the quadrature weight. Index the matrix with the initial quantum number first.

These are the Franck-Condon factors of the transition. They are not what the rate of this model is built from, but they are the natural reference: the older treatment of nonradiative capture keeps only these, and the departure from that treatment is the point of the source. They also provide the cleanest check on the basis, since the squared overlaps out of any low-lying initial state must sum to one.

Returns
-------
An (n_phonon + 1, n_phonon + 1) float64 array whose entry (m, n) is the overlap of the initial-state phonon state m with the final-state phonon state n.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):
    """Build the matrix of overlaps between the phonon states of the two charge states, the
    initial-state oscillator centred at the origin of the mass-weighted coordinate and the
    final-state oscillator centred at the lattice relaxation. An (n_phonon + 1, n_phonon +
    1) float64 array whose entry (m, n) is the overlap of the initial-state phonon state m
    with the final-state phonon state n."""
    return np.zeros((n_phonon + 1, n_phonon + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


DELTA_E   = 1.02

def _oracle_franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):
    if float(delta_q) <= 0.0 or float(hw) <= 0.0 or int(n_phonon) < 1:
        raise ValueError("delta_q, hw must be positive and n_phonon at least one")
    ell = float(_oracle_configuration_coordinate(delta_q, hw, DELTA_E)[0])
    tab = _oracle_oscillator_basis(n_phonon, u_lo, u_hi, n_quad)
    u, phi = tab[0], tab[1:]
    du = u[1]-u[0]
    shift = float(delta_q)/ell
    phi_f = _oracle_oscillator_basis(n_phonon, u_lo-shift, u_hi-shift, n_quad)[1:]
    return (phi*du) @ phi_f.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndelta_q = 1.20\nhw = 0.038\nn_phonon = 60\nu_lo = -18.0\nu_hi = 22.0\nn_quad = 12001\n',
         'call': 'franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)'},
        {'setup': 'import numpy as np\ndelta_q = 4.90\nhw = 0.038\nn_phonon = 90\nu_lo = -24.0\nu_hi = 34.0\nn_quad = 18001\n',
         'call': 'franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)'},
        {'setup': 'import numpy as np\ndelta_q = 2.50\nhw = 0.045\nn_phonon = 70\nu_lo = -20.0\nu_hi = 26.0\nn_quad = 14001\n',
         'call': 'franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad)'},
    ]
