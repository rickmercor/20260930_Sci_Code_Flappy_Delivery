"""
Tabulate the harmonic oscillator eigenfunctions that the phonon matrix elements are built from, on a uniform grid of the reduced coordinate, for every quantum number from zero up to the retained cut-off. Normalise each eigenfunction so that the grid quadrature reproduces orthonormality, and generate them by the three-term recurrence in the normalised functions rather than by evaluating Hermite polynomials and a Gaussian separately, which overflows well before the cut-off this calculation needs. Return the grid in the first row and the eigenfunctions in the rows below it.

The transition rate of the model is a sum over phonon quantum numbers that runs to several hundred, because the energy released by the electronic transition has to be absorbed by that many phonons. At those orders the Hermite polynomial and the Gaussian are separately astronomically large and small, so the only stable route is a recurrence that carries the normalised product.

Returns
-------
A (n_phonon + 2, n_quad) float64 array whose first row is the reduced coordinate grid and whose row k + 1 is the normalised eigenfunction of quantum number k.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def oscillator_basis(n_phonon, u_lo, u_hi, n_quad):
    """Tabulate the harmonic oscillator eigenfunctions that the phonon matrix elements are
    built from, on a uniform grid of the reduced coordinate, for every quantum number from
    zero up to the retained cut-off. A (n_phonon + 2, n_quad) float64 array whose first row
    is the reduced coordinate grid and whose row k + 1 is the normalised eigenfunction of
    quantum number k."""
    return np.zeros((n_phonon + 2, n_quad))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_oscillator_basis(n_phonon, u_lo, u_hi, n_quad):
    n_phonon = int(n_phonon); n_quad = int(n_quad)
    u_lo = float(u_lo); u_hi = float(u_hi)
    if n_phonon < 1 or n_quad < 3 or not u_hi > u_lo:
        raise ValueError("n_phonon, n_quad and the quadrature window must be valid")
    u = np.linspace(u_lo, u_hi, n_quad)
    out = np.empty((n_phonon+2, n_quad))
    out[0] = u
    out[1] = np.pi**-0.25*np.exp(-0.5*u*u)
    if n_phonon >= 1:
        out[2] = np.sqrt(2.0)*u*out[1]
    for k in range(2, n_phonon+1):
        out[k+1] = np.sqrt(2.0/k)*u*out[k] - np.sqrt((k-1.0)/k)*out[k-1]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn_phonon = 40\nu_lo = -14.0\nu_hi = 14.0\nn_quad = 6001\n',
         'call': 'oscillator_basis(n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_oscillator_basis(n_phonon, u_lo, u_hi, n_quad)'},
        {'setup': 'import numpy as np\nn_phonon = 120\nu_lo = -22.0\nu_hi = 30.0\nn_quad = 12001\n',
         'call': 'oscillator_basis(n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_oscillator_basis(n_phonon, u_lo, u_hi, n_quad)'},
        {'setup': 'import numpy as np\nn_phonon = 220\nu_lo = -35.0\nu_hi = 60.0\nn_quad = 47501\n',
         'call': 'oscillator_basis(n_phonon, u_lo, u_hi, n_quad)',
         'gold_call': '_oracle_oscillator_basis(n_phonon, u_lo, u_hi, n_quad)'},
    ]
