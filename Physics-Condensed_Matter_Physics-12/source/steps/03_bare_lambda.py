"""
Give the spin-diagonalised bare interaction in the four channels, derived from the spin structure of the on-site interaction.

The frequency-independent bare vertex of an on-site interaction carries no frequency structure, so each channel is a single number. Derive the four from the spin structure: an on-site interaction acts only between opposite spins, so the equal-spin bare vertex vanishes and the opposite-spin one is U. The density and magnetic constants are the sum and the difference of the equal-spin and opposite-spin vertices, in that order. The singlet and triplet constants are the sum and the difference of the direct and exchange opposite-spin terms, which coincide for a frequency-independent interaction. Note the consequence for what follows: the density and magnetic constants come out equal and opposite, so any later combination that ADDS them loses their leading order while one that SUBTRACTS them doubles it.

Returns
-------
numpy array of shape (4,), complex, the channel constants ordered density, magnetic, singlet, triplet
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bare_lambda(U: float) -> 'np.ndarray':
    """Give the spin-diagonalised bare interaction in the four channels, derived from the spin structure of the on-site interaction.

    Parameters
    ----------
    U : float, on-site interaction

    Returns
    -------
    numpy array of shape (4,), complex, the channel constants ordered density, magnetic, singlet, triplet

    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def _oracle_bare_lambda(U: float) -> 'np.ndarray':
    if not np.isfinite(U):
        raise ValueError("the interaction must be finite")
    return np.array([U, -U, 2.0*U, 0.0], dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np", "call": "bare_lambda(5.0)",
         "gold_call": "_oracle_bare_lambda(5.0)"},
        # boundary: the non-interacting limit
        {"setup": "import numpy as np", "call": "bare_lambda(0.0)",
         "gold_call": "_oracle_bare_lambda(0.0)"},
        # edge: attraction reverses every channel
        {"setup": "import numpy as np", "call": "bare_lambda(-2.5)",
         "gold_call": "_oracle_bare_lambda(-2.5)"},
    ]
