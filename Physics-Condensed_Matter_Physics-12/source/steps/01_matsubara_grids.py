"""
Build the fermionic and bosonic Matsubara grids of a finite frequency box and return them concatenated, fermionic first.

Fermionic Matsubara frequencies are nu = (2n+1) pi / beta and bosonic ones are omega = 2 m pi / beta, and every frequency is referred to below by its INTEGER label n or m rather than by its value. A box of Nf fermionic frequencies holds the integer labels n = -(Nf//2) up to Nf - Nf//2 - 1 inclusive, and a box of Nb bosonic frequencies holds the labels m over the same range with Nb in place of Nf; both are stored in increasing order of the label. With Nf even the fermionic frequencies are symmetric about zero, and with Nb odd the bosonic frequencies are symmetric about zero and include it. Any frequency argument whose label falls outside its box contributes zero. Everything downstream is indexed by position in these grids, so the ordering fixes the layout of every later object.

Returns
-------
numpy array of shape (Nf+Nb,), the Nf fermionic frequencies followed by the Nb bosonic ones, each in increasing order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def matsubara_grids(beta: float, Nf: int, Nb: int) -> 'np.ndarray':
    """Build the fermionic and bosonic Matsubara grids of a finite frequency box and return them concatenated, fermionic first.

    Parameters
    ----------
    beta : float, inverse temperature
    Nf : int, number of fermionic frequencies
    Nb : int, number of bosonic frequencies

    Returns
    -------
    numpy array of shape (Nf+Nb,), the Nf fermionic frequencies followed by the Nb bosonic ones, each in increasing order

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




def _oracle_matsubara_grids(beta: float, Nf: int, Nb: int) -> 'np.ndarray':
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("the inverse temperature must be positive and finite")
    if Nf < 2 or Nb < 1:
        raise ValueError("the boxes must hold at least two fermionic and one bosonic frequency")
    nu = (2.0*_nf_int(Nf) + 1.0)*np.pi/beta
    om = 2.0*_nb_int(Nb)*np.pi/beta
    return np.concatenate([nu, om])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # normal: an even fermionic box and an odd bosonic one
        {"setup": "import numpy as np", "call": "matsubara_grids(1.0, 8, 7)",
         "gold_call": "_oracle_matsubara_grids(1.0, 8, 7)"},
        # boundary: the smallest legal boxes
        {"setup": "import numpy as np", "call": "matsubara_grids(1.0, 2, 1)",
         "gold_call": "_oracle_matsubara_grids(1.0, 2, 1)"},
        # edge: a low temperature, where the grids are dense near zero
        {"setup": "import numpy as np", "call": "matsubara_grids(12.5, 6, 5)",
         "gold_call": "_oracle_matsubara_grids(12.5, 6, 5)"},
    ]
