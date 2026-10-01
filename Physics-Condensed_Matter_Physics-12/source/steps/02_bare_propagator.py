"""
Give the non-interacting Green's function of the hybridised single-site model on the fermionic box, including the interaction-induced shift of the chemical potential and the damping supplied by the bath.

Fermionic Matsubara frequencies are nu = (2n+1) pi / beta and bosonic ones are omega = 2 m pi / beta, and every frequency is referred to below by its INTEGER label n or m rather than by its value. A box of Nf fermionic frequencies holds the integer labels n = -(Nf//2) up to Nf - Nf//2 - 1 inclusive, and a box of Nb bosonic frequencies holds the labels m over the same range with Nb in place of Nf; both are stored in increasing order of the label. With Nf even the fermionic frequencies are symmetric about zero, and with Nb odd the bosonic frequencies are symmetric about zero and include it. Any frequency argument whose label falls outside its box contributes zero. The site Hamiltonian is U(n_up - 1/2)(n_dn - 1/2) - dmu(n_up + n_dn). Expanding the product moves a term -U/2 (n_up + n_dn) into the quadratic part, so the effective chemical potential is dmu + U/2. The site is additionally coupled to a wide, flat bath, whose only effect on the Matsubara axis is a constant imaginary part: the inverse bare propagator is i nu + dmu + U/2 + i hyb sign(nu), which is the flat-band limit of a hybridisation function and damps the propagator uniformly at every frequency. That U/2 is cancelled later by the Hartree term of the self-energy, which is what leaves the interacting propagator particle-hole symmetric at dmu = 0.

Returns
-------
numpy array of shape (Nf,), complex, the propagator on the fermionic box
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bare_propagator(beta: float, Nf: int, U: float, dmu: float, hyb: float) -> 'np.ndarray':
    """Give the non-interacting Green's function of the hybridised single-site model on the fermionic box, including the interaction-induced shift of the chemical potential and the damping supplied by the bath.

    Parameters
    ----------
    beta : float, inverse temperature
    Nf : int, number of fermionic frequencies
    U : float, on-site interaction
    dmu : float, chemical potential measured from half filling
    hyb : float, hybridisation strength of the flat bath, non-negative

    Returns
    -------
    numpy array of shape (Nf,), complex, the propagator on the fermionic box

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




def _oracle_bare_propagator(beta: float, Nf: int, U: float, dmu: float, hyb: float) -> 'np.ndarray':
    if not np.isfinite([beta, U, dmu, hyb]).all():
        raise ValueError("the model parameters must be finite")
    if beta <= 0.0:
        raise ValueError("the inverse temperature must be positive")
    if hyb < 0.0:
        raise ValueError("the hybridisation strength must not be negative")
    nu = _oracle_matsubara_grids(beta, Nf, 1)[:Nf]
    return 1.0/(1j*nu + dmu + 0.5*U + 1j*hyb*np.sign(nu))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np", "call": "bare_propagator(1.0, 8, 5.0, 1.0, 0.5)",
         "gold_call": "_oracle_bare_propagator(1.0, 8, 5.0, 1.0, 0.5)"},
        # boundary: no interaction and no bath, so only the chemical potential survives
        {"setup": "import numpy as np", "call": "bare_propagator(1.0, 8, 0.0, 1.0, 0.0)",
         "gold_call": "_oracle_bare_propagator(1.0, 8, 0.0, 1.0, 0.0)"},
        # edge: particle-hole symmetry with a strong bath, which probes the sign structure of the damping
        {"setup": "import numpy as np", "call": "bare_propagator(2.0, 6, 3.0, 0.0, 1.25)",
         "gold_call": "_oracle_bare_propagator(2.0, 6, 3.0, 0.0, 1.25)"},
    ]
