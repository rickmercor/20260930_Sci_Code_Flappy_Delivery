"""
Build the two bare two-particle bubbles from a propagator: the particle-hole product G(nu)G(nu+omega) and the particle-particle product G(nu)G(-nu-omega).

Fermionic Matsubara frequencies are nu = (2n+1) pi / beta and bosonic ones are omega = 2 m pi / beta, and every frequency is referred to below by its INTEGER label n or m rather than by its value. A box of Nf fermionic frequencies holds the integer labels n = -(Nf//2) up to Nf - Nf//2 - 1 inclusive, and a box of Nb bosonic frequencies holds the labels m over the same range with Nb in place of Nf; both are stored in increasing order of the label. With Nf even the fermionic frequencies are symmetric about zero, and with Nb odd the bosonic frequencies are symmetric about zero and include it. Any frequency argument whose label falls outside its box contributes zero. Both bubbles are diagonal in the first fermionic index, so only the diagonal is returned, as a function of nu and omega. In integer labels the shifted arguments are n+m for the particle-hole product and -n-m-1 for the particle-particle one; the latter follows from -nu-omega = (2(-n-m-1)+1) pi / beta. Neither bubble carries a factor of beta or a sign here.

Returns
-------
numpy array of shape (2, Nf, Nb), complex, the particle-hole bubble first and the particle-particle bubble second
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_bubbles(G: 'np.ndarray', Nb: int) -> 'np.ndarray':
    """Build the two bare two-particle bubbles from a propagator: the particle-hole product G(nu)G(nu+omega) and the particle-particle product G(nu)G(-nu-omega).

    Parameters
    ----------
    G : array of shape (Nf,), complex, the propagator on the fermionic box
    Nb : int, number of bosonic frequencies

    Returns
    -------
    numpy array of shape (2, Nf, Nb), complex, the particle-hole bubble first and the particle-particle bubble second

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




def _oracle_pair_bubbles(G: 'np.ndarray', Nb: int) -> 'np.ndarray':
    G = np.asarray(G, dtype=complex)
    if G.ndim != 1:
        raise ValueError("the Green's function must be a one-dimensional array over nu")
    if Nb < 1:
        raise ValueError("the bosonic box must hold at least one frequency")
    Nf = G.size
    nf, nb = _nf_int(Nf), _nb_int(Nb)
    out = np.zeros((2, Nf, Nb), dtype=complex)
    rows = np.arange(Nf)
    for k, m in enumerate(nb):
        jph = _fidx(Nf, nf + m)
        ok = jph >= 0
        out[0, ok, k] = G[rows[ok]]*G[jph[ok]]
        jpp = _fidx(Nf, -nf - m - 1)
        ok = jpp >= 0
        out[1, ok, k] = G[rows[ok]]*G[jpp[ok]]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nnu = (2*np.arange(-4,4)+1)*np.pi\nG = 1.0/(1j*nu + 0.5)\nG_g = G.copy()",
         "call": "pair_bubbles(G, 7)", "gold_call": "_oracle_pair_bubbles(G_g, 7)"},
        # boundary: a single bosonic frequency, omega = 0
        {"setup": "import numpy as np\nnu = (2*np.arange(-3,3)+1)*np.pi\nG = 1.0/(1j*nu)\nG_g = G.copy()",
         "call": "pair_bubbles(G, 1)", "gold_call": "_oracle_pair_bubbles(G_g, 1)"},
        # edge: a bosonic box wider than the fermionic one, so most shifts leave the box
        {"setup": "import numpy as np\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 1.0)\nG_g = G.copy()",
         "call": "pair_bubbles(G, 9)", "gold_call": "_oracle_pair_bubbles(G_g, 9)"},
    ]
