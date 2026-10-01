"""
Assemble the channel-irreducible vertices by combining the bare channel constants with the reducible vertices of all four channels under the frequency substitutions and weights given below.

Fermionic Matsubara frequencies are nu = (2n+1) pi / beta and bosonic ones are omega = 2 m pi / beta, and every frequency is referred to below by its INTEGER label n or m rather than by its value. A box of Nf fermionic frequencies holds the integer labels n = -(Nf//2) up to Nf - Nf//2 - 1 inclusive, and a box of Nb bosonic frequencies holds the labels m over the same range with Nb in place of Nf; both are stored in increasing order of the label. With Nf even the fermionic frequencies are symmetric about zero, and with Nb odd the bosonic frequencies are symmetric about zero and include it. Any frequency argument whose label falls outside its box contributes zero. Vertices are indexed [channel, nu, nu', omega] with the channels ordered density, magnetic, singlet, triplet. Write A for the substitution (n, n', m) -> (n, n+m, n'-n), B for (n, n', m) -> (n, n', -m-n-n'-1) and C for (n, n', m) -> (n, -n'-m-1, n'-n), each acting on the integer labels of the entry being written. The four channels are the standard spin combinations of the two-particle vertex: writing X_uu and X_ud for its equal- and opposite-spin components, the two particle-hole channels are X_d = X_uu + X_ud and X_m = X_uu - X_ud, and the two particle-particle channels are X_t = X_uu and X_s = 2 X_ud - X_uu. Then
    Gamma_d = Lambda_d - (1/2) Phi_d[A] - (3/2) Phi_m[A] + (1/2) Phi_s[B] + (3/2) Phi_t[B],
    Gamma_m = Lambda_m - (1/2) Phi_d[A] + (1/2) Phi_m[A] - (1/2) Phi_s[B] + (1/2) Phi_t[B],
    Gamma_s = Lambda_s + (1/2) Phi_d[B] - (3/2) Phi_m[B] + (1/2) Phi_d[C] - (3/2) Phi_m[C],
    Gamma_t = Lambda_t + (1/2) Phi_d[B] + (1/2) Phi_m[B] - (1/2) Phi_d[C] - (1/2) Phi_m[C].
Note that the density and magnetic channels contribute to their own irreducible vertex under A, not only to the other channels. The channel constants carry no frequency dependence and broadcast over all three indices.

Returns
-------
numpy array of shape (4, Nf, Nf, Nb), complex, the irreducible vertices in the same channel order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def irreducible_vertices(Phi: 'np.ndarray', Lam: 'np.ndarray') -> 'np.ndarray':
    """Assemble the channel-irreducible vertices by combining the bare channel constants with the reducible vertices of all four channels under the frequency substitutions and weights given below.

    Parameters
    ----------
    Phi : array of shape (4, Nf, Nf, Nb), complex, the reducible vertices
    Lam : array of shape (4,), the bare channel constants

    Returns
    -------
    numpy array of shape (4, Nf, Nf, Nb), complex, the irreducible vertices in the same channel order

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




def _oracle_irreducible_vertices(Phi: 'np.ndarray', Lam: 'np.ndarray') -> 'np.ndarray':
    Phi = np.asarray(Phi, dtype=complex)
    if Phi.ndim != 4 or Phi.shape[0] != 4:
        raise ValueError("Phi must have shape (4, Nf, Nf, Nb) ordered d, m, s, t")
    Lam = np.asarray(Lam, dtype=complex)
    if Lam.shape != (4,):
        raise ValueError("Lambda must hold the four channel constants d, m, s, t")
    Nf, Nb = Phi.shape[1], Phi.shape[3]
    if Phi.shape[2] != Nf:
        raise ValueError("the two fermionic axes of Phi must have equal length")
    A, B, C = _shift_indices(Nf, Nb)
    dA, mA = _gather(Phi[0], *A), _gather(Phi[1], *A)
    sB, tB = _gather(Phi[2], *B), _gather(Phi[3], *B)
    dB, mB = _gather(Phi[0], *B), _gather(Phi[1], *B)
    dC, mC = _gather(Phi[0], *C), _gather(Phi[1], *C)
    Gam = np.empty_like(Phi)
    Gam[0] = Lam[0] - 0.5*dA - 1.5*mA + 0.5*sB + 1.5*tB
    Gam[1] = Lam[1] - 0.5*dA + 0.5*mA - 0.5*sB + 0.5*tB
    Gam[2] = Lam[2] + 0.5*dB - 1.5*mB + 0.5*dC - 1.5*mC
    Gam[3] = Lam[3] + 0.5*dB + 0.5*mB - 0.5*dC - 0.5*mC
    return Gam

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # normal: a generic reducible vertex
        {"setup": "import numpy as np\nrng = np.random.default_rng(3)\nPhi = (rng.normal(size=(4,4,4,3)) + 1j*rng.normal(size=(4,4,4,3)))*0.3\nLam = np.array([2.0,-2.0,4.0,1.0], dtype=complex)\nPhi_g = Phi.copy()\nLam_g = Lam.copy()",
         "call": "irreducible_vertices(Phi, Lam)", "gold_call": "_oracle_irreducible_vertices(Phi_g, Lam_g)"},
        # boundary: no reducible vertex at all, so Gamma must collapse onto Lambda
        {"setup": "import numpy as np\nPhi = np.zeros((4,4,4,3), dtype=complex)\nLam = np.array([1.5,-1.5,3.0,0.0], dtype=complex)\nPhi_g = Phi.copy()\nLam_g = Lam.copy()",
         "call": "irreducible_vertices(Phi, Lam)", "gold_call": "_oracle_irreducible_vertices(Phi_g, Lam_g)"},
        # edge: one channel alone, which isolates the frequency shifts that carry it
        {"setup": "import numpy as np\nPhi = np.zeros((4,6,6,5), dtype=complex)\nPhi[1] = 1.0\nLam = np.zeros(4, dtype=complex)\nPhi_g = Phi.copy()\nLam_g = Lam.copy()",
         "call": "irreducible_vertices(Phi, Lam)", "gold_call": "_oracle_irreducible_vertices(Phi_g, Lam_g)"},
    ]
