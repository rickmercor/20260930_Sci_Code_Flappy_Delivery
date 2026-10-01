"""
Recover the channel-irreducible vertices from the FULL vertices by inverting the ladder relations, rather than by summing them.

The ladder relation of the previous sweep expresses the reducible vertex as a product of the irreducible one with a bubble and the full one, channel by channel and at fixed bosonic frequency, where the full vertex is F = Phi + Gamma. Written as matrices in the two fermionic labels at fixed bosonic label, with D the diagonal matrix of the relevant bubble, those relations are
    Phi_d = (1/beta) Gamma_d D_ph F_d,   Phi_m = (1/beta) Gamma_m D_ph F_m,
    Phi_s = -(1/(2 beta)) F_s D_pp Gamma_s,   Phi_t = +(1/(2 beta)) F_t D_pp Gamma_t.
Substitute Phi = F - Gamma and solve each relation for Gamma. Each is linear in Gamma and the solution is a single matrix inverse per channel and bosonic frequency; which side that inverse ends up on is fixed by where the bubble sits in the relation above, so read them carefully rather than assuming the four channels come out alike. The bubble is diagonal in the contracted fermionic label, so it enters as a diagonal matrix and the products above do not commute; the two orders that differ only by the push-through identity, F (1 + D F)^-1 and (1 + F D)^-1 F, are however the same matrix, so either may be used. What is not interchangeable is the SIGN, which is opposite in the singlet and triplet relations, and the factor, which is 1/beta in the two particle-hole channels and 1/(2 beta) in the two particle-particle ones. This is the resummed form of the same ladder, so it is exact wherever the geometric series converges.

Returns
-------
numpy array of shape (4, Nf, Nf, Nb), complex, the irreducible vertices in channel order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gamma_from_full_vertex(F: 'np.ndarray', G: 'np.ndarray', beta: float, Nb: int) -> 'np.ndarray':
    """Recover the channel-irreducible vertices from the FULL vertices by inverting the ladder relations, rather than by summing them.

    Parameters
    ----------
    F : array of shape (4, Nf, Nf, Nb), complex, the full vertices in channel order
    G : array of shape (Nf,), complex, the propagator the bubbles are built from
    beta : float, inverse temperature
    Nb : int, number of bosonic frequencies

    Returns
    -------
    numpy array of shape (4, Nf, Nf, Nb), complex, the irreducible vertices in channel order

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




def _oracle_gamma_from_full_vertex(F: 'np.ndarray', G: 'np.ndarray', beta: float, Nb: int) -> 'np.ndarray':
    F = np.asarray(F, dtype=complex); G = np.asarray(G, dtype=complex)
    if F.ndim != 4 or F.shape[0] != 4:
        raise ValueError("the full vertex must carry the four channels first")
    if beta <= 0.0:
        raise ValueError("the inverse temperature must be positive")
    Nf = G.size
    if F.shape[1:] != (Nf, Nf, Nb):
        raise ValueError("the full vertex and the propagator disagree about the box")
    bub = _oracle_pair_bubbles(G, Nb)
    I = np.eye(Nf, dtype=complex)
    # batched over the bosonic index: (Nb, Nf, Nf) matrices in the two fermionic legs
    Dph = bub[0].T[:, :, None]*I            # diag(chi_ph) per bosonic frequency
    Dpp = bub[1].T[:, :, None]*I
    Gam = np.empty_like(F)
    for r in (0, 1):                        # Phi_r = (1/beta) Gamma_r chi_ph F_r
        Fr = np.moveaxis(F[r], 2, 0)
        Gam[r] = np.moveaxis(Fr @ np.linalg.inv(I + (Dph @ Fr)/beta), 0, 2)
    Fs = np.moveaxis(F[2], 2, 0)            # Phi_s = -(1/2beta) F_s chi_pp Gamma_s
    Gam[2] = np.moveaxis(np.linalg.solve(I - (Fs @ Dpp)/(2.0*beta), Fs), 0, 2)
    Ft = np.moveaxis(F[3], 2, 0)            # Phi_t = +(1/2beta) F_t chi_pp Gamma_t
    Gam[3] = np.moveaxis(np.linalg.solve(I + (Ft @ Dpp)/(2.0*beta), Ft), 0, 2)
    return Gam

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nrng = np.random.default_rng(11)\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 1.0 + 0.5j*np.sign(nu))\nF = (rng.normal(size=(4,Nf,Nf,Nb)) + 1j*rng.normal(size=(4,Nf,Nf,Nb)))*0.35\nF_g = F.copy()\nG_g = G.copy()",
         "call": "gamma_from_full_vertex(F, G, 1.0, 3)",
         "gold_call": "_oracle_gamma_from_full_vertex(F_g, G_g, 1.0, 3)"},
        # boundary: a vanishing full vertex leaves nothing to invert
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 1.5)\nF = np.zeros((4,Nf,Nf,Nb), dtype=complex)\nF_g = F.copy()\nG_g = G.copy()",
         "call": "gamma_from_full_vertex(F, G, 1.0, 3)",
         "gold_call": "_oracle_gamma_from_full_vertex(F_g, G_g, 1.0, 3)"},
        # edge: a strongly asymmetric vertex at a different temperature, where the resummation is far
        # from its first order, so a wrong channel sign or a wrong 1/beta factor shows up plainly
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nrng = np.random.default_rng(23)\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 0.8 + 0.5j*np.sign(nu))\nF = (rng.normal(size=(4,Nf,Nf,Nb)) + 1j*rng.normal(size=(4,Nf,Nf,Nb)))*0.9\nF_g = F.copy()\nG_g = G.copy()",
         "call": "gamma_from_full_vertex(F, G, 2.0, 3)",
         "gold_call": "_oracle_gamma_from_full_vertex(F_g, G_g, 2.0, 3)"},
    ]
