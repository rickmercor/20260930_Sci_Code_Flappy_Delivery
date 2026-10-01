"""
Apply one full sweep of the self-consistent scheme to a state, returning the next state.

Fermionic Matsubara frequencies are nu = (2n+1) pi / beta and bosonic ones are omega = 2 m pi / beta, and every frequency is referred to below by its INTEGER label n or m rather than by its value. A box of Nf fermionic frequencies holds the integer labels n = -(Nf//2) up to Nf - Nf//2 - 1 inclusive, and a box of Nb bosonic frequencies holds the labels m over the same range with Nb in place of Nf; both are stored in increasing order of the label. With Nf even the fermionic frequencies are symmetric about zero, and with Nb odd the bosonic frequencies are symmetric about zero and include it. Any frequency argument whose label falls outside its box contributes zero. Vertices are indexed [channel, nu, nu', omega] with the channels ordered density, magnetic, singlet, triplet. The state is the four reducible vertices followed by the propagator, flattened in the order density, magnetic, singlet, triplet, propagator, so its length is 4 Nf^2 Nb + Nf and each vertex block is stored in C order over (nu, nu', omega).
One sweep: build the irreducible vertices from the state, build the two bubbles from its propagator, and then for each bosonic frequency form, as matrix products over the fermionic indices,
    new Phi_d = (1/beta) [Gamma_d * bubble_ph] (Phi_d + Gamma_d),
    new Phi_m = (1/beta) [Gamma_m * bubble_ph] (Phi_m + Gamma_m),
    new Phi_s = -(1/(2 beta)) [(Phi_s + Gamma_s) * bubble_pp] Gamma_s,
    new Phi_t = +(1/(2 beta)) [(Phi_t + Gamma_t) * bubble_pp] Gamma_t,
where * multiplies each matrix column by the bubble at that fermionic index. Note the two particle-particle channels put the bubble between the full and the irreducible vertex in the opposite order to the particle-hole ones, and carry the extra factor of one half with opposite signs. Finally build the self-energy from the INCOMING reducible vertices and the irreducible vertices already formed from them, that is from the same pair used above and not from the freshly swept vertices, and return the propagator 1/(1/G_0 - Sigma) where G_0 is the non-interacting propagator of this model at the same beta, U, dmu and hyb, carrying the same U/2 shift in its denominator.

Returns
-------
numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the next state in the same layout
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def parquet_map(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float, dmu: float, hyb: float, Nf: int, Nb: int) -> 'np.ndarray':
    """Apply one full sweep of the self-consistent scheme to a state, returning the next state.

    Parameters
    ----------
    state : array of shape (4*Nf*Nf*Nb + Nf,), complex
    Lam : array of shape (4,), bare channel constants
    beta, U, dmu, hyb : float, model parameters
    Nf, Nb : int, box sizes

    Returns
    -------
    numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the next state in the same layout

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




def _oracle_parquet_map(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float,
                        dmu: float, hyb: float, Nf: int, Nb: int) -> 'np.ndarray':
    state = np.asarray(state, dtype=complex)
    if state.size != 4*Nf*Nf*Nb + Nf:
        raise ValueError("the state must hold four vertex blocks and one Green's function")
    Phi, G = _unpack(state, Nf, Nb)
    Gam = _oracle_irreducible_vertices(Phi, Lam)
    bub = _oracle_pair_bubbles(G, Nb)
    new = np.empty_like(Phi)
    for k in range(Nb):
        new[0][:, :, k] = (Gam[0][:, :, k]*bub[0][None, :, k]) @ (Phi[0][:, :, k] + Gam[0][:, :, k])
        new[1][:, :, k] = (Gam[1][:, :, k]*bub[0][None, :, k]) @ (Phi[1][:, :, k] + Gam[1][:, :, k])
        new[2][:, :, k] = -0.5*((Phi[2][:, :, k] + Gam[2][:, :, k])*bub[1][None, :, k]) @ Gam[2][:, :, k]
        new[3][:, :, k] = +0.5*((Phi[3][:, :, k] + Gam[3][:, :, k])*bub[1][None, :, k]) @ Gam[3][:, :, k]
    new = new/beta
    Sig = _oracle_self_energy(Phi, Gam, G, beta, U)
    G0 = _oracle_bare_propagator(beta, Nf, U, dmu, hyb)
    return _pack(new, 1.0/(1.0/G0 - Sig))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 1.0 + 1.5)\nstate = np.concatenate([np.zeros(4*L, dtype=complex), G])\nLam = np.array([3.0,-3.0,6.0,0.0], dtype=complex)\nstate_g = state.copy()\nLam_g = Lam.copy()",
         "call": "parquet_map(state, Lam, 1.0, 3.0, 1.0, 0.5, 4, 3)", "gold_call": "_oracle_parquet_map(state_g, Lam_g, 1.0, 3.0, 1.0, 0.5, 4, 3)"},
        # boundary: no interaction, so the vertex blocks must stay zero
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 0.5)\nstate = np.concatenate([np.zeros(4*L, dtype=complex), G])\nLam = np.zeros(4, dtype=complex)\nstate_g = state.copy()\nLam_g = Lam.copy()",
         "call": "parquet_map(state, Lam, 1.0, 0.0, 0.5, 0.0, 4, 3)", "gold_call": "_oracle_parquet_map(state_g, Lam_g, 1.0, 0.0, 0.5, 0.0, 4, 3)"},
        # edge: a fully populated state, so every frequency shift is exercised
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nrng = np.random.default_rng(5)\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 2.0)\nstate = np.concatenate([(rng.normal(size=4*L)+1j*rng.normal(size=4*L))*0.25, G])\nLam = np.array([2.0,-2.0,4.0,0.0], dtype=complex)\nstate_g = state.copy()\nLam_g = Lam.copy()",
         "call": "parquet_map(state, Lam, 1.0, 2.0, 0.0, 0.75, 4, 3)", "gold_call": "_oracle_parquet_map(state_g, Lam_g, 1.0, 2.0, 0.0, 0.75, 4, 3)"},
    ]
