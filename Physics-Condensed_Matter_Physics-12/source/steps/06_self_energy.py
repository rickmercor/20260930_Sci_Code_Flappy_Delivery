"""
Give the self-energy from the equation of motion: a static Hartree term plus a dynamic term built from whichever combination of the density and magnetic full vertices survives at leading order in the interaction.

Fermionic Matsubara frequencies are nu = (2n+1) pi / beta and bosonic ones are omega = 2 m pi / beta, and every frequency is referred to below by its INTEGER label n or m rather than by its value. A box of Nf fermionic frequencies holds the integer labels n = -(Nf//2) up to Nf - Nf//2 - 1 inclusive, and a box of Nb bosonic frequencies holds the labels m over the same range with Nb in place of Nf; both are stored in increasing order of the label. With Nf even the fermionic frequencies are symmetric about zero, and with Nb odd the bosonic frequencies are symmetric about zero and include it. Any frequency argument whose label falls outside its box contributes zero. Vertices are indexed [channel, nu, nu', omega] with the channels ordered density, magnetic, singlet, triplet. The full vertex in a channel is the sum of its reducible and irreducible parts, F_r = Phi_r + Gamma_r. The self-energy is
    Sigma(nu) = U n/2 - (U / (2 beta^2)) sum_{nu1, omega} C(nu, nu1, omega) G(nu1) G(nu1+omega) G(nu+omega),
where C is one of the two combinations F_d + F_m or F_d - F_m of the density and magnetic full vertices. Exactly one of the two is correct and the weak-coupling limit decides which: the dynamic part of this self-energy must be second order in the interaction, and one of the two combinations cancels identically at leading order because of the relation between the bare channel constants. Work out which one survives and use it.
The density is n/2 = (1/beta) sum_nu G(nu) e^{i nu 0^+}. The convergence factor matters: G falls off as 1/(i nu), and truncating the sum to the box without that contact contribution does not give zero, it gives the wrong number. Subtract 1/(i nu) frequency by frequency inside the sum, divide by beta, and then add one half to the result, so that n/2 tends to one half in the non-interacting particle-hole symmetric limit. The static term is U times that n/2, with no further subtraction.

Returns
-------
numpy array of shape (Nf,), complex, the self-energy on the fermionic box
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_energy(Phi: 'np.ndarray', Gam: 'np.ndarray', G: 'np.ndarray', beta: float, U: float) -> 'np.ndarray':
    """Give the self-energy from the equation of motion: a static Hartree term plus a dynamic term built from whichever combination of the density and magnetic full vertices survives at leading order in the interaction.

    Parameters
    ----------
    Phi : array of shape (4, Nf, Nf, Nb), complex, reducible vertices
    Gam : array of shape (4, Nf, Nf, Nb), complex, irreducible vertices
    G : array of shape (Nf,), complex, the propagator
    beta : float, inverse temperature
    U : float, on-site interaction

    Returns
    -------
    numpy array of shape (Nf,), complex, the self-energy on the fermionic box

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




def _oracle_self_energy(Phi: 'np.ndarray', Gam: 'np.ndarray', G: 'np.ndarray',
                        beta: float, U: float) -> 'np.ndarray':
    Phi = np.asarray(Phi, dtype=complex); Gam = np.asarray(Gam, dtype=complex)
    G = np.asarray(G, dtype=complex)
    if Phi.shape != Gam.shape:
        raise ValueError("Phi and Gamma must have the same shape")
    if beta <= 0.0:
        raise ValueError("the inverse temperature must be positive")
    Nf, Nb = G.size, Phi.shape[3]
    nu = (2.0*_nf_int(Nf) + 1.0)*np.pi/beta
    # (1/beta) sum_nu G(nu) e^{i nu 0^+} = n/2. G falls off as 1/(i nu); that tail is subtracted
    # frequency by frequency and its exact sum, one half, is added back. On a box symmetric about
    # zero the subtracted tail itself sums to zero, so it is the added half that carries the
    # contact contribution -- dropping it leaves only the regular part, 0.0704 at the benchmark, and
    # an eightfold-too-small static self-energy (0.351877 in place of 2.851877).
    dens = np.sum(G - 1.0/(1j*nu))/beta + 0.5
    bub = _oracle_pair_bubbles(G, Nb)[0]
    nf, nb = _nf_int(Nf), _nb_int(Nb)
    acc = np.zeros(Nf, dtype=complex)
    rows = np.arange(Nf)
    for k, m in enumerate(nb):
        j = _fidx(Nf, nf + m)
        ok = j >= 0
        gshift = np.zeros(Nf, dtype=complex)
        gshift[ok] = G[j[ok]]
        F = Phi[0][:, :, k] + Gam[0][:, :, k] - Phi[1][:, :, k] - Gam[1][:, :, k]
        acc += (F @ bub[:, k])*gshift
    return U*dens*np.ones(Nf, dtype=complex) - (U/(2.0*beta**2))*acc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nimport numpy as _n\nNf, Nb = 6, 5\nnu = (2*np.arange(-3,3)+1)*np.pi\nG = 1.0/(1j*nu + 1.0 + 2.5)\nrng = np.random.default_rng(7)\nPhi = (rng.normal(size=(4,Nf,Nf,Nb)) + 1j*rng.normal(size=(4,Nf,Nf,Nb)))*0.2\nGam = (rng.normal(size=(4,Nf,Nf,Nb)) + 1j*rng.normal(size=(4,Nf,Nf,Nb)))*0.2\nPhi_g = Phi.copy()\nGam_g = Gam.copy()\nG_g = G.copy()",
         "call": "self_energy(Phi, Gam, G, 1.0, 5.0)", "gold_call": "_oracle_self_energy(Phi_g, Gam_g, G_g, 1.0, 5.0)"},
        # boundary: no vertex at all, leaving only the Hartree term
        {"setup": "import numpy as np\nNf, Nb = 6, 5\nnu = (2*np.arange(-3,3)+1)*np.pi\nG = 1.0/(1j*nu + 2.5)\nPhi = np.zeros((4,Nf,Nf,Nb), dtype=complex)\nGam = np.zeros((4,Nf,Nf,Nb), dtype=complex)\nPhi_g = Phi.copy()\nGam_g = Gam.copy()\nG_g = G.copy()",
         "call": "self_energy(Phi, Gam, G, 1.0, 5.0)", "gold_call": "_oracle_self_energy(Phi_g, Gam_g, G_g, 1.0, 5.0)"},
        # edge: density and magnetic set equal, so the dynamic term cancels identically
        {"setup": "import numpy as np\nNf, Nb = 5, 5\nnu = (2*np.arange(-2,3)+1)*np.pi\nG = 1.0/(1j*nu + 1.5)\nrng = np.random.default_rng(11)\nblk = (rng.normal(size=(Nf,Nf,Nb)) + 1j*rng.normal(size=(Nf,Nf,Nb)))*0.4\nPhi = np.stack([blk, blk, 0*blk, 0*blk])\nGam = np.stack([blk, blk, 0*blk, 0*blk])\nPhi_g = Phi.copy()\nGam_g = Gam.copy()\nG_g = G.copy()",
         "call": "self_energy(Phi, Gam, G, 1.0, 3.0)", "gold_call": "_oracle_self_energy(Phi_g, Gam_g, G_g, 1.0, 3.0)"},
    ]
