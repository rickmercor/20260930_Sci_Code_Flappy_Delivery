"""
Perform one sweep of the alternative iteration that carries the FULL vertices instead of the reducible ones, in the same flattened layout.

The state is packed exactly as before but holds F_d, F_m, F_s, F_t in place of the reducible vertices, followed by the propagator. This sweep carries the full vertices: it takes the state's F to a fresh F through the same channel mixing the other sweep uses, with the irreducible vertices recovered from F by inverting the ladder relations instead of summed from reducible parts, and the parquet decomposition then relating the fresh irreducible vertices back to a full vertex. Compose it from the steps already built rather than rewriting either of them. The propagator is rebuilt from the equation of motion and the Dyson equation. One point there is worth stating outright, because a converged state cannot reveal it: the equation of motion is a functional of whatever the sweep iterates, and this sweep iterates the full vertex, so the full vertex entering it is the state's OWN F -- the one the ladders were just inverted from, equivalently Phi + Gamma[F] -- and not Gamma_fresh + Phi. The two agree exactly at a fixed point and differ away from one, so the choice is invisible to a residual check and shows up only in the derivative, which is the quantity being asked for. Note what this map shares with the other one and what it does not: a state is a fixed point of one exactly when it is a fixed point of the other, because both reduce to the same condition that the mixing reproduce the irreducible vertices; but the two linearise differently, so their spectra and their damping requirements are not the same.

Returns
-------
numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the next state in the same layout
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def strong_coupling_map(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float, dmu: float, hyb: float, Nf: int, Nb: int) -> 'np.ndarray':
    """Perform one sweep of the alternative iteration that carries the FULL vertices instead of the reducible ones, in the same flattened layout.

    Parameters
    ----------
    state : array of shape (4*Nf*Nf*Nb + Nf,), complex, full vertices then propagator
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




def _oracle_strong_coupling_map(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float,
                                dmu: float, hyb: float, Nf: int, Nb: int) -> 'np.ndarray':
    state = np.asarray(state, dtype=complex)
    if state.size != 4*Nf*Nf*Nb + Nf:
        raise ValueError("the state must hold four vertex blocks and one Green's function")
    F, G = _unpack(state, Nf, Nb)
    Gam = _oracle_gamma_from_full_vertex(F, G, beta, Nb)
    Phi = F - Gam
    Gtil = _oracle_irreducible_vertices(Phi, Lam)
    # The equation of motion is a functional of whatever this sweep iterates, which here is the FULL
    # vertex. Phi + Gam is the state's own F; Phi + Gtil is not, away from the fixed point. The two
    # coincide at the fixed point, so this choice is invisible to a residual check and visible only
    # in the Jacobian -- which is precisely the quantity being asked for.
    Sig = _oracle_self_energy(Phi, Gam, G, beta, U)
    G0 = _oracle_bare_propagator(beta, Nf, U, dmu, hyb)
    return _pack(Gtil + Phi, 1.0/(1.0/G0 - Sig))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nrng = np.random.default_rng(7)\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 1.0 + 0.5j*np.sign(nu))\nstate = np.concatenate([(rng.normal(size=4*L)+1j*rng.normal(size=4*L))*0.3, G])\nLam = np.array([2.0,-2.0,4.0,0.0], dtype=complex)\nstate_g = state.copy()\nLam_g = Lam.copy()",
         "call": "strong_coupling_map(state, Lam, 1.0, 2.0, 1.0, 0.5, 4, 3)",
         "gold_call": "_oracle_strong_coupling_map(state_g, Lam_g, 1.0, 2.0, 1.0, 0.5, 4, 3)"},
        # boundary: no interaction and no bath, where the mixing has nothing to feed on
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 0.5)\nstate = np.concatenate([np.zeros(4*L, dtype=complex), G])\nLam = np.zeros(4, dtype=complex)\nstate_g = state.copy()\nLam_g = Lam.copy()",
         "call": "strong_coupling_map(state, Lam, 1.0, 0.0, 0.5, 0.0, 4, 3)",
         "gold_call": "_oracle_strong_coupling_map(state_g, Lam_g, 1.0, 0.0, 0.5, 0.0, 4, 3)"},
        # edge: a larger full vertex, where the inverted ladder differs most from the summed one
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nrng = np.random.default_rng(29)\nnu = (2*np.arange(-2,2)+1)*np.pi\nG = 1.0/(1j*nu + 0.4 + 0.75j*np.sign(nu))\nstate = np.concatenate([(rng.normal(size=4*L)+1j*rng.normal(size=4*L))*0.55, G])\nLam = np.array([3.0,-3.0,6.0,0.0], dtype=complex)\nstate_g = state.copy()\nLam_g = Lam.copy()",
         "call": "strong_coupling_map(state, Lam, 1.0, 3.0, 0.0, 0.75, 4, 3)",
         "gold_call": "_oracle_strong_coupling_map(state_g, Lam_g, 1.0, 3.0, 0.0, 0.75, 4, 3)"},
    ]
