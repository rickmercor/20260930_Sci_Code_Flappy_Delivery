"""
Drive the state to a fixed point of the sweep by damped iteration, mixing a fraction p of the swept state into the old one at every step.

The update is state <- p * sweep(state) + (1-p) * state. Iterate until the largest absolute change between consecutive states, divided by the larger of one and the largest absolute entry of the new state, falls below 1e-13. Raise rather than return if the iteration produces a non-finite entry or fails to meet that criterion within twenty thousand steps. The fixed point reached from a vanishing vertex is the one continuously connected to the non-interacting limit.

Returns
-------
numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the converged state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relax_parquet(state0: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float, dmu: float, hyb: float, Nf: int, Nb: int, p: float) -> 'np.ndarray':
    """Drive the state to a fixed point of the sweep by damped iteration, mixing a fraction p of the swept state into the old one at every step.

    Parameters
    ----------
    state0 : array of shape (4*Nf*Nf*Nb + Nf,), complex, the starting state
    Lam : array of shape (4,), bare channel constants
    beta, U, dmu, hyb : float, model parameters
    Nf, Nb : int, box sizes
    p : float, damping in (0, 1]

    Returns
    -------
    numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the converged state

    Raises
    ------
    ValueError
        If p lies outside (0, 1], if the iterate becomes non-finite, or if the stopping criterion is not met within the step budget.

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




def _oracle_relax_parquet(state0: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float,
                          dmu: float, hyb: float, Nf: int, Nb: int, p: float) -> 'np.ndarray':
    if not (0.0 < p <= 1.0):
        raise ValueError("the damping parameter must lie in (0, 1]")
    v = np.asarray(state0, dtype=complex).copy()
    for _ in range(20000):
        nv = p*_oracle_parquet_map(v, Lam, beta, U, dmu, hyb, Nf, Nb) + (1.0 - p)*v
        if not np.all(np.isfinite(nv)):
            raise ValueError("the damped iteration diverged")
        err = np.max(np.abs(nv - v))/max(1.0, np.max(np.abs(nv)))
        v = nv
        if err < 1e-13:
            return v
    raise ValueError("the damped iteration did not converge within the iteration budget")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nnu = (2*np.arange(-2,2)+1)*np.pi\nG0 = 1.0/(1j*nu + 1.0 + 1.0)\nstate0 = np.concatenate([np.zeros(4*L, dtype=complex), G0])\nLam = np.array([2.0,-2.0,4.0,0.0], dtype=complex)\nstate0_g = state0.copy()\nLam_g = Lam.copy()",
         "call": "relax_parquet(state0, Lam, 1.0, 2.0, 1.0, 0.5, 4, 3, 0.3)", "gold_call": "_oracle_relax_parquet(state0_g, Lam_g, 1.0, 2.0, 1.0, 0.5, 4, 3, 0.3)"},
        # boundary: undamped iteration
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nnu = (2*np.arange(-2,2)+1)*np.pi\nG0 = 1.0/(1j*nu + 0.5 + 0.5)\nstate0 = np.concatenate([np.zeros(4*L, dtype=complex), G0])\nLam = np.array([1.0,-1.0,2.0,0.0], dtype=complex)\nstate0_g = state0.copy()\nLam_g = Lam.copy()",
         "call": "relax_parquet(state0, Lam, 1.0, 1.0, 0.5, 0.25, 4, 3, 1.0)", "gold_call": "_oracle_relax_parquet(state0_g, Lam_g, 1.0, 1.0, 0.5, 0.25, 4, 3, 1.0)"},
        # edge: started away from zero, which must reach the same fixed point
        {"setup": "import numpy as np\nNf, Nb = 4, 3\nL = Nf*Nf*Nb\nrng = np.random.default_rng(9)\nnu = (2*np.arange(-2,2)+1)*np.pi\nG0 = 1.0/(1j*nu + 1.0 + 1.0)\nstate0 = np.concatenate([(rng.normal(size=4*L)+0j)*0.2, G0])\nLam = np.array([2.0,-2.0,4.0,0.0], dtype=complex)\nstate0_g = state0.copy()\nLam_g = Lam.copy()",
         "call": "relax_parquet(state0, Lam, 1.0, 2.0, 1.0, 0.5, 4, 3, 0.4)", "gold_call": "_oracle_relax_parquet(state0_g, Lam_g, 1.0, 2.0, 1.0, 0.5, 4, 3, 0.4)"},
    ]
