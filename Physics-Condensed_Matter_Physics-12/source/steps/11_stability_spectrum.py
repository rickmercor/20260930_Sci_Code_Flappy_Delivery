"""
Give the eigenvalues of the matrix that controls local convergence of the damped iteration for EITHER sweep, namely the identity minus the derivative of the named sweep with respect to the state.

The scheme selects which sweep is differentiated: 'reducible' is the sweep that carries the reducible vertices, 'full' the one that carries the full vertices, and any other value is an error. The state must be in the layout that sweep expects. Call the sweep's derivative at the state D, a square matrix of the state's length, and form Pi = 1 - D. The damped iteration with damping p has local multipliers 1 - p lambda, where lambda runs over the eigenvalues of Pi, so Pi is what decides stability. The sweep is holomorphic in the state, so take D by a central difference with a step of 1e-5 applied to each component in turn, with no real part taken anywhere. A central difference divides by the step, so a smaller one is not a better one here: at 1e-6 the rounding noise of the sweep is amplified enough to move the final answer in its ninth digit, while the truncation error at 1e-5 is still below that. Return the raw eigenvalues in whatever order the eigensolver produces.

Returns
-------
numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the eigenvalues of Pi
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stability_spectrum(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float, dmu: float, hyb: float, Nf: int, Nb: int, scheme: str) -> 'np.ndarray':
    """Give the eigenvalues of the matrix that controls local convergence of the damped iteration for EITHER sweep, namely the identity minus the derivative of the named sweep with respect to the state.

    Parameters
    ----------
    state : array of shape (4*Nf*Nf*Nb + Nf,), complex, in the layout the named sweep expects
    Lam : array of shape (4,), bare channel constants
    beta, U, dmu, hyb : float, model parameters
    Nf, Nb : int, box sizes
    scheme : str, either 'reducible' or 'full'

    Returns
    -------
    numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the eigenvalues of Pi

    Raises
    ------
    ValueError
        If the state length does not match the box, or if scheme is neither 'reducible' nor 'full'.

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




def _oracle_stability_spectrum(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float,
                               dmu: float, hyb: float, Nf: int, Nb: int,
                               scheme: str) -> 'np.ndarray':
    state = np.asarray(state, dtype=complex)
    n = state.size
    if n != 4*Nf*Nf*Nb + Nf:
        raise ValueError("the state has the wrong length for this box")
    if scheme == "reducible":
        sweep = _oracle_parquet_map
    elif scheme == "full":
        sweep = _oracle_strong_coupling_map
    else:
        raise ValueError("the scheme must be either 'reducible' or 'full'")
    h = 1e-5
    J = np.empty((n, n), dtype=complex)
    for j in range(n):
        e = np.zeros(n, dtype=complex); e[j] = h
        J[:, j] = (sweep(state + e, Lam, beta, U, dmu, hyb, Nf, Nb)
                   - sweep(state - e, Lam, beta, U, dmu, hyb, Nf, Nb))/(2.0*h)
    return np.linalg.eigvals(np.eye(n) - J)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'Nf, Nb = 4, 3\n'
               'L = Nf*Nf*Nb\n'
               'nu = (2*np.arange(-2,2)+1)*np.pi\n'
               'G0 = 1.0/(1j*nu + 1.0 + 1.0 + 0.5j*np.sign(nu))\n'
               'state0 = np.concatenate([np.zeros(4*L, dtype=complex), G0])\n'
               'Lam = np.array([2.0,-2.0,4.0,0.0], dtype=complex)\n'
               'state0_g = state0.copy()\n'
               'Lam_g = Lam.copy()\n'
               'st = relax_parquet(state0, Lam.copy(), 1.0, 2.0, 1.0, 0.5, 4, 3, 0.3)\n'
               'st_g = _oracle_relax_parquet(state0_g, Lam_g.copy(), 1.0, 2.0, 1.0, 0.5, 4, 3, 0.3)\n'
               'def ev(a):\n'
               '    e = np.asarray(a)\n'
               '    assert e.shape == (4*Nf*Nf*Nb + Nf,), "Expected a one-dimensional eigenvalue vector"\n'
               '    return np.array([e.sum(), (e**2).sum(), (e**3).sum(), e.real.min(), '
               'np.min(2*np.abs(e.real)/np.abs(e)**2)])\n',
      'call': "ev(stability_spectrum(st, Lam, 1.0, 2.0, 1.0, 0.5, 4, 3, 'reducible'))",
      'gold_call': "ev(_oracle_stability_spectrum(st_g, Lam_g, 1.0, 2.0, 1.0, 0.5, 4, 3, 'reducible'))",
      'tol': 1e-07},
     {'setup': 'import numpy as np\n'
               'Nf, Nb = 4, 3\n'
               'L = Nf*Nf*Nb\n'
               'nu = (2*np.arange(-2,2)+1)*np.pi\n'
               'G0 = 1.0/(1j*nu + 1.0 + 1.0 + 0.5j*np.sign(nu))\n'
               'state0 = np.concatenate([np.zeros(4*L, dtype=complex), G0])\n'
               'Lam = np.array([2.0,-2.0,4.0,0.0], dtype=complex)\n'
               'state0_g = state0.copy()\n'
               'Lam_g = Lam.copy()\n'
               'st = relax_parquet(state0, Lam.copy(), 1.0, 2.0, 1.0, 0.5, 4, 3, 0.3)\n'
               'st_g = _oracle_relax_parquet(state0_g, Lam_g.copy(), 1.0, 2.0, 1.0, 0.5, 4, 3, 0.3)\n'
               'Phi = st[:4*L].reshape(4,Nf,Nf,Nb)\n'
               'Phi_g = st_g[:4*L].reshape(4,Nf,Nf,Nb)\n'
               'Gm = irreducible_vertices(Phi.copy(), Lam.copy())\n'
               'Gm_g = _oracle_irreducible_vertices(Phi_g.copy(), Lam_g.copy())\n'
               'fst = np.concatenate([(Phi+Gm).ravel(), st[4*L:]])\n'
               'fst_g = np.concatenate([(Phi_g+Gm_g).ravel(), st_g[4*L:]])\n'
               'def ev(a):\n'
               '    e = np.asarray(a)\n'
               '    assert e.shape == (4*Nf*Nf*Nb + Nf,), "Expected a one-dimensional eigenvalue vector"\n'
               '    return np.array([e.sum(), (e**2).sum(), (e**3).sum(), e.real.min(), '
               'np.min(2*np.abs(e.real)/np.abs(e)**2)])\n',
      'call': "ev(stability_spectrum(fst, Lam, 1.0, 2.0, 1.0, 0.5, 4, 3, 'full'))",
      'gold_call': "ev(_oracle_stability_spectrum(fst_g, Lam_g, 1.0, 2.0, 1.0, 0.5, 4, 3, 'full'))",
      'tol': 1e-07},
     {'setup': 'import numpy as np\n'
               'Nf, Nb = 4, 3\n'
               'L = Nf*Nf*Nb\n'
               'nu = (2*np.arange(-2,2)+1)*np.pi\n'
               'G0 = 1.0/(1j*nu + 0.5)\n'
               'state0 = np.concatenate([np.zeros(4*L, dtype=complex), G0])\n'
               'Lam = np.zeros(4, dtype=complex)\n'
               'def ev(a):\n'
               '    e = np.asarray(a)\n'
               '    assert e.shape == (4*Nf*Nf*Nb + Nf,), "Expected a one-dimensional eigenvalue vector"\n'
               '    return np.array([e.sum(), (e**2).sum(), (e**3).sum(), e.real.min(), '
               'np.min(2*np.abs(e.real)/np.abs(e)**2)])\n'
               'state0_g = state0.copy()\n'
               'Lam_g = Lam.copy()',
      'call': "ev(stability_spectrum(state0, Lam, 1.0, 0.0, 0.5, 0.0, 4, 3, 'reducible'))",
      'gold_call': "ev(_oracle_stability_spectrum(state0_g, Lam_g, 1.0, 0.0, 0.5, 0.0, 4, 3, 'reducible'))"}]
