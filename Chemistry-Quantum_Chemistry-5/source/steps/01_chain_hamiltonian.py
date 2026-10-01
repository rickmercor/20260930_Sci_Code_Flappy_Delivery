"""
Return the one-body matrix h and the density-density interaction matrix W of the task's open chain of N sites: h has the site energies eps on its diagonal and -t between nearest neighbours; W has U on its diagonal and V between nearest neighbours, and the interaction energy of two site charges n_i, n_j is W_ij n_i n_j.

The task's system is an open extended-Hubbard chain: electrons hop between neighbouring sites and interact through on-site and nearest-neighbour density-density terms; the site energies make the chain asymmetric, so its ground state carries a dipole.

Returns
-------
ndarray of float64 with shape (2, N, N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def chain_hamiltonian(N: int, t: float, U: float, V: float, eps: "np.ndarray") -> "np.ndarray":
    '''One-body and interaction matrices of the task's open chain model.

    Parameters
    ----------
    N : int
        Number of sites (N >= 2).
    t : float
        Nearest-neighbour hopping amplitude.
    U : float
        On-site interaction.
    V : float
        Nearest-neighbour interaction.
    eps : np.ndarray
        One-dimensional array of the N site energies.

    Returns
    -------
    hW : np.ndarray
        Array of shape (2, N, N): hW[0] is the one-body matrix h (site energies on the diagonal, -t
        between neighbouring sites), hW[1] is the symmetric interaction matrix W (U on the diagonal,
        V between neighbouring sites, zero elsewhere).

    Raises
    ------
    ValueError
        If N < 2, eps does not have N finite entries, t is zero, or t, U, V are not finite.
    '''
    return [[[0.0]], [[0.0]]]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.linalg import eigh, solve


def _check_scalar(x, name, positive=False, nonneg=False):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("%s must be finite" % name)
    if positive and x <= 0:
        raise ValueError("%s must be positive" % name)
    if nonneg and x < 0:
        raise ValueError("%s must be non-negative" % name)
    return x

def _check_int(n, name, minimum=0):
    if isinstance(n, bool) or int(n) != n or int(n) < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)

def _check_array(a, name, ndim=None, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    return a


def _oracle_chain_hamiltonian(N: int, t: float, U: float, V: float, eps: "np.ndarray") -> "np.ndarray":
    N = _check_int(N, "N", 2)
    t = _check_scalar(t, "t"); U = _check_scalar(U, "U"); V = _check_scalar(V, "V")
    eps = _check_array(eps, "eps", shape=(N,))
    if t == 0.0:
        raise ValueError("the hopping amplitude t must be non-zero (the sites would not be connected)")
    h = np.diag(eps)
    idx = np.arange(N - 1)
    h[idx, idx + 1] = -t
    h[idx + 1, idx] = -t
    W = U * np.eye(N)
    W[idx, idx + 1] = V
    W[idx + 1, idx] = V
    return np.stack([h, W])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nN, t, U, V, eps = 6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0])",
            "call": "chain_hamiltonian(N, t, U, V, eps.copy())",
            "gold_call": "_oracle_chain_hamiltonian(N, t, U, V, eps.copy())",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nN, t, U, V, eps = 4, 0.8, 2.0, 0.7, np.array([-0.6, 0.0, 0.0, 0.3])",
            "call": "chain_hamiltonian(N, t, U, V, eps.copy())",
            "gold_call": "_oracle_chain_hamiltonian(N, t, U, V, eps.copy())",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nN, t, U, V, eps = 5, 1.2, 1.0, 0.0, np.array([0.5, -0.2, 0.0, 0.1, -0.4])",
            "call": "chain_hamiltonian(N, t, U, V, eps.copy())",
            "gold_call": "_oracle_chain_hamiltonian(N, t, U, V, eps.copy())",
            "tol": 1e-12,
        },
    ]
