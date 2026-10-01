"""
Return the converged orbitals of the closed-shell mean field F = h + J - alpha K of the task's chain, where the Hartree operator J is diagonal with J_ii = sum_j W_ij n_j (n_j the site occupations of the nelec/2 doubly occupied orbitals) and the exchange operator is K_ij = W_ij P_ij / 2 with P = 2 C_occ C_occ^T the spatial density matrix; iterate to self-consistency (density matrix converged to 1e-12) and return the canonical orbitals of the converged operator in increasing energy order.

The source's density matrices are built on top of a generalized Kohn-Sham reference whose exchange-correlation potential may contain a fraction of exact exchange; for the task's model the reference is the mean field that keeps a fraction alpha of the exchange operator (alpha = 1 is Hartree-Fock).

Returns
-------
ndarray of float64 with shape (N, N), orthonormal columns in increasing orbital-energy order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scaled_exchange_orbitals(h: "np.ndarray", W: "np.ndarray", nelec: int, alpha: float) -> "np.ndarray":
    '''Orbitals of the closed-shell scaled-exchange mean field of the task's chain.

    Parameters
    ----------
    h : np.ndarray
        One-body matrix of step 01, shape (N, N).
    W : np.ndarray
        Interaction matrix of step 01, shape (N, N).
    nelec : int
        Number of electrons (even, at most 2N).
    alpha : float
        Fraction of the exchange operator kept in the mean field, 0 <= alpha <= 1.

    Returns
    -------
    C : np.ndarray
        Orthonormal orbital coefficients of shape (N, N), one orbital per column, ordered by
        increasing orbital energy; each column is scaled so that its component of largest magnitude
        is positive.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, h or W is not symmetric, nelec is odd or exceeds 2N, alpha
        is negative, the iteration does not converge, or the reference is degenerate at the Fermi
        level.
    '''
    return [[0.0]]

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


def _fix_signs(C):
    """make the largest-magnitude component of every column positive (removes the eigenvector sign freedom)"""
    C = np.array(C, dtype=np.float64, copy=True)
    for k in range(C.shape[1]):
        col = C[:, k]
        if col[np.argmax(np.abs(col))] < 0:
            C[:, k] = -col
    return C


def _oracle_scaled_exchange_orbitals(h: "np.ndarray", W: "np.ndarray", nelec: int, alpha: float) -> "np.ndarray":
    h = _check_array(h, "h", ndim=2); N = h.shape[0]
    W = _check_array(W, "W", shape=(N, N))
    nelec = _check_int(nelec, "nelec", 2)
    if nelec % 2 or nelec > 2 * N:
        raise ValueError("nelec must be even and at most 2N")
    alpha = _check_scalar(alpha, "alpha", nonneg=True)
    if not (np.allclose(h, h.T) and np.allclose(W, W.T)):
        raise ValueError("h and W must be symmetric")
    nocc = nelec // 2
    e, C = eigh(h)
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    damp = 0.3
    for it in range(20000):
        J = np.diag(W @ np.diag(P)); K = 0.5 * W * P
        e, C = eigh(h + J - alpha * K)
        Pn = 2.0 * C[:, :nocc] @ C[:, :nocc].T
        if np.max(np.abs(Pn - P)) < 1e-13:
            P = Pn
            break
        P = (1.0 - damp) * Pn + damp * P
    else:
        raise ValueError("mean field did not converge")
    J = np.diag(W @ np.diag(P)); K = 0.5 * W * P
    e, C = eigh(h + J - alpha * K)                          # canonical orbitals of the converged operator
    if nocc < N and e[nocc] - e[nocc - 1] < 1e-8:
        raise ValueError("closed-shell reference is degenerate at the Fermi level")
    return _fix_signs(C)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0]))\nh, W, nelec, alpha = hW[0], hW[1], 6, 0.75",
            "call": "scaled_exchange_orbitals(h.copy(), W.copy(), nelec, alpha)",
            "gold_call": "_oracle_scaled_exchange_orbitals(h.copy(), W.copy(), nelec, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(4, 0.8, 2.0, 0.7, np.array([-0.6, 0.0, 0.0, 0.3]))\nh, W, nelec, alpha = hW[0], hW[1], 4, 1.0",
            "call": "scaled_exchange_orbitals(h.copy(), W.copy(), nelec, alpha)",
            "gold_call": "_oracle_scaled_exchange_orbitals(h.copy(), W.copy(), nelec, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(5, 1.2, 1.0, 0.5, np.array([0.5, -0.2, 0.0, 0.1, -0.4]))\nh, W, nelec, alpha = hW[0], hW[1], 4, 0.5",
            "call": "scaled_exchange_orbitals(h.copy(), W.copy(), nelec, alpha)",
            "gold_call": "_oracle_scaled_exchange_orbitals(h.copy(), W.copy(), nelec, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nh = np.array([[0.0, -1.0], [-1.0, 0.0]])\nW = np.eye(2)\nnelec, alpha = 4, 0.75",
            "call": "scaled_exchange_orbitals(h.copy(), W.copy(), nelec, alpha)",
            "gold_call": "_oracle_scaled_exchange_orbitals(h.copy(), W.copy(), nelec, alpha)",
            "tol": 1e-09,
        },
    ]
