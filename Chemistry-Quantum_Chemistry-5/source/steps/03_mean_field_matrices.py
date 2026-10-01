"""
Return, in the orbital basis defined by C, the mean-field operator of step 02 (whose diagonal holds the orbital energies) and the full exchange operator K of step 02 built from the same nelec/2 doubly occupied orbitals; the exchange operator is needed separately because the source's static self-energy compares the exact-exchange operator with the exchange-correlation potential of the reference.

The source formulates everything in the orthonormal molecular-orbital basis of the reference; the reference's one-body operator is diagonal there, while the exact-exchange operator is not.

Returns
-------
ndarray of float64 with shape (2, N, N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mean_field_matrices(h: "np.ndarray", W: "np.ndarray", C: "np.ndarray", nelec: int, alpha: float) -> "np.ndarray":
    '''Mean-field operator and exchange operator of step 02 in the orbital basis.

    Parameters
    ----------
    h : np.ndarray
        One-body matrix of step 01, shape (N, N).
    W : np.ndarray
        Interaction matrix of step 01, shape (N, N).
    C : np.ndarray
        Orthonormal orbitals of step 02, shape (N, N).
    nelec : int
        Number of electrons (even, at most 2N).
    alpha : float
        Exchange fraction of the mean field, 0 <= alpha <= 1.

    Returns
    -------
    FK : np.ndarray
        Array of shape (2, N, N): FK[0] = C^T F C, the mean-field operator F = h + J - alpha K in
        the orbital basis (diagonal at convergence, its diagonal being the orbital energies); FK[1]
        = C^T K C, the full (unscaled) exchange operator in the orbital basis.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, C is not orthonormal, nelec is odd or exceeds 2N, or alpha
        is negative.
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

def _mean_field_pieces(h, W, C, nocc, alpha):
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T                 # spatial density matrix (both spins)
    J = np.diag(W @ np.diag(P))                            # Hartree: J_ii = sum_j W_ij n_j
    K = 0.5 * W * P                                        # exchange: K_ij = (1/2) W_ij P_ij
    F = h + J - alpha * K
    return P, J, K, F


def _oracle_mean_field_matrices(h: "np.ndarray", W: "np.ndarray", C: "np.ndarray", nelec: int, alpha: float) -> "np.ndarray":
    h = _check_array(h, "h", ndim=2); N = h.shape[0]
    W = _check_array(W, "W", shape=(N, N)); C = _check_array(C, "C", shape=(N, N))
    nelec = _check_int(nelec, "nelec", 2)
    if nelec % 2 or nelec > 2 * N:
        raise ValueError("nelec must be even and at most 2N")
    alpha = _check_scalar(alpha, "alpha", nonneg=True)
    if not np.allclose(C.T @ C, np.eye(N), atol=1e-8):
        raise ValueError("C must be orthonormal")
    P, J, K, F = _mean_field_pieces(h, W, C, nelec // 2, alpha)
    return np.stack([C.T @ F @ C, C.T @ K @ C])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0]))\nh, W, nelec, alpha = hW[0], hW[1], 6, 0.75\nC = _oracle_scaled_exchange_orbitals(h, W, nelec, alpha)",
            "call": "mean_field_matrices(h.copy(), W.copy(), C.copy(), nelec, alpha)",
            "gold_call": "_oracle_mean_field_matrices(h.copy(), W.copy(), C.copy(), nelec, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(4, 0.8, 2.0, 0.7, np.array([-0.6, 0.0, 0.0, 0.3]))\nh, W, nelec, alpha = hW[0], hW[1], 4, 1.0\nC = _oracle_scaled_exchange_orbitals(h, W, nelec, alpha)",
            "call": "mean_field_matrices(h.copy(), W.copy(), C.copy(), nelec, alpha)",
            "gold_call": "_oracle_mean_field_matrices(h.copy(), W.copy(), C.copy(), nelec, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(5, 1.2, 1.0, 0.5, np.array([0.5, -0.2, 0.0, 0.1, -0.4]))\nh, W, nelec, alpha = hW[0], hW[1], 4, 0.5\nC = _oracle_scaled_exchange_orbitals(h, W, nelec, alpha)",
            "call": "mean_field_matrices(h.copy(), W.copy(), C.copy(), nelec, alpha)",
            "gold_call": "_oracle_mean_field_matrices(h.copy(), W.copy(), C.copy(), nelec, alpha)",
            "tol": 1e-09,
        },
    ]
