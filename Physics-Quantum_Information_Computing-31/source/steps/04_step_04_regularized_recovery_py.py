"""
Run regularized factored sensing from an acquisition estimate.

The squared factor norm is the nuclear norm of the PSD recovered matrix. Trace normalization would change the objective and its stationary points.

Returns
-------
(d, d) complex array, the recovered PSD matrix after the prescribed updates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regularized_recovery(rho_init: "np.ndarray", r: int, A_ops: "np.ndarray", y: "np.ndarray", n_iters: int, lam: float) -> "np.ndarray":
    """Recover an unconstrained PSD matrix from a spectral factor warm start.

    Parameters
    ----------
    rho_init : np.ndarray
        Hermitian PSD initial matrix of shape (d,d). If truncation discards
        a positive eigenvalue, assume a strict gap at the retained boundary.
    r : int
        Number of largest eigenpairs in the initial factor, 1 <= r <= d.
        Clip numerical negative eigenvalues to zero before square roots.
    A_ops : np.ndarray
        Hermitian operators of shape (m,d,d).
    y : np.ndarray
        Observations of shape (m,), matching A_ops.
    n_iters : int
        Nonnegative number of regularized_backtrack_step updates.
    lam : float
        Nonnegative regularization coefficient for every update.

    Returns
    -------
    result : np.ndarray
        Recovered matrix F F-dagger, shape (d,d), after exactly n_iters updates.
        Eigenvector phases and unitary rotations within a fully retained
        eigenspace are immaterial.

    Raises
    ------
    ValueError
        If n_iters or lam is negative, r is outside [1,d], or measurement lengths differ.
    FloatingPointError
        If an update's line search underflows.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_regularized_recovery(rho_init: "np.ndarray", r: int, A_ops: "np.ndarray", y: "np.ndarray", n_iters: int, lam: float) -> "np.ndarray":
    if n_iters<0 or not 1<=r<=len(rho_init) or lam<0 or len(A_ops)!=len(y):
        raise ValueError('invalid rank, iterations, regularization, or measurement lengths')
    values,vectors=np.linalg.eigh(rho_init)
    indices=np.argsort(values)[::-1][:r]
    F=vectors[:,indices]*np.sqrt(np.maximum(values[indices],0))
    for _ in range(n_iters):
        F=_oracle_regularized_backtrack_step(F,A_ops,y,lam)
    return F@F.conj().T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return separate normal, boundary, and edge comparison cases."""
    return [{'setup': 'import numpy as np\n'
               'A_ops=np.array([[[1,0],[0,0]],[[0,0],[0,2]],[[0,1j],[-1j,0]],[[0,1],[1,0]]],complex)\n'
               'y=np.array([0.6,0.8,0.05,0.1])\n'
               'rho=np.array([[0.7,0.1j],[-0.1j,0.3]],complex)\n'
               'r=2; n=35; lam=0.012',
      'call': 'regularized_recovery(rho.copy(),r,A_ops.copy(),y.copy(),n,lam)',
      'gold_call': '_oracle_regularized_recovery(rho.copy(),r,A_ops.copy(),y.copy(),n,lam)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A_ops=np.array([[[1,0],[0,0]],[[0,0],[0,2]],[[0,1j],[-1j,0]],[[0,1],[1,0]]],complex)\n'
               'y=np.array([0.6,0.8,0.05,0.1])\n'
               'rho=np.array([[0.7,0.1j],[-0.1j,0.3]],complex)\n'
               'r=2; n=0; lam=0.0',
      'call': 'regularized_recovery(rho.copy(),r,A_ops.copy(),y.copy(),n,lam)',
      'gold_call': '_oracle_regularized_recovery(rho.copy(),r,A_ops.copy(),y.copy(),n,lam)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A_ops=np.array([[[1,0],[0,0]],[[0,0],[0,2]],[[0,1j],[-1j,0]],[[0,1],[1,0]]],complex)\n'
               'y=np.array([0.6,0.8,0.05,0.1])\n'
               'rho=np.array([[0.7,0.1j],[-0.1j,0.3]],complex)\n'
               'r=1; n=12; lam=0.08',
      'call': 'regularized_recovery(rho.copy(),r,A_ops.copy(),y.copy(),n,lam)',
      'gold_call': '_oracle_regularized_recovery(rho.copy(),r,A_ops.copy(),y.copy(),n,lam)',
      'tol': 1e-08}]
