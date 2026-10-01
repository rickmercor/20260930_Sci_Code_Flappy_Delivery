"""
Jointly recover a partially observed matrix that shares its row axis with a completed N-way landscape, by iterating a proximal block-coordinate scheme to a critical point of one penalized objective, and return the recovered matrix.

The objective has six terms: one half the squared Frobenius misfit of the landscape iterate to the supplied completed landscape over all cells; one half the squared Frobenius misfit of the matrix iterate to its observed cells only; lam_coupling times one half the squared Frobenius norm of the residual between the landscape's first-mode flattening and the matrix times the transpose of a coupling operator; delta times one half the squared Frobenius norm of that operator; lam_landscape times the nuclear norm of the flattening; and lam_health times the nuclear norm of the matrix. Conventions the tests depend on: the flattening reshapes the landscape row-major (C order) to (n_rows, product of the remaining sizes); the landscape iterate starts at the supplied landscape, the matrix iterate at its observed values with zeros elsewhere, and the coupling operator at zero; each sweep updates the landscape block, then the matrix block, then the operator block; the landscape and matrix blocks use step sizes of 0.9 times the reciprocal of their block Lipschitz constants, 1 + lam_coupling for the landscape and 1 + lam_coupling times the squared spectral norm of the current operator for the matrix; the function returns the matrix iterate after exactly n_iters sweeps, so n_iters = 0 returns the initialization. On the task's instance a few thousand sweeps reach the critical point, which is insensitive to the starting point.

Returns
-------
np.ndarray of shape health_dims: the recovered health matrix after n_iters block sweeps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def recover_coupled_matrices(
    landscape: np.ndarray,
    health_obs: dict,
    health_dims: tuple,
    lam_landscape: float,
    lam_health: float,
    lam_coupling: float,
    delta: float,
    n_iters: int,
) -> np.ndarray:
    """Recover the coupled matrix by proximal block-coordinate iteration.

    Parameters
    ----------
    landscape : np.ndarray
        The completed N-way landscape (first axis is the shared row axis); the
        fixed target of the landscape data term and the landscape iterate's
        starting point.
    health_obs : dict
        Mapping from 2-tuple (row, column) indices to observed float values of
        the partially observed matrix.
    health_dims : tuple
        (n_rows, n_cols) of the matrix; n_rows must equal landscape.shape[0].
    lam_landscape, lam_health, lam_coupling : float
        Nuclear-norm weight on the landscape flattening, nuclear-norm weight
        on the matrix, and weight on the coupling-residual term.
    delta : float
        Ridge weight on the coupling operator (must be positive).
    n_iters : int
        Number of block sweeps to perform (0 returns the initialization).

    Returns
    -------
    health_matrix : np.ndarray
        Array of shape health_dims, the matrix iterate after n_iters sweeps.

    Raises
    ------
    ValueError
        If health_dims[0] differs from landscape.shape[0], a nuclear-norm or
        coupling weight is negative, delta is not positive, n_iters is
        negative, or an observed index is out of bounds.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _svt(X: np.ndarray, tau: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (U * np.maximum(s - tau, 0.0)) @ Vt


def _oracle_recover_coupled_matrices(
    landscape: np.ndarray,
    health_obs: dict,
    health_dims: tuple,
    lam_landscape: float,
    lam_health: float,
    lam_coupling: float,
    delta: float,
    n_iters: int,
) -> np.ndarray:
    if landscape.shape[0] != health_dims[0]:
        raise ValueError("health_dims[0] must match landscape.shape[0]")
    if lam_landscape < 0 or lam_health < 0 or lam_coupling < 0:
        raise ValueError("lam_landscape, lam_health, lam_coupling must be non-negative")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if n_iters < 0:
        raise ValueError("n_iters must be non-negative")
    for idx in health_obs:
        if not (0 <= idx[0] < health_dims[0] and 0 <= idx[1] < health_dims[1]):
            raise ValueError("health_obs index out of bounds for health_dims")

    n_rows = landscape.shape[0]
    flat_dim = int(np.prod(landscape.shape[1:]))
    n_cols = health_dims[1]
    yM = np.zeros(health_dims)
    OmM = np.zeros(health_dims)
    for idx, val in health_obs.items():
        yM[idx] = val
        OmM[idx] = 1.0
    T = landscape.copy()
    M = yM * OmM
    G = np.zeros((flat_dim, n_cols))
    tT = 0.9 / (1.0 + lam_coupling)
    for _ in range(n_iters):
        grad_T = (T - landscape) + lam_coupling * (
            T.reshape(n_rows, flat_dim) - M @ G.T
        ).reshape(landscape.shape)
        T = _svt((T - tT * grad_T).reshape(n_rows, flat_dim), tT * lam_landscape).reshape(landscape.shape)
        U1 = T.reshape(n_rows, flat_dim)
        grad_M = OmM * (M - yM) + lam_coupling * (M @ G.T - U1) @ G
        step_M = 0.9 / (1.0 + lam_coupling * np.linalg.norm(G, 2) ** 2)
        M = _svt(M - step_M * grad_M, step_M * lam_health)
        G = np.linalg.solve(
            lam_coupling * M.T @ M + delta * np.eye(n_cols), lam_coupling * M.T @ U1
        ).T
    return M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
landscape = np.einsum('i,j,k->ijk', [0.5, 0.8, 0.3], [0.4, 0.6], [1.0, 1.2])
health_obs = {(0,0): 0.35, (1,1): 0.62, (2,0): 0.15}
health_dims = (3, 2)
lam_landscape = lam_health = lam_coupling = 0.1
delta = 0.05
n_iters = 3000
""",
            "call": "recover_coupled_matrices(landscape.copy(), health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters)",
            "gold_call": "_oracle_recover_coupled_matrices(landscape.copy(), health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters)",
        },
        {
            "setup": """
import numpy as np
landscape = np.ones((3, 2, 2))
health_obs = {(0,0): 0.3, (1,1): 0.7}
health_dims = (3, 2)
lam_landscape = lam_health = lam_coupling = 0.1
delta = 0.05
n_iters = 0
""",
            "call": "recover_coupled_matrices(landscape.copy(), health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters)",
            "gold_call": "_oracle_recover_coupled_matrices(landscape.copy(), health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters)",
        },
        {
            "setup": """
import numpy as np
u = np.array([0.6688, 0.4928, 0.9064, 0.8976, 0.6248, 0.7040])
v = np.array([1.0, 1.5, 1.75, 1.375])
w = np.array([1.0, 8.0/11.0, 9.0/11.0])
landscape = np.einsum('i,j,k->ijk', u, v, w)
health_obs = {(0,2):0.5971,(0,3):0.6371,(1,1):0.6107,(1,3):0.6874,(2,2):0.5769,
              (2,3):0.6285,(3,0):0.6851,(3,2):0.7550,(4,1):0.6332,(4,3):0.7840,
              (5,2):0.5173,(5,3):0.5639}
health_dims = (6, 4)
lam_landscape = lam_health = lam_coupling = 0.20
delta = 0.10
n_iters = 5000
""",
            "call": "recover_coupled_matrices(landscape.copy(), health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters)",
            "gold_call": "_oracle_recover_coupled_matrices(landscape.copy(), health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters)",
        },
    ]
