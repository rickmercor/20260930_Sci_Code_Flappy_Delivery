"""
Project a Hermitian matrix onto the set of rank-r-or-less density matrices (Hermitian, positive semidefinite, trace 1, rank at most r), in Frobenius norm.

This step enforces the physical density-matrix constraints after an unconstrained Hermitian

estimate has been formed.

Returns
-------
(d, d) complex array, the nearest rank-constrained density matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def project_rank_r_density_matrix(Ybar: "np.ndarray", r: int) -> "np.ndarray":
    """Find the closest rank-<=r density matrix to a given Hermitian matrix, in Frobenius norm.

    Parameters
    ----------
    Ybar : np.ndarray
        (d, d) Hermitian complex matrix to project.
    r : int
        Maximum allowed rank of the output, 1 <= r <= d.

    Returns
    -------
    result : np.ndarray
        (d, d) complex Hermitian matrix, positive semidefinite, trace 1, rank at most r,
        minimizing the Frobenius distance to Ybar among all such matrices.

    Raises
    ------
    ValueError
        If Ybar is not square, not Hermitian (within a small tolerance), or r is not in [1, d].
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_project_rank_r_density_matrix(Ybar: "np.ndarray", r: int) -> "np.ndarray":
    d = Ybar.shape[0]
    if Ybar.shape[0] != Ybar.shape[1]:
        raise ValueError("Ybar must be square")
    if not np.allclose(Ybar, Ybar.conj().T, atol=1e-8):
        raise ValueError("Ybar must be Hermitian")
    if not (1 <= r <= d):
        raise ValueError("r must satisfy 1 <= r <= d")

    w, V = np.linalg.eigh(Ybar)
    idx = np.argsort(w)[::-1]
    w_sorted = w[idx]
    V_sorted = V[:, idx]
    top_r = w_sorted[:r]

    u = np.sort(top_r)[::-1]
    css = np.cumsum(u)
    candidates = np.nonzero(u * np.arange(1, r + 1) > (css - 1))[0]
    rho_idx = candidates[-1]
    theta = (css[rho_idx] - 1) / (rho_idx + 1)
    proj = np.maximum(top_r - theta, 0)

    sigma = (V_sorted[:, :r] * proj) @ V_sorted[:, :r].conj().T
    return sigma

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: 3x3 Hermitian with a mix of positive and negative eigenvalues, r=2 ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(3)
A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
Q, _ = np.linalg.qr(A)
Ybar = (Q * np.array([0.9, 0.3, -0.2])) @ Q.conj().T
Ybar = (Ybar + Ybar.conj().T)/2
r = 2
""",
            "call": "project_rank_r_density_matrix(np.array(Ybar, copy=True), r)",
            "gold_call": "_oracle_project_rank_r_density_matrix(np.array(Ybar, copy=True), r)",
        },
        # --- Boundary: r equals the full dimension, projection is a pure simplex projection of all eigenvalues ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(9)
A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
Q, _ = np.linalg.qr(A)
Ybar = (Q * np.array([0.8, 0.5, -0.3])) @ Q.conj().T
Ybar = (Ybar + Ybar.conj().T)/2
r = 3
""",
            "call": "project_rank_r_density_matrix(np.array(Ybar, copy=True), r)",
            "gold_call": "_oracle_project_rank_r_density_matrix(np.array(Ybar, copy=True), r)",
        },
        # --- Edge: the actual task instance's Ybar (Stage 1 output before projection), r=2 ---
        {
            "setup": """
import numpy as np
rng0 = np.random.default_rng(31)
eigvals = np.array([0.6, 0.4, 0.0, 0.0])
Araw = rng0.standard_normal((4,4)) + 1j*rng0.standard_normal((4,4))
Q, _ = np.linalg.qr(Araw)
rho_star = (Q * eigvals) @ Q.conj().T
rho_star = (rho_star + rho_star.conj().T)/2
V_perp = Q[:, 2:]
rng1 = np.random.default_rng(4100)
Z = (rng1.standard_normal((2,6)) + 1j*rng1.standard_normal((2,6)))/np.sqrt(2)
K, N = 6, 12
E_coords = (Z @ Z.conj().T - K*np.eye(2))/N
Ybar = rho_star + V_perp @ E_coords @ V_perp.conj().T
r = 2
""",
            "call": "project_rank_r_density_matrix(np.array(Ybar, copy=True), r)",
            "gold_call": "_oracle_project_rank_r_density_matrix(np.array(Ybar, copy=True), r)",
        },
    ]
