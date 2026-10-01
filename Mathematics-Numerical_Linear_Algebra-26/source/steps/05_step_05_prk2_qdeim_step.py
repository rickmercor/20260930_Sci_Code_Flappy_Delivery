"""
Implement one prk2_qdeim_step on the rank-r matrix manifold.

Advance Y_i = U diag(s) Vh by one positive step h under rhs_fn and return the updated factors (U_new, s_new, Vh_new) at the requested rank. Recover the stage structure, coefficients, oblique projections, and rank retractions from the literature on second-order PRK-DEIM integration.

A projected second-order Runge-Kutta step combines stage velocities after mapping them to the tangent space of the current low-rank state. Intermediate and final rank truncations keep both stages and the returned factors on the rank-r manifold.

Returns
-------
Tuple[np.ndarray, np.ndarray, np.ndarray] as specified by the function Returns section.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from typing import Tuple

def prk2_qdeim_step(U: np.ndarray, s: np.ndarray, Vh: np.ndarray, rhs_fn, h: float, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one PRK2-QDEIM time step on the rank-r manifold.

    Parameters
    ----------
    U : np.ndarray, shape (n, r)
        Left orthonormal factor of Y_i.
    s : np.ndarray, shape (r,)
        Singular values of Y_i.
    Vh : np.ndarray, shape (r, n)
        Right factor (conjugate transpose of V) of Y_i.
    rhs_fn : callable
        Velocity field F(A), maps (n, n) -> (n, n).
    h : float
        Step size.
    r : int
        Target rank for truncation.

    Returns
    -------
    U_new : np.ndarray, shape (n, r)
    s_new : np.ndarray, shape (r,)
    Vh_new : np.ndarray, shape (r, n)
    """
    n = U.shape[0]
    U_new = np.zeros((n, r), dtype=complex)
    s_new = np.zeros(r)
    Vh_new = np.zeros((r, n), dtype=complex)
    return (U_new, s_new, Vh_new)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from typing import Tuple

def _oracle_prk2_qdeim_step(U: np.ndarray, s: np.ndarray, Vh: np.ndarray, rhs_fn, h: float, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    U = np.asarray(U, dtype=complex)
    s = np.asarray(s, dtype=float)
    Vh = np.asarray(Vh, dtype=complex)
    Y = U @ np.diag(s) @ Vh
    V = Vh.conj().T
    F1 = rhs_fn(Y)
    P1 = _oracle_oblique_project(U, V, F1)
    Z2 = Y + h * P1
    U2, s2, Vh2 = np.linalg.svd(Z2, full_matrices=False)
    U2, s2, Vh2 = (U2[:, :r], s2[:r], Vh2[:r, :])
    Y2 = U2 @ np.diag(s2) @ Vh2
    V2 = Vh2.conj().T
    F2 = rhs_fn(Y2)
    P2 = _oracle_oblique_project(U2, V2, F2)
    Y_new = Y + 0.5 * h * (P1 + P2)
    U_n, s_n, Vh_n = np.linalg.svd(Y_new, full_matrices=False)
    return (U_n[:, :r], s_n[:r], Vh_n[:r, :])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Case 1
        {
            "setup": """import numpy as np
np.random.seed(7)
n, r = 8, 3
A0 = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(A0, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: 0.5j * A
h = 0.01
def result_svs(step_fn):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h, r)
    return [round(float(x), 8) for x in s]
""",
            "call": "result_svs(prk2_qdeim_step)",
            "gold_call": "result_svs(_oracle_prk2_qdeim_step)",
        },
        # Case 2
        {
            "setup": """import numpy as np
np.random.seed(5)
n = 4
r = n
A0 = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(A0, full_matrices=False)
rhs_fn = lambda A: 1j * A
h = 0.01
def recon(step_fn):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h, r)
    Y = U @ np.diag(s) @ Vh
    return np.round(Y, 8).tolist()
""",
            "call": "recon(prk2_qdeim_step)",
            "gold_call": "recon(_oracle_prk2_qdeim_step)",
        },
        # Case 3
        {
            "setup": """import numpy as np
np.random.seed(50)
n = 4
r = n
A0 = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(A0, full_matrices=False)
rhs_fn = lambda A: 1j * A
exact = lambda h_val: np.exp(1j * h_val) * A0
def local_err(step_fn, h_val):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h_val, r)
    Y = U @ np.diag(s) @ Vh
    return float(np.linalg.norm(Y - exact(h_val)))
def check_order2(step_fn):
    e1 = local_err(step_fn, 0.01)
    e2 = local_err(step_fn, 0.005)
    if e1 < 1e-15 or e2 < 1e-15:
        return True
    ratio = e1 / e2
    return 3.0 < ratio < 5.0
""",
            "call": "check_order2(prk2_qdeim_step)",
            "gold_call": "check_order2(_oracle_prk2_qdeim_step)",
        },
        # Case 4
        {
            "setup": """import numpy as np
np.random.seed(88)
n, r = 10, 3
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.5
rhs_fn = lambda A: 0.5j * (B @ A + A @ B) + 1j * alpha * (A * A * A)
h = 0.005
def result_svs(step_fn):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h, r)
    return [round(float(x), 8) for x in s]
""",
            "call": "result_svs(prk2_qdeim_step)",
            "gold_call": "result_svs(_oracle_prk2_qdeim_step)",
        },
        # Case 5
        {
            "setup": """import numpy as np
np.random.seed(101)
n, r = 8, 3
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
rhs_fn = lambda A: 0.5j * (B @ A + A @ B) + 0.3j * (A * A * A)
h = 0.002
def recon(step_fn):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h, r)
    Y = U @ np.diag(s) @ Vh
    return np.round(Y, 7).tolist()
""",
            "call": "recon(prk2_qdeim_step)",
            "gold_call": "recon(_oracle_prk2_qdeim_step)",
        },
        # Case 6
        {
            "setup": """import numpy as np
np.random.seed(12)
n, r = 6, 2
A0 = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(A0, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: 0.1j * A
h = 0.001
def shapes(step_fn):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h, r)
    return (U.shape, s.shape, Vh.shape)
""",
            "call": "shapes(prk2_qdeim_step)",
            "gold_call": "shapes(_oracle_prk2_qdeim_step)",
        },
        # Case 7
        {
            "setup": """import numpy as np
np.random.seed(222)
n, r = 10, 4
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
rhs_fn = lambda A: 0.5j * (B @ A + A @ B) + 1j * 0.5 * (A * A * A)
h = 0.003
def recon_norm(step_fn):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h, r)
    Y = U @ np.diag(s) @ Vh
    return round(float(np.linalg.norm(Y, 'fro')), 8)
""",
            "call": "recon_norm(prk2_qdeim_step)",
            "gold_call": "recon_norm(_oracle_prk2_qdeim_step)",
        },
        # Case 8
        {
            "setup": """import numpy as np
np.random.seed(333)
n, r = 6, 3
A0 = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(A0, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: (0.3 + 0.7j) * A
h = 0.01
def recon_vals(step_fn):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h, r)
    Y = U @ np.diag(s) @ Vh
    return tuple(np.round(Y.ravel()[:6], 8).tolist())
""",
            "call": "recon_vals(prk2_qdeim_step)",
            "gold_call": "recon_vals(_oracle_prk2_qdeim_step)",
        },
        # Case 9
        {
            "setup": """import numpy as np
np.random.seed(444)
n, r = 8, 3
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
rhs_fn = lambda A: 0.5j * (B @ A + A @ B) + 1j * 0.5 * (A * A * A)
h = 0.005
def svs_nonneg(step_fn):
    U, s, Vh = step_fn(U0, s0, Vh0, rhs_fn, h, r)
    return all(float(x) >= -1e-15 for x in s)
""",
            "call": "svs_nonneg(prk2_qdeim_step)",
            "gold_call": "svs_nonneg(_oracle_prk2_qdeim_step)",
        },
    ]
