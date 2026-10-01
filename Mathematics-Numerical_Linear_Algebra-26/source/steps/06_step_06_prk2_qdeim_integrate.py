"""
Implement prk2_qdeim_integrate over a fixed time interval.

Starting from Y_0 = U0 diag(s0) Vh0, repeatedly call the earlier prk2_qdeim_step routine to advance from t0 to t_end and return the rank-r factors at the final time. Reject a nonpositive step size or a reversed interval with ValueError.

Low-rank time integration applies the single-step projected update repeatedly over a fixed interval. Carrying the factored representation between steps preserves the requested rank and avoids reconstructing a new integration method inside the driver.

Returns
-------
Tuple[np.ndarray, np.ndarray, np.ndarray] as specified by the function Returns section.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from typing import Tuple

def prk2_qdeim_integrate(U0: np.ndarray, s0: np.ndarray, Vh0: np.ndarray, rhs_fn, t0: float, t_end: float, h: float, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a low-rank ODE from t0 to t_end using PRK2-QDEIM.

Parameters
----------
U0 : np.ndarray, shape (n, r)
    Left orthonormal factor at t0.
s0 : np.ndarray, shape (r,)
    Singular values at t0.
Vh0 : np.ndarray, shape (r, n)
    Right factor (V*) at t0.
rhs_fn : callable
    Velocity field F(A), maps (n, n) array -> (n, n) array.
t0 : float
    Start time.
t_end : float
    End time.
h : float
    Step size (must be strictly positive).
r : int
    Target rank for truncation.

Returns
-------
U_N : np.ndarray, shape (n, r)
s_N : np.ndarray, shape (r,)
Vh_N : np.ndarray, shape (r, n)

Implementation requirement
--------------------------
Call the previously defined public functions ``prk2_qdeim_step`` rather than reproducing their algorithms locally."""
    n = U0.shape[0]
    return (np.zeros((n, r), dtype=complex), np.zeros(r), np.zeros((r, n), dtype=complex))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from typing import Tuple

def _oracle_prk2_qdeim_integrate(U0: np.ndarray, s0: np.ndarray, Vh0: np.ndarray, rhs_fn, t0: float, t_end: float, h: float, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    if h <= 0:
        raise ValueError('h must be > 0')
    if t_end < t0:
        raise ValueError('t_end must be >= t0')
    n_steps = int(round((t_end - t0) / h))
    U, s, Vh = (np.asarray(U0, dtype=complex).copy(), np.asarray(s0, dtype=float).copy(), np.asarray(Vh0, dtype=complex).copy())
    for _ in range(n_steps):
        U, s, Vh = _oracle_prk2_qdeim_step(U, s, Vh, rhs_fn, h, r)
    return (U, s, Vh)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Case 1
        {
            "setup": """import numpy as np
np.random.seed(20)
n, r = 6, 2
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: 0.5j * A
h = 0.01
def recon_norm(int_fn):
    U, s, Vh = int_fn(U0, s0, Vh0, rhs_fn, 0.0, h, h, r)
    return round(float(np.linalg.norm(U @ np.diag(s) @ Vh, 'fro')), 8)
""",
            "call": "recon_norm(prk2_qdeim_integrate)",
            "gold_call": "recon_norm(_oracle_prk2_qdeim_integrate)",
        },
        # Case 2
        {
            "setup": """import numpy as np
np.random.seed(44)
n, r = 8, 3
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: 0.1j * A
def shape_check(int_fn):
    U, s, Vh = int_fn(U0, s0, Vh0, rhs_fn, 0.0, 0.05, 0.01, r)
    return (U.shape == (n, r), s.shape == (r,), Vh.shape == (r, n))
""",
            "call": "shape_check(prk2_qdeim_integrate)",
            "gold_call": "shape_check(_oracle_prk2_qdeim_integrate)",
        },
        # Case 3
        {
            "setup": """import numpy as np
np.random.seed(60)
n, r = 4, 4
A0 = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(A0, full_matrices=False)
rhs_fn = lambda A: 1j * A
def result_svs(int_fn):
    U, s, Vh = int_fn(U0, s0, Vh0, rhs_fn, 0.0, 0.1, 0.01, r)
    return [round(float(x), 8) for x in s]
""",
            "call": "result_svs(prk2_qdeim_integrate)",
            "gold_call": "result_svs(_oracle_prk2_qdeim_integrate)",
        },
        # Case 4
        {
            "setup": """import numpy as np
np.random.seed(15)
n, r = 6, 2
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.5
rhs_fn = lambda A: 0.5j * (B @ A + A @ B) + 1j * alpha * (A * A * A)
def result_norm(int_fn):
    U, s, Vh = int_fn(U0, s0, Vh0, rhs_fn, 0.0, 0.05, 0.01, r)
    return round(float(np.sum(s)), 6)
""",
            "call": "result_norm(prk2_qdeim_integrate)",
            "gold_call": "result_norm(_oracle_prk2_qdeim_integrate)",
        },
        # Case 5
        {
            "setup": """import numpy as np
np.random.seed(333)
n, r = 8, 3
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
rhs_fn = lambda A: 0.5j * (B @ A + A @ B) + 0.2j * (A * A * A)
def recon(int_fn):
    U, s, Vh = int_fn(U0, s0, Vh0, rhs_fn, 0.0, 0.025, 0.005, r)
    Y = U @ np.diag(s) @ Vh
    return np.round(Y, 7).tolist()
""",
            "call": "recon(prk2_qdeim_integrate)",
            "gold_call": "recon(_oracle_prk2_qdeim_integrate)",
        },
        # Case 6
        {
            "setup": """import numpy as np
np.random.seed(99)
n, r = 8, 4
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: 0.5j * A
def svs_nonneg(int_fn):
    U, s, Vh = int_fn(U0, s0, Vh0, rhs_fn, 0.0, 0.1, 0.01, r)
    return all(float(x) >= -1e-15 for x in s)
""",
            "call": "svs_nonneg(prk2_qdeim_integrate)",
            "gold_call": "svs_nonneg(_oracle_prk2_qdeim_integrate)",
        },
        # Case 7
        {
            "setup": """import numpy as np
np.random.seed(10)
n, r = 4, 2
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: 0.1j * A
def run_model():
    try:
        prk2_qdeim_integrate(U0, s0, Vh0, rhs_fn, 0.0, 0.1, -0.01, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_prk2_qdeim_integrate(U0, s0, Vh0, rhs_fn, 0.0, 0.1, -0.01, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 8
        {
            "setup": """import numpy as np
np.random.seed(10)
n, r = 4, 2
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: 0.1j * A
def run_model():
    try:
        prk2_qdeim_integrate(U0, s0, Vh0, rhs_fn, 1.0, 0.5, 0.01, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_prk2_qdeim_integrate(U0, s0, Vh0, rhs_fn, 1.0, 0.5, 0.01, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 9
        {
            "setup": """import numpy as np
np.random.seed(50)
n, r = 6, 2
G = np.random.randn(n, n) + 1j * np.random.randn(n, n)
U0, s0, Vh0 = np.linalg.svd(G, full_matrices=False)
U0, s0, Vh0 = U0[:, :r], s0[:r], Vh0[:r, :]
rhs_fn = lambda A: 0.5j * A
def unchanged(int_fn):
    U, s, Vh = int_fn(U0.copy(), s0.copy(), Vh0.copy(), rhs_fn, 0.5, 0.5, 0.01, r)
    Y_orig = U0 @ np.diag(s0) @ Vh0
    Y_out = U @ np.diag(s) @ Vh
    return float(np.linalg.norm(Y_orig - Y_out)) < 1e-12
""",
            "call": "unchanged(prk2_qdeim_integrate)",
            "gold_call": "unchanged(_oracle_prk2_qdeim_integrate)",
        },
    ]
