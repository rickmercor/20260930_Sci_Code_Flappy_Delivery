"""
Construct the core matrix required by the subsequent randomized preconditioning subproblem from the previously computed state and sketch.

The randomized construction uses a small core representation of the sampled matrix to form the subsequent approximation state. For this benchmark, the core is constructed deterministically from the previously computed quantities and the prescribed regularization parameter. Inputs are samples $W=A\Omega$ of a symmetric operator $A$, with $V$ a thin-QR basis of $W$ and an invertible shifted sampled matrix. Use the full core convention in Algorithm 1, Step 6. Averaging the computed core with its transpose may remove floating-point asymmetry.

Returns
-------
np.ndarray, shape $(k,k)$ — symmetric core matrix $H$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def form_regularized_core(
    W: np.ndarray,
    V: np.ndarray,
    Omega: np.ndarray,
    eps: float,
) -> np.ndarray:
    r"""
    Form the core matrix required by the subsequent randomized
    preconditioning subproblems.

    Raises
    ------
    ValueError
        If $W$, $V$, or $\Omega$ is not a 2D array, if $W$, $V$, and
        $\Omega$ do not have identical shapes, or if $\varepsilon\leq0$
        (the parameter eps).
    """
    return H

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_form_regularized_core(
    W: np.ndarray,
    V: np.ndarray,
    Omega: np.ndarray,
    eps: float,
) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    V = np.asarray(V, dtype=float)
    Omega = np.asarray(Omega, dtype=float)

    if W.ndim != 2 or V.ndim != 2 or Omega.ndim != 2:
        raise ValueError("W, V, and Omega must be 2D arrays")

    if W.shape != V.shape or W.shape != Omega.shape:
        raise ValueError("W, V, and Omega must have identical shapes")

    if not (isinstance(eps, (int, float)) and float(eps) > 0.0):
        raise ValueError("eps must be positive")

    k = W.shape[1]

    Z = Omega.T @ W
    M = V.T @ W

    H = M @ np.linalg.solve(
        Z + float(eps) * np.eye(k, dtype=float), M.T
    )

    H = 0.5 * (H + H.T)

    return H.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: ordinary case ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(41)
m, k = 20, 5
S = rng.standard_normal((m, m))
A = S + S.T
Omega = rng.standard_normal((m, k))
W = A @ Omega
V, R = np.linalg.qr(W, mode="reduced")
eps = 1e-6
""",
            "call": "form_regularized_core(W, V, Omega, eps)",
            "gold_call": "_oracle_form_regularized_core(W, V, Omega, eps)",
        },

        # --- Valid: larger epsilon ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(42)
m, k = 15, 4
S = rng.standard_normal((m, m))
A = S + S.T
Omega = rng.standard_normal((m, k))
W = A @ Omega
V, R = np.linalg.qr(W, mode="reduced")
eps = 1e-2
""",
            "call": "form_regularized_core(W, V, Omega, eps)",
            "gold_call": "_oracle_form_regularized_core(W, V, Omega, eps)",
        },

        # --- Valid: exactly rank-deficient symmetric sampled matrix ---
        {
            "setup": """import numpy as np
W = np.diag([2.0, -3.0, 0.0])
V = np.eye(3)
Omega = np.eye(3)
eps = 1e-6
""",
            "call": "form_regularized_core(W, V, Omega, eps)",
            "gold_call": "np.diag([4.0 / (2.0 + eps), 9.0 / (-3.0 + eps), 0.0])",
        },

        # --- Invalid: eps <= 0 ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(43)
m, k = 10, 3
W = rng.standard_normal((m, k))
V, R = np.linalg.qr(W, mode="reduced")
Omega = rng.standard_normal((m, k))
eps = 0.0

def run_model():
    try:
        form_regularized_core(W, V, Omega, eps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_form_regularized_core(W, V, Omega, eps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
