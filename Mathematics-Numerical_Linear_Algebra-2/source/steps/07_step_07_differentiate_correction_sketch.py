"""
Compute the first and mixed second derivatives of the recovered correction under the two-parameter sketch perturbation $\Omega(s,t)=\Omega+sP+tQ$.

The source-defined full regularized correction changes with the sampled subspace. Here $W(s,t)=W+sW_s+tW_t$, with $W=A\Omega$, $W_s=AP$, and $W_t=AQ$ for one fixed symmetric operator $A$. The benchmark asks for derivatives at $s=t=0$ with $\varepsilon$ held fixed. Use the full Algorithm 1 core convention. This sensitivity calculation is a benchmark extension, not a derivative formula asserted by the source.

Returns
-------
np.ndarray, shape $(3,m,m)$ — the stacked derivatives $(\widehat{\Delta}_s,\widehat{\Delta}_t,\widehat{\Delta}_{st})$ at $s=t=0$; the mixed derivative is unscaled.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def differentiate_correction_sketch(
    W: np.ndarray,
    Omega: np.ndarray,
    W_s: np.ndarray,
    W_t: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    eps: float,
) -> np.ndarray:
    r"""
    Compute the derivative state required by the subsequent
    generalized-projector subproblems.

    Raises
    ------
    ValueError
        If any input array is not 2D, their shapes differ, or
        $\varepsilon\leq0$ or is nonfinite.
    """
    return derivatives

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_differentiate_correction_sketch(
    W: np.ndarray,
    Omega: np.ndarray,
    W_s: np.ndarray,
    W_t: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    eps: float,
) -> np.ndarray:
    W, Omega, W_s, W_t, P, Q = [
        np.asarray(a, dtype=float) for a in (W, Omega, W_s, W_t, P, Q)
    ]
    if any(a.ndim != 2 or a.shape != W.shape for a in (W, Omega, W_s, W_t, P, Q)):
        raise ValueError("all six arrays must be 2D with identical shapes")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    Z = Omega.T @ W
    Z_s = P.T @ W + Omega.T @ W_s
    Z_t = Q.T @ W + Omega.T @ W_t
    Z_st = P.T @ W_t + Q.T @ W_s
    B = np.linalg.solve(Z + eps * np.eye(W.shape[1]), np.eye(W.shape[1]))
    B_s = -B @ Z_s @ B
    B_t = -B @ Z_t @ B
    B_st = B @ Z_s @ B @ Z_t @ B + B @ Z_t @ B @ Z_s @ B - B @ Z_st @ B
    D_s = W_s @ B @ W.T + W @ B_s @ W.T + W @ B @ W_s.T
    D_t = W_t @ B @ W.T + W @ B_t @ W.T + W @ B @ W_t.T
    D_st = (
        W_s @ B_t @ W.T + W_s @ B @ W_t.T
        + W_t @ B_s @ W.T + W @ B_st @ W.T + W @ B_s @ W_t.T
        + W_t @ B @ W_s.T + W @ B_t @ W_s.T
    )
    derivatives = np.stack((D_s, D_t, D_st))
    return 0.5 * (derivatives + derivatives.transpose(0, 2, 1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(81)
m, k = 8, 3
A = np.diag([4., -3., 2., -1., .8, -.5, .3, -.1])
Omega = rng.standard_normal((m, k))
P = rng.standard_normal((m, k))
Q = rng.standard_normal((m, k))
W, W_s, W_t = A @ Omega, A @ P, A @ Q
eps = 0.1
""",
            "call": "differentiate_correction_sketch(W, Omega, W_s, W_t, P, Q, eps)",
            "gold_call": "_oracle_differentiate_correction_sketch(W, Omega, W_s, W_t, P, Q, eps)",
        },
        {
            "setup": """import numpy as np
A = np.diag([2., -1.])
Omega = np.array([[1.], [0.]])
P = Q = np.array([[0.], [1.]])
W, W_s, W_t = A @ Omega, A @ P, A @ Q
eps = 0.5
""",
            "call": "differentiate_correction_sketch(W, Omega, W_s, W_t, P, Q, eps)",
            "gold_call": "np.array([[[0., -.8], [-.8, 0.]], [[0., -.8], [-.8, 0.]], [[1.28, 0.], [0., .8]]])",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(83)
m, k = 6, 2
A = np.diag([3., -2., 1., -.5, .2, -.1])
Omega = rng.standard_normal((m, k))
P = np.zeros((m, k))
Q = rng.standard_normal((m, k))
W, W_s, W_t = A @ Omega, A @ P, A @ Q
eps = 0.01
""",
            "call": "differentiate_correction_sketch(W, Omega, W_s, W_t, P, Q, eps)",
            "gold_call": "_oracle_differentiate_correction_sketch(W, Omega, W_s, W_t, P, Q, eps)",
        },
        {
            "setup": """import numpy as np
W = Omega = W_s = W_t = P = np.ones((3, 2))
Q = np.ones((3, 1))
eps = 0.1

def run_model():
    try:
        differentiate_correction_sketch(W, Omega, W_s, W_t, P, Q, eps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_differentiate_correction_sketch(W, Omega, W_s, W_t, P, Q, eps)
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
