"""
Compute the mixed derivative of the invariant projector onto the leading generalized eigenspace of a symmetric matrix pencil with a varying positive-definite metric.

Let $D(s,t)$ and $T(s,t)$ be real symmetric matrix-valued maps with $T(s,t)$ positive definite near the origin. Let $U_r(s,t)$ span the eigenspace of the $r$ largest algebraic eigenvalues of $Du=\lambda Tu$, normalized by $U_r^TTU_r=I_r$. The selected cluster is separated from its complement by a positive spectral gap, but eigenvalues within either cluster may be repeated. The metric-orthogonal invariant projector is $\Pi=U_rU_r^TT$. Its mixed derivative must include variation of the metric and remain invariant under basis rotations within repeated-eigenvalue clusters. This is a benchmark-defined spectral sensitivity extension; no projector-derivative identity is attributed to the source paper.

Returns
-------
np.ndarray, shape $(n,n)$ — $\Pi_{st}=\partial_s\partial_t\Pi(0,0)$ in the original coordinates, with no factorial factor; it need not be Euclidean symmetric.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mixed_generalized_projector_derivative(
    D: np.ndarray,
    T: np.ndarray,
    D_s: np.ndarray,
    T_s: np.ndarray,
    D_t: np.ndarray,
    T_t: np.ndarray,
    D_st: np.ndarray,
    T_st: np.ndarray,
    r: int,
) -> np.ndarray:
    r"""
    Compute the mixed derivative state required by the final numerical
    evaluation.

    Raises
    ------
    ValueError
        If the arrays do not share a nonempty square shape, are not finite
        and symmetric to absolute tolerance $10^{-12}$, $T$ is not SPD,
        or $r$ is not an integer satisfying $1\leq r<n$.
    """
    return projector_st

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mixed_generalized_projector_derivative(
    D: np.ndarray,
    T: np.ndarray,
    D_s: np.ndarray,
    T_s: np.ndarray,
    D_t: np.ndarray,
    T_t: np.ndarray,
    D_st: np.ndarray,
    T_st: np.ndarray,
    r: int,
) -> np.ndarray:
    arrays = [np.asarray(x, dtype=float) for x in (D, T, D_s, T_s, D_t, T_t, D_st, T_st)]
    D = arrays[0]
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("matrix inputs must be nonempty and square")
    n = D.shape[0]
    for x in arrays:
        if x.shape != D.shape or not np.all(np.isfinite(x)) or not np.allclose(x, x.T, atol=1e-12, rtol=0):
            raise ValueError("matrix inputs must have the same shape and be finite and symmetric")
    if not isinstance(r, (int, np.integer)) or not 1 <= r < n:
        raise ValueError("r must be an integer satisfying 1 <= r < n")
    D, T, D_s, T_s, D_t, T_t, D_st, T_st = [0.5 * (x + x.T) for x in arrays]
    try:
        L = np.linalg.cholesky(T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("T must be SPD") from exc
    S = np.linalg.solve(L, D)
    S = np.linalg.solve(L, S.T).T
    eigenvalues, U = np.linalg.eigh(0.5 * (S + S.T))
    V = np.linalg.solve(L.T, U)
    V_inv = U.T @ L.T
    selected = np.zeros(n)
    selected[-r:] = 1.0

    A = np.linalg.solve(T, D)
    A_s = np.linalg.solve(T, D_s - T_s @ A)
    A_t = np.linalg.solve(T, D_t - T_t @ A)
    A_st = np.linalg.solve(T, D_st - T_st @ A - T_s @ A_t - T_t @ A_s)
    E_s, E_t, E_st = [V_inv @ x @ V for x in (A_s, A_t, A_st)]
    cross = selected[:, None] != selected[None, :]
    gaps = eigenvalues[:, None] - eigenvalues[None, :]
    factors = np.zeros((n, n))
    factors[cross] = (selected[:, None] - selected[None, :])[cross] / gaps[cross]
    P_s = factors * E_s
    P_t = factors * E_t
    commutator = (
        E_s @ P_t - P_t @ E_s + E_t @ P_s - P_s @ E_t
        + E_st * selected[None, :] - selected[:, None] * E_st
    )
    P_st = np.zeros((n, n))
    P_st[cross] = -commutator[cross] / gaps[cross]
    products = P_s @ P_t + P_t @ P_s
    inside = (selected[:, None] == 1) & (selected[None, :] == 1)
    outside = (selected[:, None] == 0) & (selected[None, :] == 0)
    P_st[inside] = -products[inside]
    P_st[outside] = products[outside]
    return V @ P_st @ V_inv

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(91)
n, r = 6, 2
L = np.eye(n) + 0.1 * np.tril(rng.standard_normal((n, n)))
T = L @ L.T
D = L @ np.diag([5., 5., 1., 1., -2., -2.]) @ L.T
jets = []
for _ in range(6):
    X = 0.2 * rng.standard_normal((n, n))
    jets.append(X + X.T)
D_s, T_s, D_t, T_t, D_st, T_st = jets
""",
            "call": "mixed_generalized_projector_derivative(D, T, D_s, T_s, D_t, T_t, D_st, T_st, r)",
            "gold_call": "_oracle_mixed_generalized_projector_derivative(D, T, D_s, T_s, D_t, T_t, D_st, T_st, r)",
        },
        {
            "setup": """import numpy as np
D, T = np.diag([3., 1.]), np.eye(2)
D_s = D_t = np.array([[0., 1.], [1., 0.]])
T_s = T_t = D_st = T_st = np.zeros((2, 2))
r = 1
""",
            "call": "mixed_generalized_projector_derivative(D, T, D_s, T_s, D_t, T_t, D_st, T_st, r)",
            "gold_call": "np.diag([-0.5, 0.5])",
        },
        {
            "setup": """import numpy as np
D, T = np.diag([3., 1.]), np.eye(2)
T_s = T_t = np.array([[0., 1.], [1., 0.]])
D_s = D_t = D_st = T_st = np.zeros((2, 2))
r = 1
""",
            "call": "mixed_generalized_projector_derivative(D, T, D_s, T_s, D_t, T_t, D_st, T_st, r)",
            "gold_call": "np.diag([-1.5, 1.5])",
        },
        {
            "setup": """import numpy as np
D = T = np.eye(2)
D_s = T_s = D_t = T_t = D_st = T_st = np.zeros((2, 2))
r = 0

def run_model():
    try:
        mixed_generalized_projector_derivative(D, T, D_s, T_s, D_t, T_t, D_st, T_st, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_mixed_generalized_projector_derivative(D, T, D_s, T_s, D_t, T_t, D_st, T_st, r)
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
