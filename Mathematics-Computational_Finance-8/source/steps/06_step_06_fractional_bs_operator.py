"""
Form the semi-discrete Black-Scholes spatial operator from the differentiation matrices, leaving the Dirichlet rows empty.

After spatial discretisation the time-fractional equation in time to maturity becomes a system of fractional ordinary differential equations whose right-hand side is L U with L = (1/2) sigma^2 diag(S)^2 D_SS + (r - q) diag(S) D_S - r I. The rows of L at the two end nodes are set to zero because those unknowns are prescribed by boundary data rather than by the equation.

Returns
-------
np.ndarray, float, shape (N, N): the operator L with rows 0 and N - 1 equal to zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fractional_bs_operator(s_grid: np.ndarray, diff_matrices: np.ndarray, sigma: float,
                           rate: float, dividend: float) -> np.ndarray:
    '''Semi-discrete spatial operator of the pricing equation.

    Parameters
    ----------
    s_grid : np.ndarray
        Strictly increasing asset grid of N >= 3 nodes.
    diff_matrices : np.ndarray
        Shape (2, N, N) array: [0] first-derivative and [1] second-derivative
        matrix on ``s_grid``.
    sigma : float
        Volatility, sigma >= 0.
    rate : float
        Risk-free rate.
    dividend : float
        Continuous dividend yield.

    Returns
    -------
    operator : np.ndarray
        Shape (N, N) float array with rows 0 and N - 1 set to zero.

    Raises
    ------
    ValueError
        If the grid is not a finite strictly increasing one-dimensional array
        of at least three nodes, if diff_matrices does not have shape
        (2, N, N) or is not finite, if sigma is negative, or if any
        coefficient is not finite.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((np.asarray(s_grid).size, np.asarray(s_grid).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_fractional_bs_operator(s_grid: np.ndarray, diff_matrices: np.ndarray, sigma: float,
                                   rate: float, dividend: float) -> np.ndarray:
    import numpy as np

    s = np.asarray(s_grid, dtype=float)
    if s.ndim != 1 or s.size < 3 or not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0.0):
        raise ValueError("s_grid must be a finite, strictly increasing array of at least 3 nodes")
    n = s.size
    d = np.asarray(diff_matrices, dtype=float)
    if d.shape != (2, n, n) or not np.all(np.isfinite(d)):
        raise ValueError("diff_matrices must be a finite array of shape (2, N, N)")
    for name, value in (("sigma", sigma), ("rate", rate), ("dividend", dividend)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(sigma) < 0.0:
        raise ValueError("sigma must be non-negative")

    op = (0.5 * float(sigma) ** 2 * (s ** 2)[:, None] * d[1]
          + (float(rate) - float(dividend)) * s[:, None] * d[0]
          - float(rate) * np.eye(n))
    # End rows are replaced by boundary data in the time stepper.
    op[0, :] = 0.0
    op[-1, :] = 0.0
    return op

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: with centred second-order matrices, exact on affine
        # functions, the operator maps u = S + 1 to (r - q) S - r (S + 1)
        # = -q S - r on interior rows, and the empty end rows map it to 0.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 4.0, 9)
h = S[1] - S[0]
D = np.zeros((2, 9, 9))
for i in range(1, 8):
    D[0, i, i - 1], D[0, i, i + 1] = -0.5 / h, 0.5 / h
    D[1, i, i - 1], D[1, i, i], D[1, i, i + 1] = 1 / h ** 2, -2 / h ** 2, 1 / h ** 2
EXPECTED = np.round(np.concatenate([[0.0], -0.03 * S[1:-1] - 0.05, [0.0]]), 10)
""",
            "call": "np.round(fractional_bs_operator(S, D, 0.3, 0.05, 0.03) @ (S + 1.0), 10)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: dense deterministic matrices on a uniform grid.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(7)
S = np.linspace(0.0, 30.0, 11)
D = rng.normal(size=(2, 11, 11))
""",
            "call": "fractional_bs_operator(S, D, 0.4, 0.05, 0.0)",
            "gold_call": "_oracle_fractional_bs_operator(S, D, 0.4, 0.05, 0.0)",
        },
        # --- Boundary: zero volatility, so only the drift and discount act.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
S = np.array([0.0, 0.5, 1.7, 2.0, 3.6])
D = rng.normal(size=(2, 5, 5))
""",
            "call": "fractional_bs_operator(S, D, 0.0, 0.02, 0.07)",
            "gold_call": "_oracle_fractional_bs_operator(S, D, 0.0, 0.02, 0.07)",
        },
        # --- Edge: smallest grid, a negative rate and a large dividend.
        {
            "setup": """import numpy as np
S = np.array([0.0, 1.0, 2.0])
D = np.arange(18.0).reshape(2, 3, 3)
""",
            "call": "fractional_bs_operator(S, D, 1.5, -0.01, 0.2)",
            "gold_call": "_oracle_fractional_bs_operator(S, D, 1.5, -0.01, 0.2)",
        },
        # --- Invalid: matrix shape does not match the grid ---
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 1.0, 5)
D = np.zeros((2, 4, 4))
def run_model():
    try:
        fractional_bs_operator(S, D, 0.2, 0.05, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fractional_bs_operator(S, D, 0.2, 0.05, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative volatility ---
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 1.0, 5)
D = np.zeros((2, 5, 5))
def run_model():
    try:
        fractional_bs_operator(S, D, -0.2, 0.05, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fractional_bs_operator(S, D, -0.2, 0.05, 0.0)
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
