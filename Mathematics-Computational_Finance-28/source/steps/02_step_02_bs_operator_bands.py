"""
Return the three-band coefficients of the discounted Black-Scholes generator on a uniform asset grid.

The generator acting on a function V of the asset price is A V = (1/2) b^2 S^2 V'' + r S V' - r V. On a uniform grid it is discretised with second-order central differences, so row i couples V_{i-1}, V_i and V_{i+1} only. The same interior formula is evaluated at every node, including the two end nodes, whose coefficients are not used by the time stepper because those values are prescribed as boundary data.

Returns
-------
np.ndarray, float, shape (3, n): row 0 the coefficients of V_{i-1}, row 1 of V_i, row 2 of V_{i+1}, for each node i.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bs_operator_bands(s_grid: np.ndarray, rate: float, vol: float) -> np.ndarray:
    '''Tridiagonal coefficients of the discounted Black-Scholes generator.

    Parameters
    ----------
    s_grid : np.ndarray
        Strictly increasing, uniformly spaced, non-negative grid of n >= 3 prices.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b >= 0.

    Returns
    -------
    bands : np.ndarray
        Shape (3, n) float array: [0] sub-diagonal, [1] diagonal, [2]
        super-diagonal coefficient of each row.

    Raises
    ------
    ValueError
        If the grid is not finite, non-negative, strictly increasing and
        uniform (relative spacing tolerance 1e-9) with at least three nodes,
        or if rate or vol is outside its domain.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((3, np.asarray(s_grid).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_bs_operator_bands(s_grid: np.ndarray, rate: float, vol: float) -> np.ndarray:
    import numpy as np

    s = np.asarray(s_grid, dtype=float)
    if s.ndim != 1 or s.size < 3 or not np.all(np.isfinite(s)) or s[0] < 0.0:
        raise ValueError("s_grid must be a finite, non-negative array of at least 3 nodes")
    d = np.diff(s)
    h = d[0]
    if h <= 0.0 or not np.allclose(d, h, rtol=1e-9, atol=0.0):
        raise ValueError("s_grid must be strictly increasing and uniformly spaced")
    if isinstance(rate, bool) or not np.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    if isinstance(vol, bool) or not np.isfinite(float(vol)) or float(vol) < 0.0:
        raise ValueError("vol must be finite and non-negative")
    r, b = float(rate), float(vol)

    diffusion = 0.5 * b * b * s ** 2 / h ** 2
    drift = r * s / (2.0 * h)
    lower = diffusion - drift
    diag = -2.0 * diffusion - r
    upper = diffusion + drift
    return np.vstack([lower, diag, upper]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: central differences are exact on quadratics, so on the
        # interior rows the bands applied to V = S^2 give (b^2 + r) S^2, the
        # generator applied to S^2; at S = 0 the row is exactly (0, -r, 0).
        # A short grid keeps the cancelling band terms small, so round-off
        # is not amplified.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 4.0, 5)
r, b = 0.05, 0.3
def apply(fn):
    B = fn(S, r, b)
    V = S ** 2
    interior = B[0, 1:-1] * V[:-2] + B[1, 1:-1] * V[1:-1] + B[2, 1:-1] * V[2:]
    return np.concatenate([B[:, 0], interior])
EXPECTED = np.concatenate([[0.0, -r, 0.0], (b * b + r) * S[1:-1] ** 2])
""",
            "call": "apply(bs_operator_bands)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: a pricing-size grid.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 400.0, 401)
""",
            "call": "bs_operator_bands(S, 0.05, 0.2)",
            "gold_call": "_oracle_bs_operator_bands(S, 0.05, 0.2)",
        },
        # --- Boundary: zero volatility, pure drift and discount.
        {
            "setup": """import numpy as np
S = np.linspace(10.0, 20.0, 6)
""",
            "call": "bs_operator_bands(S, 0.1, 0.0)",
            "gold_call": "_oracle_bs_operator_bands(S, 0.1, 0.0)",
        },
        # --- Edge: smallest grid, negative rate, large volatility.
        {
            "setup": """import numpy as np
S = np.array([0.0, 0.5, 1.0])
""",
            "call": "bs_operator_bands(S, -0.02, 1.5)",
            "gold_call": "_oracle_bs_operator_bands(S, -0.02, 1.5)",
        },
        # --- Invalid: non-uniform grid ---
        {
            "setup": """import numpy as np
S = np.array([0.0, 1.0, 2.5, 3.0])
def run_model():
    try:
        bs_operator_bands(S, 0.05, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bs_operator_bands(S, 0.05, 0.2)
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
def run_model():
    try:
        bs_operator_bands(S, 0.05, -0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bs_operator_bands(S, 0.05, -0.2)
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
