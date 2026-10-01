"""
Compute, on the asset grid at time 0, the expected discounted integral of given drift-rate functions accumulated until the stopping time defined by the exercise regions, or until maturity.

For a drift-rate function h of the asset price, W(t, s) is the expected value, starting from S_t = s, of the integral from t to the stopping time (capped at maturity) of e^{-r(u-t)} h(S_u) du, where the path is stopped the first time it stands on an exercise node. W solves the discounted Black-Scholes equation with source h on the continuation set and vanishes where the path is stopped and at maturity. It is computed backward on the same uniform time levels as the exercise regions: each level is one fully implicit step of that equation with W = 0 at both end nodes, after which W is set to zero on that level's exercise nodes. Several source rows can be integrated at once, with the same stopping rule.

Returns
-------
np.ndarray, float, shape (k, n): W at time 0 on the grid for each of the k source rows.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def stopped_drift_integral(s_grid: np.ndarray, sources: np.ndarray, exercise: np.ndarray,
                           rate: float, vol: float, maturity: float) -> np.ndarray:
    '''Expected discounted drift integral up to the stopping time, at time 0.

    Parameters
    ----------
    s_grid : np.ndarray
        Uniform grid of n >= 3 prices.
    sources : np.ndarray
        Shape (k, n) drift-rate values on the grid, k >= 1 (a 1-D array is
        treated as a single row).
    exercise : np.ndarray
        Shape (n_steps, n) exercise indicators of the levels t_0, ...,
        t_{n_steps - 1}; entries are 0 or 1.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b >= 0.
    maturity : float
        Maturity T > 0; the step is maturity / n_steps.

    Returns
    -------
    values : np.ndarray
        Shape (k, n) float array.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, the indicators are not 0 or 1, any
        input is not finite, or any parameter is outside its domain.

    Notes
    -----
    Build the steps from ``bs_operator_bands`` and ``implicit_step``. Include
    every import your implementation needs inside the function body.
    '''
    return np.zeros(np.atleast_2d(sources).shape, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_stopped_drift_integral(s_grid: np.ndarray, sources: np.ndarray, exercise: np.ndarray,
                                   rate: float, vol: float, maturity: float) -> np.ndarray:
    import numpy as np

    s = np.asarray(s_grid, dtype=float)
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    ex = np.asarray(exercise, dtype=float)
    if s.ndim != 1 or s.size < 3:
        raise ValueError("s_grid must be one-dimensional with at least 3 nodes")
    n = s.size
    if src.ndim != 2 or src.shape[1] != n or src.shape[0] < 1:
        raise ValueError("sources must have shape (k, n)")
    if ex.ndim != 2 or ex.shape[1] != n or ex.shape[0] < 1:
        raise ValueError("exercise must have shape (n_steps, n)")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(ex))):
        raise ValueError("sources and exercise must be finite")
    if not np.all((ex == 0.0) | (ex == 1.0)):
        raise ValueError("exercise indicators must be 0 or 1")
    if isinstance(maturity, bool) or not np.isfinite(float(maturity)) or float(maturity) <= 0.0:
        raise ValueError("maturity must be a finite positive number")
    bands = _oracle_bs_operator_bands(s, rate, vol)   # validates the grid, rate and vol
    m = ex.shape[0]
    dt = float(maturity) / m
    stopped = ex > 0.5

    out = np.empty_like(src)
    for j in range(src.shape[0]):
        w = np.zeros(n)
        for level in range(m - 1, -1, -1):
            w = _oracle_implicit_step(bands, dt, w + dt * src[j], 0.0, 0.0)
            w[stopped[level]] = 0.0
        out[j] = w
    return out

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: if every node is an exercise node at every level, the
        # path is stopped at once and nothing accumulates.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 100.0, 21)
src = np.vstack([np.ones(21), S])
EXPECTED = np.zeros((2, 21))
""",
            "call": "stopped_drift_integral(S, src, np.ones((6, 21)), 0.05, 0.2, 1.0)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned: no exercise at all, checked against an independent dense
        # implementation of the same backward recursion.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 50.0, 26)
n, m, T, r, b = 26, 8, 0.8, 0.04, 0.3
h = S[1] - S[0]
src = np.exp(-(S - 25.0) ** 2 / 20.0) - 0.1
def dense():
    a = 0.5 * b * b * S ** 2 / h ** 2; c = r * S / (2 * h)
    A = np.zeros((n, n))
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = a[i] - c[i], -2 * a[i] - r, a[i] + c[i]
    M = np.eye(n) - (T / m) * A
    M[0, :] = 0.0; M[0, 0] = 1.0; M[-1, :] = 0.0; M[-1, -1] = 1.0
    w = np.zeros(n)
    for _ in range(m):
        g = w + (T / m) * src; g[0] = 0.0; g[-1] = 0.0
        w = np.linalg.solve(M, g)
    return w[None, :]
EXPECTED = dense()
""",
            "call": "stopped_drift_integral(S, src, np.zeros((m, n)), r, b, T)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: a put-like exercise region shrinking toward maturity.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 200.0, 81)
m = 20
ex = np.zeros((m, 81))
for k in range(m):
    ex[k, 1:int(30 + k)] = 1.0
src = np.vstack([-5.0 * (S < 100.0), np.exp(-(S - 100.0) ** 2 / 2.0)])
""",
            "call": "stopped_drift_integral(S, src, ex, 0.05, 0.2, 1.0)",
            "gold_call": "_oracle_stopped_drift_integral(S, src, ex, 0.05, 0.2, 1.0)",
        },
        # --- Boundary: a single level and a one-dimensional source.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 10.0, 11)
ex = np.zeros((1, 11)); ex[0, :3] = 1.0
""",
            "call": "stopped_drift_integral(S, S * 0.1, ex, 0.02, 0.5, 0.25)",
            "gold_call": "_oracle_stopped_drift_integral(S, S * 0.1, ex, 0.02, 0.5, 0.25)",
        },
        # --- Edge: zero volatility, where the equation is pure transport.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 40.0, 41)
ex = np.zeros((15, 41)); ex[:, 30:] = 1.0
""",
            "call": "stopped_drift_integral(S, np.ones(41), ex, 0.1, 0.0, 3.0)",
            "gold_call": "_oracle_stopped_drift_integral(S, np.ones(41), ex, 0.1, 0.0, 3.0)",
        },
        # --- Invalid: indicator values other than 0 and 1 ---
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 10.0, 6)
def run_model():
    try:
        stopped_drift_integral(S, np.ones(6), 0.5 * np.ones((3, 6)), 0.05, 0.2, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_stopped_drift_integral(S, np.ones(6), 0.5 * np.ones((3, 6)), 0.05, 0.2, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: source length does not match the grid ---
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 10.0, 6)
def run_model():
    try:
        stopped_drift_integral(S, np.ones(5), np.zeros((3, 6)), 0.05, 0.2, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_stopped_drift_integral(S, np.ones(5), np.zeros((3, 6)), 0.05, 0.2, 1.0)
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
