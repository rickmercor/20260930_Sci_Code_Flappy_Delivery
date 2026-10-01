"""
March the semi-discrete fractional system implicitly in time to maturity with the L1 scheme and return the terminal vector.

The unknown U(tau) solves the Caputo system D^alpha U = L U on the uniform grid tau_m = m delta, delta = maturity / n, starting from U(0) = initial_values. The scheme is implicit: at every level m = 1, ..., n the L1 approximation of the Caputo derivative at tau_m, which involves the whole history U^0, ..., U^m, is set equal to L U^m, and the first and last equations are replaced by the prescribed boundary values of that level. The full history enters at every level, which is the discrete memory of the model.

Returns
-------
np.ndarray, float, shape (N,): the solution U^n at tau = maturity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def l1_time_march(operator: np.ndarray, initial_values: np.ndarray, left_values: np.ndarray,
                  right_values: np.ndarray, alpha: float, maturity: float) -> np.ndarray:
    '''Implicit L1 time stepping of D^alpha U = L U with Dirichlet end values.

    Parameters
    ----------
    operator : np.ndarray
        Shape (N, N) spatial operator L (its first and last rows are ignored).
    initial_values : np.ndarray
        Shape (N,) solution at tau = 0.
    left_values : np.ndarray
        Shape (n + 1,) Dirichlet values of the first node at tau_0, ..., tau_n.
    right_values : np.ndarray
        Shape (n + 1,) Dirichlet values of the last node at tau_0, ..., tau_n.
    alpha : float
        Order of the Caputo derivative, 0 < alpha <= 1.
    maturity : float
        Final time, maturity > 0; the step is maturity / n.

    Returns
    -------
    terminal : np.ndarray
        Shape (N,) float array, the solution at tau = maturity.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n < 1, N < 3, any input is not
        finite, alpha is outside (0, 1], or maturity is not positive.

    Notes
    -----
    Obtain the history coefficients from ``l1_caputo_coefficients``. Include
    every import your implementation needs inside the function body.
    '''
    return np.zeros(np.asarray(initial_values).size, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_l1_time_march(operator: np.ndarray, initial_values: np.ndarray, left_values: np.ndarray,
                          right_values: np.ndarray, alpha: float, maturity: float) -> np.ndarray:
    import numpy as np

    op = np.asarray(operator, dtype=float)
    u0 = np.asarray(initial_values, dtype=float)
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    if u0.ndim != 1 or u0.size < 3:
        raise ValueError("initial_values must be one-dimensional with at least 3 entries")
    n_nodes = u0.size
    if op.shape != (n_nodes, n_nodes):
        raise ValueError("operator must have shape (N, N)")
    if left.ndim != 1 or left.shape != right.shape or left.size < 2:
        raise ValueError("boundary arrays must be one-dimensional of equal length n + 1 >= 2")
    for arr in (op, u0, left, right):
        if not np.all(np.isfinite(arr)):
            raise ValueError("inputs must be finite")
    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if isinstance(maturity, bool) or not np.isfinite(float(maturity)) or float(maturity) <= 0.0:
        raise ValueError("maturity must be positive")

    alpha = float(alpha)
    n_steps = left.size - 1
    delta = float(maturity) / n_steps
    scale = delta ** (-alpha)

    history = np.empty((n_steps + 1, n_nodes))
    history[0] = u0
    eye = np.eye(n_nodes)
    for m in range(1, n_steps + 1):
        coeff = _oracle_l1_caputo_coefficients(alpha, m)
        # Past levels U^{m-1}, ..., U^0 weighted by l_{1,m}, ..., l_{m,m}.
        rhs = -scale * (coeff[1:] @ history[m - 1::-1])
        system = scale * coeff[0] * eye - op
        system[0, :] = 0.0
        system[0, 0] = 1.0
        system[-1, :] = 0.0
        system[-1, -1] = 1.0
        rhs[0] = left[m]
        rhs[-1] = right[m]
        history[m] = np.linalg.solve(system, rhs)
    return history[n_steps].astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned limit: at alpha = 1 the L1 march is exactly the implicit
        # Euler scheme, computed here independently.
        {
            "setup": """import numpy as np
N, n, T = 9, 12, 0.6
L = np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
L = 3.0 * L - 0.1 * np.eye(N)
u0 = np.sin(np.linspace(0.0, np.pi, N)) + 0.5
left = 0.5 * np.exp(-np.linspace(0.0, T, n + 1))
right = 0.5 + np.linspace(0.0, T, n + 1)
def implicit_euler():
    dt = T / n
    u = u0.copy()
    for m in range(1, n + 1):
        A = np.eye(N) / dt - L
        b = u / dt
        A[0, :] = 0.0; A[0, 0] = 1.0; b[0] = left[m]
        A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = right[m]
        u = np.linalg.solve(A, b)
    return u
EXPECTED = implicit_euler()
""",
            "call": "l1_time_march(L, u0, left, right, 1.0, T)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned: with no spatial operator a constant state has zero Caputo
        # derivative, so it must be preserved exactly by the full memory sum.
        {
            "setup": """import numpy as np
N, n = 6, 30
L = np.zeros((N, N))
u0 = 2.5 * np.ones(N)
edge = 2.5 * np.ones(n + 1)
EXPECTED = np.round(u0, 10)
""",
            "call": "np.round(l1_time_march(L, u0, edge, edge, 0.4, 1.0), 10)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: a diffusive operator with memory order 0.6.
        {
            "setup": """import numpy as np
N, n, T = 9, 40, 1.0
L = 4.0 * (np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1))
u0 = np.maximum(np.linspace(-1.0, 1.0, N), 0.0)
left = np.zeros(n + 1)
right = 1.0 + 0.2 * np.linspace(0.0, T, n + 1)
""",
            "call": "l1_time_march(L, u0, left, right, 0.6, T)",
            "gold_call": "_oracle_l1_time_march(L, u0, left, right, 0.6, T)",
        },
        # --- Boundary: a single time step.
        {
            "setup": """import numpy as np
N = 5
L = -np.eye(N) + 0.3 * np.diag(np.ones(N - 1), 1)
u0 = np.arange(1.0, 6.0)
""",
            "call": "l1_time_march(L, u0, np.array([1.0, 0.9]), np.array([5.0, 4.0]), 0.7, 0.25)",
            "gold_call": "_oracle_l1_time_march(L, u0, np.array([1.0, 0.9]), np.array([5.0, 4.0]), 0.7, 0.25)",
        },
        # --- Edge: a weak memory order over a long horizon, where the history
        # weights decay slowly.
        {
            "setup": """import numpy as np
N, n = 7, 60
L = 0.8 * (np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1))
u0 = np.cos(np.linspace(0.0, 2.0, N))
left = np.linspace(1.0, 0.0, n + 1)
right = np.cos(2.0) * np.ones(n + 1)
""",
            "call": "l1_time_march(L, u0, left, right, 0.15, 3.0)",
            "gold_call": "_oracle_l1_time_march(L, u0, left, right, 0.15, 3.0)",
        },
        # --- Invalid: boundary arrays of different lengths ---
        {
            "setup": """import numpy as np
L = np.zeros((4, 4))
def run_model():
    try:
        l1_time_march(L, np.ones(4), np.ones(5), np.ones(6), 0.5, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_l1_time_march(L, np.ones(4), np.ones(5), np.ones(6), 0.5, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: order outside (0, 1] ---
        {
            "setup": """import numpy as np
L = np.zeros((4, 4))
def run_model():
    try:
        l1_time_march(L, np.ones(4), np.ones(5), np.ones(5), 0.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_l1_time_march(L, np.ones(4), np.ones(5), np.ones(5), 0.0, 1.0)
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
