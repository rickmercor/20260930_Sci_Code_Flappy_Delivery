"""
Implement rk4_integrate for a matrix-valued ordinary differential equation.

Advance the initial matrix A0 from t0 to t_end with the classical fixed-step four-stage Runge-Kutta method and return the matrix at the final time. The routine is used for both the short pre-propagation and the full-order reference trajectory; reject a nonpositive step size or a reversed interval with ValueError.

A fixed-step explicit Runge-Kutta integrator approximates a matrix ODE by combining four velocity evaluations within each time step. The same deterministic routine supplies the short pre-propagated state and the longer full-order reference trajectory, so its step count and endpoint handling must be consistent.

Returns
-------
np.ndarray as specified by the function Returns section.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rk4_integrate(A0: np.ndarray, rhs_fn, t0: float, t_end: float, h: float) -> np.ndarray:
    """Integrate a matrix ODE using the classical four-stage Runge-Kutta method.

    Parameters
    ----------
    A0 : np.ndarray, shape (n, n)
        Initial matrix state.
    rhs_fn : callable
        Right-hand side function F(A) returning an (n, n) array.
    t0 : float
        Start time.
    t_end : float
        End time.
    h : float
        Step size (must be strictly positive).

    Returns
    -------
    A : np.ndarray, shape (n, n)
        Matrix state at t_end.
    """
    A = A0.copy()
    return A

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rk4_integrate(A0: np.ndarray, rhs_fn, t0: float, t_end: float, h: float) -> np.ndarray:
    """Reference implementation."""
    A0 = np.asarray(A0, dtype=complex)
    if h <= 0:
        raise ValueError('h must be > 0')
    if t_end < t0:
        raise ValueError('t_end must be >= t0')
    n_steps = int(round((t_end - t0) / h))
    A = A0.copy()
    for _ in range(n_steps):
        k1 = rhs_fn(A)
        k2 = rhs_fn(A + 0.5 * h * k1)
        k3 = rhs_fn(A + 0.5 * h * k2)
        k4 = rhs_fn(A + h * k3)
        A = A + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return A

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Case 1
        {
            "setup": """import numpy as np
n = 3
c = 1j
A0 = np.eye(n, dtype=complex) * 2.0
rhs_fn = lambda A: c * A
t0, t_end, h = 0.0, 0.1, 1e-3
exact = np.exp(c * 0.1) * A0
def rel_err(A):
    return float(np.linalg.norm(A - exact) / np.linalg.norm(exact))
""",
            "call": "rel_err(rk4_integrate(A0, rhs_fn, t0, t_end, h)) < 1e-10",
            "gold_call": "rel_err(_oracle_rk4_integrate(A0, rhs_fn, t0, t_end, h)) < 1e-10",
        },
        # Case 2
        {
            "setup": """import numpy as np
n = 2
A0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
rhs_fn = lambda A: 1j * A
t0, t_end, h = 0.0, 0.01, 0.01
""",
            "call": "np.round(rk4_integrate(A0, rhs_fn, t0, t_end, h), 12).tolist()",
            "gold_call": "np.round(_oracle_rk4_integrate(A0, rhs_fn, t0, t_end, h), 12).tolist()",
        },
        # Case 3
        {
            "setup": """import numpy as np
n = 2
A0 = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=complex)
rhs_fn = lambda A: 0.5j * A
exact = np.exp(0.5j * 0.1) * A0
def err(h_val):
    A1 = rk4_integrate(A0, rhs_fn, 0.0, 0.1, h_val)
    return float(np.linalg.norm(A1 - exact))
def gold_err(h_val):
    A1 = _oracle_rk4_integrate(A0, rhs_fn, 0.0, 0.1, h_val)
    return float(np.linalg.norm(A1 - exact))
def check_order(e_fn):
    e1 = e_fn(0.01)
    e2 = e_fn(0.005)
    if e1 < 1e-15 or e2 < 1e-15:
        return True
    ratio = e1 / e2
    return 12.0 < ratio < 20.0
""",
            "call": "check_order(err)",
            "gold_call": "check_order(gold_err)",
        },
        # Case 4
        {
            "setup": """import numpy as np
n = 4
np.random.seed(99)
A0 = (np.random.randn(n, n) + 1j * np.random.randn(n, n))
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.0
rhs_fn = lambda A: 0.5j * (B @ A + A @ B)
t0, t_end, h = 0.0, 0.05, 1e-3
def norm_change(A_final):
    return abs(np.linalg.norm(A_final) - np.linalg.norm(A0))
""",
            "call": "norm_change(rk4_integrate(A0, rhs_fn, t0, t_end, h)) < 1e-8",
            "gold_call": "norm_change(_oracle_rk4_integrate(A0, rhs_fn, t0, t_end, h)) < 1e-8",
        },
        # Case 5
        {
            "setup": """import numpy as np
n = 4
A0 = np.ones((n, n), dtype=complex) * 0.5
B = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
alpha = 0.5
rhs_fn = lambda A: 0.5j * (B @ A + A @ B) + 1j * alpha * (A * A * A)
t0, t_end, h = 0.0, 0.01, 1e-3
def output_norm(A):
    return round(float(np.linalg.norm(A)), 10)
""",
            "call": "output_norm(rk4_integrate(A0, rhs_fn, t0, t_end, h))",
            "gold_call": "output_norm(_oracle_rk4_integrate(A0, rhs_fn, t0, t_end, h))",
        },
        # Case 6
        {
            "setup": """import numpy as np
A0 = np.eye(2, dtype=complex)
rhs_fn = lambda A: 1j * A
def run_model():
    try:
        rk4_integrate(A0, rhs_fn, 0.0, 1.0, -0.01)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rk4_integrate(A0, rhs_fn, 0.0, 1.0, -0.01)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 7
        {
            "setup": """import numpy as np
A0 = np.eye(2, dtype=complex)
rhs_fn = lambda A: 1j * A
def run_model():
    try:
        rk4_integrate(A0, rhs_fn, 1.0, 0.5, 0.01)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rk4_integrate(A0, rhs_fn, 1.0, 0.5, 0.01)
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
call_count = [0]
A0 = np.eye(2, dtype=complex)
def counting_rhs(A):
    call_count[0] += 1
    return 0.01j * A
def count_calls(integrate_fn):
    call_count[0] = 0
    integrate_fn(A0, counting_rhs, 0.0, 0.1, 0.01)
    return call_count[0]
""",
            "call": "count_calls(rk4_integrate)",
            "gold_call": "count_calls(_oracle_rk4_integrate)",
        },
    ]
