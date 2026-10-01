"""
Take one fully implicit time step of a linear system defined by three-band generator coefficients, with prescribed values at both end nodes.

Given the bands of a generator A, a step length dt and a right-hand side, the step returns the vector x that satisfies (I - dt A) x = rhs on every interior row, while its first and last entries equal the prescribed end values. This single backward Euler step is the building block of both backward recursions in the task: the American put and the stopped drift integral.

Returns
-------
np.ndarray, float, shape (n,): the solution x.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def implicit_step(bands: np.ndarray, dt: float, rhs: np.ndarray, left_value: float,
                  right_value: float) -> np.ndarray:
    '''One backward Euler step with Dirichlet end values.

    Parameters
    ----------
    bands : np.ndarray
        Shape (3, n) generator coefficients (sub-diagonal, diagonal,
        super-diagonal per row), n >= 3.
    dt : float
        Step length, dt >= 0.
    rhs : np.ndarray
        Shape (n,) right-hand side.
    left_value : float
        Value imposed on the first entry.
    right_value : float
        Value imposed on the last entry.

    Returns
    -------
    x : np.ndarray
        Shape (n,) float solution.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n < 3, any input is not finite, or
        dt is negative.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros(np.asarray(rhs).size, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_implicit_step(bands: np.ndarray, dt: float, rhs: np.ndarray, left_value: float,
                          right_value: float) -> np.ndarray:
    import numpy as np
    from scipy.linalg import solve_banded

    B = np.asarray(bands, dtype=float)
    f = np.asarray(rhs, dtype=float)
    if f.ndim != 1 or f.size < 3 or B.shape != (3, f.size):
        raise ValueError("bands must have shape (3, n) and rhs shape (n,), n >= 3")
    for name, value in (("dt", dt), ("left_value", left_value), ("right_value", right_value)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(dt) < 0.0:
        raise ValueError("dt must be non-negative")
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(f))):
        raise ValueError("bands and rhs must be finite")
    dt = float(dt)
    n = f.size

    # Banded storage for solve_banded((1, 1), ...): row 0 super, 1 diag, 2 sub.
    ab = np.zeros((3, n))
    ab[0, 1:] = -dt * B[2, :-1]
    ab[1, :] = 1.0 - dt * B[1, :]
    ab[2, :-1] = -dt * B[0, 1:]
    # Dirichlet rows.
    ab[1, 0] = 1.0
    ab[0, 1] = 0.0
    ab[1, -1] = 1.0
    ab[2, -2] = 0.0
    x = f.copy()
    x[0] = float(left_value)
    x[-1] = float(right_value)
    return solve_banded((1, 1), ab, x).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: with dt = 0 the step returns the right-hand side with its
        # end entries replaced by the prescribed values.
        {
            "setup": """import numpy as np
B = np.arange(15.0).reshape(3, 5)
rhs = np.array([9.0, 1.0, 2.0, 3.0, 9.0])
EXPECTED = np.array([-4.0, 1.0, 2.0, 3.0, 7.0])
""",
            "call": "implicit_step(B, 0.0, rhs, -4.0, 7.0)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned: an independent dense solve of the same system.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3)
n = 9
B = rng.normal(size=(3, n))
B[1] -= 4.0
rhs = rng.normal(size=n)
dt = 0.3
def dense(Bm, dt, f, a, z):
    M = np.eye(n)
    for i in range(1, n - 1):
        M[i, i - 1] -= dt * Bm[0, i]; M[i, i] -= dt * Bm[1, i]; M[i, i + 1] -= dt * Bm[2, i]
    g = f.copy(); g[0], g[-1] = a, z
    return np.linalg.solve(M, g)
EXPECTED = dense(B, dt, rhs, 1.5, -0.5)
""",
            "call": "implicit_step(B, dt, rhs, 1.5, -0.5)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: Black-Scholes-like bands on a larger grid.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 200.0, 101)
h = S[1] - S[0]
a = 0.5 * 0.04 * S ** 2 / h ** 2; c = 0.05 * S / (2 * h)
B = np.vstack([a - c, -2 * a - 0.05, a + c])
rhs = np.maximum(100.0 - S, 0.0)
""",
            "call": "implicit_step(B, 0.01, rhs, 100.0, 0.0)",
            "gold_call": "_oracle_implicit_step(B, 0.01, rhs, 100.0, 0.0)",
        },
        # --- Boundary: the smallest system, one interior unknown.
        {
            "setup": """import numpy as np
B = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0], [7.0, 8.0, 9.0]])
""",
            "call": "implicit_step(B, 0.5, np.array([0.0, 1.0, 0.0]), 2.0, 3.0)",
            "gold_call": "_oracle_implicit_step(B, 0.5, np.array([0.0, 1.0, 0.0]), 2.0, 3.0)",
        },
        # --- Edge: a very large step, where the diagonal dominates.
        {
            "setup": """import numpy as np
B = np.vstack([np.ones(7), -3.0 * np.ones(7), np.ones(7)])
rhs = np.linspace(1.0, 2.0, 7)
""",
            "call": "implicit_step(B, 1e4, rhs, 0.0, 0.0)",
            "gold_call": "_oracle_implicit_step(B, 1e4, rhs, 0.0, 0.0)",
        },
        # --- Invalid: negative step ---
        {
            "setup": """import numpy as np
B = np.zeros((3, 4))
def run_model():
    try:
        implicit_step(B, -0.1, np.ones(4), 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_implicit_step(B, -0.1, np.ones(4), 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: bands and right-hand side of different lengths ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        implicit_step(np.zeros((3, 5)), 0.1, np.ones(4), 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_implicit_step(np.zeros((3, 5)), 0.1, np.ones(4), 0.0, 0.0)
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
