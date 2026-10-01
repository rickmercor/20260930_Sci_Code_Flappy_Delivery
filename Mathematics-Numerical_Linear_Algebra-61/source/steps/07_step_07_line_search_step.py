"""
Advance $x$ by an exact line search along the nonzero direction $p$, using the complementary residual as seen through $S$. Require $S$ of shape $(m_r, q)$.

The line search is taken along the current direction. A zero direction is a null step and is rejected.

Returns
-------
ndarray of shape (n,): updated iterate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def line_search_step(
    A_r: np.ndarray, b_r: np.ndarray, x: np.ndarray, S: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """Return the iterate after an exact line search along $p$.

    Parameters
    ----------
    A_r : np.ndarray
        Complementary rows, shape $(m_r, n)$.
    b_r : np.ndarray
        Complementary right-hand side, length $m_r$.
    x : np.ndarray
        Current iterate, length $n$.
    S : np.ndarray
        Sketch of shape $(m_r, q)$, $q \ge 1$.
    p : np.ndarray
        Nonzero search direction, length $n$.

    Returns
    -------
    x_new : np.ndarray
        Updated iterate of length $n$.

    Raises
    ------
    ValueError
        If $A_r$ is not a nonempty 2d array, if $b_r$, $x$, or $p$ has an
        incompatible shape, if $S$ does not have shape $(m_r, q)$ with
        $q \ge 1$, or if the search direction is zero.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_line_search_step(A_r, b_r, x, S, p):
    A_r = np.asarray(A_r, dtype=float)
    if A_r.ndim != 2 or min(A_r.shape) < 1:
        raise ValueError("A_r must be a nonempty 2d array")
    m_r, n = A_r.shape
    b_r = np.asarray(b_r, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if b_r.shape != (m_r,) or x.shape != (n,) or p.shape != (n,):
        raise ValueError("incompatible shapes")
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != m_r or S.shape[1] < 1:
        raise ValueError("S must have shape (m_r, q) with q >= 1")
    nrm2 = float(np.dot(p, p))
    if nrm2 <= 0.0:
        raise ValueError("search direction must be nonzero")
    r = A_r @ x - b_r
    Sr = S.T @ r
    delta = float(np.dot(Sr, Sr) / nrm2)
    return x + delta * p

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
A_r = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b_r = np.array([1.0, 2.0, 0.0])
x = np.zeros(2)
S = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
p = np.array([1.0, -0.5])
""",
            "call": "line_search_step(A_r, b_r, x, S, p)",
            "gold_call": "_oracle_line_search_step(A_r, b_r, x, S, p)",
        },
        {
            "setup": """import numpy as np
A_r = np.array([[2.0, 1.0]])
b_r = np.array([1.0])
x = np.array([0.0, 0.0])
S = np.array([[1.0]])
p = np.array([0.0, 1.0])
""",
            "call": "line_search_step(A_r, b_r, x, S, p)",
            "gold_call": "_oracle_line_search_step(A_r, b_r, x, S, p)",
        },
        {
            "setup": """import numpy as np
A_r = np.eye(3)
b_r = np.array([1.0, -1.0, 0.5])
x = np.array([0.2, 0.1, -0.3])
S = np.ones((3, 2))
p = np.array([0.4, 0.0, -0.1])
""",
            "call": "line_search_step(A_r, b_r, x, S, p)",
            "gold_call": "_oracle_line_search_step(A_r, b_r, x, S, p)",
        },
        {
            "setup": """import numpy as np
A_r = np.eye(2)
b_r = np.zeros(2)
x = np.zeros(2)
S = np.eye(2)
p = np.zeros(2)
def run_model():
    try:
        line_search_step(A_r, b_r, x, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_line_search_step(A_r, b_r, x, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A_r = np.eye(2)
b_r = np.zeros(2)
x = np.zeros(2)
S = np.ones((3, 1))
p = np.array([1.0, 0.0])
def run_model():
    try:
        line_search_step(A_r, b_r, x, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_line_search_step(A_r, b_r, x, S, p)
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
