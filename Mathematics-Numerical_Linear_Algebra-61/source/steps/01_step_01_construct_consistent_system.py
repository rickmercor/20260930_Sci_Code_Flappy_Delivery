"""
A two-cluster singular spectrum is the regime in which an unconstrained sketched gradient mixes a well-conditioned row block with an ill-conditioned complement. This step manufactures a reproducible overdetermined consistent instance so later constrained sketches see a known deterministic pair (A, b).

A two-cluster singular spectrum is the regime in which an unconstrained sketched gradient mixes a well-conditioned row block with an ill-conditioned complement. This step manufactures a reproducible overdetermined consistent instance so later constrained sketches see a known deterministic pair (A, b).

Returns
-------
1d ndarray: packed (m, n, A.ravel(), b)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def construct_consistent_system(m: int, n: int, s: np.ndarray, seed: int) -> np.ndarray:
    """Pack $A$ of shape $(m, n)$ and a consistent right-hand side $b$.

    Parameters
    ----------
    m : int
        Row dimension, $m > n \ge 2$.
    n : int
        Column dimension.
    s : np.ndarray
        Positive singular values, shape $(n,)$.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    pack : np.ndarray
        Concatenation of $(m, n)$, $A.\mathrm{ravel}()$, and $b$.

    Raises
    ------
    ValueError
        If $m$ or $n$ is not an integer, if not $m > n \ge 2$, if $s$ does
        not have shape $(n,)$, or if $s$ is not positive and finite.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_construct_consistent_system(m, n, s, seed):
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("m and n must be integers")
    m, n = int(m), int(n)
    if n < 2 or m <= n:
        raise ValueError("require m > n >= 2")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("s must be positive and finite")
    rng = np.random.default_rng(int(seed))
    U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    A = U @ np.diag(s) @ V.T
    x_star = rng.standard_normal(n)
    b = A @ x_star
    return np.concatenate(([float(m), float(n)], A.ravel(), b))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, seed = 12, 8, 7
s = np.array([8.0, 6.0, 4.5, 1.1, 0.8, 0.55, 0.4, 0.3])
""",
            "call": "construct_consistent_system(m, n, s, seed)",
            "gold_call": "_oracle_construct_consistent_system(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 5, 3, 1
s = np.array([3.0, 1.2, 0.4])
""",
            "call": "construct_consistent_system(m, n, s, seed)",
            "gold_call": "_oracle_construct_consistent_system(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 3, 2, 0
s = np.array([2.0, 0.5])
""",
            "call": "construct_consistent_system(m, n, s, seed)",
            "gold_call": "_oracle_construct_consistent_system(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
s = np.array([-1.0, 0.5])
def run_model():
    try:
        construct_consistent_system(3, 2, s, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_consistent_system(3, 2, s, 0)
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
s = np.array([2.0, 1.0])
def run_model():
    try:
        construct_consistent_system(2, 2, s, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_consistent_system(2, 2, s, 0)
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
