"""
Draw a standard_normal sketch X of shape (t, n) from default_rng(seed). Require 1 <= t < n.

A short two-sided embedding reduces the middle-matrix least-squares problem from n-by-n to t-by-t. Drawing it from a seeded generator makes the later core deterministic.

Returns
-------
ndarray of shape (t, n): Gaussian sketch X
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def draw_gaussian_sketch(t: int, n: int, seed: int) -> np.ndarray:
    """Draw X of shape (t, n) with standard_normal entries.

    Parameters
    ----------
    t : int
        Sketch height, 1 <= t < n.
    n : int
        Sketch width (ambient dimension).
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    X : np.ndarray
        Array of shape (t, n).

    Raises
    ------
    ValueError
        If t or n is not an integer, if n < 2, or if t is not in
        1 <= t < n.
    """
    return np.zeros((t, n))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_draw_gaussian_sketch(t, n, seed):
    if not isinstance(t, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("t and n must be integers")
    t, n = int(t), int(n)
    if n < 2:
        raise ValueError("require n >= 2")
    if t < 1 or t >= n:
        raise ValueError("require 1 <= t < n")
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((t, n))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
t, n, seed = 6, 12, 11
""",
            "call": "draw_gaussian_sketch(t, n, seed)",
            "gold_call": "_oracle_draw_gaussian_sketch(t, n, seed)",
        },
        {
            "setup": """import numpy as np
t, n, seed = 3, 5, 1
""",
            "call": "draw_gaussian_sketch(t, n, seed)",
            "gold_call": "_oracle_draw_gaussian_sketch(t, n, seed)",
        },
        {
            "setup": """import numpy as np
t, n, seed = 1, 2, 4
""",
            "call": "draw_gaussian_sketch(t, n, seed)",
            "gold_call": "_oracle_draw_gaussian_sketch(t, n, seed)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        draw_gaussian_sketch(4, 4, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_draw_gaussian_sketch(4, 4, 0)
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
def run_model():
    try:
        draw_gaussian_sketch(0, 5, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_draw_gaussian_sketch(0, 5, 0)
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
