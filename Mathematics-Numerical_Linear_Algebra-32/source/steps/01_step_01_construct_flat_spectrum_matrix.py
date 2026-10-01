"""
Build A = U diag(s) V^T from thin QR factors of Gaussians drawn by default_rng(seed). The singular values s are prescribed and positive. Require m >= n >= 1.

A slowly decaying singular spectrum is the regime where a short rangefinder mixes dominant and tail directions. This step manufactures a reproducible rectangular instance with that spectrum so later one-pass sketches see a known deterministic matrix.

Returns
-------
ndarray of shape (m, n): matrix A with singular values s
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_flat_spectrum_matrix(m: int, n: int, s: np.ndarray, seed: int) -> np.ndarray:
    """Build A of shape (m, n) with singular values s.

    Parameters
    ----------
    m : int
        Row dimension, m >= n >= 1.
    n : int
        Column dimension.
    s : np.ndarray
        Positive singular values, shape (n,).
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    A : np.ndarray
        Array of shape (m, n).

    Raises
    ------
    ValueError
        If `m` or `n` is not an integer, if `m >= n >= 1` does not hold,
        if `s` does not have shape `(n,)`, or if `s` is not positive and
        finite.
    """
    return np.zeros((m, n))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_construct_flat_spectrum_matrix(m, n, s, seed):
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("m and n must be integers")
    m, n = int(m), int(n)
    if m < 1 or n < 1 or m < n:
        raise ValueError("require m >= n >= 1")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("s must be positive and finite")
    rng = np.random.default_rng(int(seed))
    U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    return U @ np.diag(s) @ V.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, seed = 12, 10, 7
s = np.array([4.0, 3.2, 2.6, 2.1, 1.7, 1.4, 1.15, 0.95, 0.8, 0.7])
""",
            "call": "construct_flat_spectrum_matrix(m, n, s, seed)",
            "gold_call": "_oracle_construct_flat_spectrum_matrix(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 6, 4, 1
s = np.array([5.0, 2.0, 0.8, 0.3])
""",
            "call": "construct_flat_spectrum_matrix(m, n, s, seed)",
            "gold_call": "_oracle_construct_flat_spectrum_matrix(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 3, 3, 2
s = np.array([4.0, 1.0, 0.25])
""",
            "call": "construct_flat_spectrum_matrix(m, n, s, seed)",
            "gold_call": "_oracle_construct_flat_spectrum_matrix(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 4, 1, 3
s = np.array([2.5])
""",
            "call": "construct_flat_spectrum_matrix(m, n, s, seed)",
            "gold_call": "_oracle_construct_flat_spectrum_matrix(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
s = np.array([1.0, -0.1])
def run_model():
    try:
        construct_flat_spectrum_matrix(2, 2, s, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_flat_spectrum_matrix(2, 2, s, 0)
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
s = np.array([1.0, 0.5, 0.2])
def run_model():
    try:
        construct_flat_spectrum_matrix(2, 3, s, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_flat_spectrum_matrix(2, 3, s, 0)
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
