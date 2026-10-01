"""
Build a packed SPD eigenproblem together with one Gaussian sketch. Form A from a thin QR of Gaussians drawn by default_rng(data_seed), draw b from that generator, then draw Omega from default_rng(sketch_seed). Pack (n, d, A.ravel(), b, Omega.ravel()). Require n >= 2, d >= 1, and positive s.

A two-cluster spectrum is the regime in which a cheap sketched Hessenberg can return non-Ritz eigenvalues. The sketch is drawn once and reused; a second embedding is a different algorithm.

Returns
-------
1d ndarray: packed (n, d, A.ravel(), b, Omega.ravel())
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_sketched_instance(
    n: int, d: int, s: np.ndarray, data_seed: int, sketch_seed: int
) -> np.ndarray:
    """Pack A, b, and Omega for a clustered-spectrum eigenproblem.

    Parameters
    ----------
    n : int
        Dimension, n >= 2.
    d : int
        Sketch dimension, d >= 1.
    s : np.ndarray
        Positive eigenvalues, shape (n,).
    data_seed : int
        RNG seed for the SPD factor draw and start vector.
    sketch_seed : int
        Independent RNG seed for the sketch.

    Returns
    -------
    pack : np.ndarray
        Concatenation of (n, d, A.ravel(), b, Omega.ravel()).

    Raises
    ------
    ValueError
        If n or d is not an integer, if n < 2 or d < 1, if s does not
        have shape (n,), or if any entry of s is nonpositive or nonfinite.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_sketched_instance(n, d, s, data_seed, sketch_seed):
    if not isinstance(n, (int, np.integer)) or not isinstance(d, (int, np.integer)):
        raise ValueError("n and d must be integers")
    n, d = int(n), int(d)
    if n < 2 or d < 1:
        raise ValueError("require n >= 2 and d >= 1")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("s must be positive and finite")
    rng = np.random.default_rng(int(data_seed))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    A = (Q * s) @ Q.T
    A = 0.5 * (A + A.T)
    b = rng.standard_normal(n)
    Omega = np.random.default_rng(int(sketch_seed)).standard_normal((d, n))
    return np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, d, data_seed, sketch_seed = 12, 8, 7, 11
s = np.array([8.0, 6.0, 5.0, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15])
""",
            "call": "assemble_sketched_instance(n, d, s, data_seed, sketch_seed)",
            "gold_call": "_oracle_assemble_sketched_instance(n, d, s, data_seed, sketch_seed)",
        },
        {
            "setup": """import numpy as np
n, d, data_seed, sketch_seed = 4, 3, 1, 2
s = np.array([3.0, 2.0, 0.4, 0.2])
""",
            "call": "assemble_sketched_instance(n, d, s, data_seed, sketch_seed)",
            "gold_call": "_oracle_assemble_sketched_instance(n, d, s, data_seed, sketch_seed)",
        },
        {
            "setup": """import numpy as np
s = np.array([1.0, -0.1])
def run_model():
    try:
        assemble_sketched_instance(2, 2, s, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_sketched_instance(2, 2, s, 0, 1)
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
s = np.array([1.0, 0.5])
def run_model():
    try:
        assemble_sketched_instance(2, 0, s, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_sketched_instance(2, 0, s, 0, 1)
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
s = np.array([1.0])
def run_model():
    try:
        assemble_sketched_instance(2, 2, s, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_sketched_instance(2, 2, s, 0, 1)
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
