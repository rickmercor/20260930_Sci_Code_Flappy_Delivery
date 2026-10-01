"""
Measure how far a symmetric iterate is from a projector by evaluating the trace of the difference between the matrix and its square.

The trace of X - X**2 is the sum over the eigenvalues of x - x**2, so it vanishes exactly when every eigenvalue has been driven to 0 or 1 and is the quantity used to monitor the purification phase of a recursive expansion. For a symmetric argument the trace of the square is the sum of the squares of all entries, so the measure is available without spending a further matrix-matrix multiplication, which is the currency the whole scheme is budgeted in.

Returns
-------
float: the trace of the matrix minus the trace of its square, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def idempotency_trace(matrix: np.ndarray) -> float:
    """Evaluate the trace of the difference between a matrix and its square.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (n, n) square symmetric float array.

    Returns
    -------
    measure : float
        The trace of ``matrix`` minus the trace of its square, as a native
        Python float.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square two-dimensional array, or if
        any of its entries is not finite. The function must raise rather
        than return a placeholder or a NaN.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_idempotency_trace(matrix: np.ndarray) -> float:
    import numpy as np

    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square two-dimensional array")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix must be finite")

    # For a symmetric argument the trace of the square is the squared Frobenius
    # norm, so no matrix-matrix multiplication is needed here.
    return float(np.trace(a) - np.sum(a * a))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values obtained independently of the oracle: for a diagonal
        # matrix the measure is the elementwise sum of x - x**2, which needs no
        # reference implementation.
        {
            "setup": """import numpy as np
d = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9])
a = np.diag(d)
EXPECTED = float(np.sum(d - d ** 2))
""",
            "call": "float(idempotency_trace(a))",
            "gold_call": "EXPECTED",
        },
        # --- Pinned: for a symmetric matrix the measure depends only on the
        # spectrum, so a conjugated diagonal must return the same number.
        {
            "setup": """import numpy as np
d = np.array([0.02, 0.31, 0.48, 0.66, 0.83, 0.97])
n = d.shape[0]
idx = np.arange(1, n + 1, dtype=float)
q = np.sqrt(2.0 / (n + 1.0)) * np.sin(np.pi * np.outer(idx, idx) / (n + 1.0))
a = q @ np.diag(d) @ q
EXPECTED = float(np.sum(d - d ** 2))
""",
            "call": "float(idempotency_trace(a))",
            "gold_call": "EXPECTED",
        },
        # --- Valid: a dense symmetric iterate of moderate size (normal
        # scenario) ---
        {
            "setup": """import numpy as np
n = 40
idx = np.arange(n)
a = 0.03 * np.cos(0.4 * np.add.outer(idx, idx)) + np.diag(np.linspace(0.05, 0.95, n))
a = 0.5 * (a + a.T)
""",
            "call": "float(idempotency_trace(a))",
            "gold_call": "float(_oracle_idempotency_trace(a))",
        },
        # --- Boundary: an exact projector, where the measure must vanish ---
        {
            "setup": """import numpy as np
v = np.array([0.6, 0.0, -0.8])
a = np.outer(v, v)
""",
            "call": "float(1e12 * idempotency_trace(a))",
            "gold_call": "float(1e12 * _oracle_idempotency_trace(a))",
        },
        # --- Boundary: eigenvalues outside the unit interval drive the measure
        # negative ---
        {
            "setup": """import numpy as np
a = np.diag(np.array([1.4, -0.2, 0.5]))
""",
            "call": "float(idempotency_trace(a))",
            "gold_call": "float(_oracle_idempotency_trace(a))",
        },
        # --- Edge: a one-by-one matrix ---
        {
            "setup": """import numpy as np
a = np.array([[0.36]])
""",
            "call": "float(idempotency_trace(a))",
            "gold_call": "float(_oracle_idempotency_trace(a))",
        },
        # --- Invalid: a rectangular argument ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        idempotency_trace(np.ones((2, 5)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_idempotency_trace(np.ones((2, 5)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-finite entry ---
        {
            "setup": """import numpy as np
bad = np.eye(3); bad[1, 1] = np.inf
def run_model():
    try:
        idempotency_trace(bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_idempotency_trace(bad)
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
