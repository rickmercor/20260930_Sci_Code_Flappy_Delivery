"""
Construct the current residual quantities and the sampled residual-weighted sketch.

The normal-equation residual is formed from the current residual, and the sampled columns are combined using the corresponding residual components.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray] containing e, r, and s
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def form_residual_sketch(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the current residual, normal-equation residual, and sketch.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix.
    b : np.ndarray
        Right-hand-side vector.
    x : np.ndarray
        Current iterate.
    tau : np.ndarray
        Distinct sampled column indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The residual, normal-equation residual, and sampled sketch.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes, tau is empty, tau contains
        non-integer or invalid indices, tau contains duplicates, or any input
        contains non-finite values.

    Notes
    -----
    The result is deterministic for identical inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_form_residual_sketch(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    tau = np.asarray(tau)

    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")

    m, n = A.shape

    if b.shape != (m,):
        raise ValueError("b has incompatible shape")

    if x.shape != (n,):
        raise ValueError("x has incompatible shape")

    if tau.ndim != 1 or tau.size < 1:
        raise ValueError("tau must be one-dimensional and nonempty")

    if not np.issubdtype(tau.dtype, np.integer):
        raise ValueError("tau must contain integer indices")

    if np.any(tau < 0) or np.any(tau >= n):
        raise ValueError("tau contains an invalid index")

    if np.unique(tau).size != tau.size:
        raise ValueError("tau must contain distinct indices")

    if not (
        np.all(np.isfinite(A))
        and np.all(np.isfinite(b))
        and np.all(np.isfinite(x))
    ):
        raise ValueError("inputs must be finite")

    e = b - A @ x
    r = A.T @ e
    s = A[:, tau] @ r[tau]

    return (
        np.asarray(e, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(s, dtype=np.float64),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
A = np.array(
    [[1.0, 2.0, 0.5],
     [0.0, 1.0, 3.0],
     [2.0, -1.0, 1.0]],
    dtype=np.float64,
)
b = np.array([1.0, 2.0, -1.0], dtype=np.float64)
x = np.array([0.2, -0.1, 0.3], dtype=np.float64)
tau = np.array([0, 2], dtype=np.int64)
""",
            "call": "form_residual_sketch(A, b, x, tau)",
            "gold_call": "_oracle_form_residual_sketch(A, b, x, tau)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0]], dtype=np.float64)
b = np.array([3.0], dtype=np.float64)
x = np.array([1.0], dtype=np.float64)
tau = np.array([0], dtype=np.int64)
""",
            "call": "form_residual_sketch(A, b, x, tau)",
            "gold_call": "_oracle_form_residual_sketch(A, b, x, tau)",
        },
        {
            "setup": """import numpy as np
A = np.eye(2, dtype=np.float64)
b = np.array([1.0, 2.0], dtype=np.float64)
x = np.zeros(2, dtype=np.float64)
tau = np.array([0, 0], dtype=np.int64)

def catches_value_error(fn):
    try:
        fn(A, b, x, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "catches_value_error(form_residual_sketch)",
            "gold_call": "catches_value_error(_oracle_form_residual_sketch)",
        },
    ]
