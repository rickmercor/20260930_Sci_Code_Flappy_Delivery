"""
Build A, take the column subset, draw the sketch, evaluate the source's specialised Frobenius residual-bound factor, form the sketched middle matrix, confirm that one reconstruction entry is finite, and return the residual-bound factor.

The orchestrator is the prompt instance: the source's specialised Frobenius residual-bound factor on the given column subset and the given unscaled draw, returned only when the source's rank assumption on the sketched column span holds and the reconstruction entry formed along the way is finite.

Returns
-------
float: specialised Frobenius residual-bound factor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_sketched_core_entry(
    n: int,
    s: np.ndarray,
    data_seed: int,
    sketch_seed: int,
    indices: np.ndarray,
    t: int,
    row: int,
    col: int,
) -> float:
    """End-to-end sketched symmetric reconstruction; return the residual factor.

    Chains the earlier steps: build A from (n, s, data_seed), take the
    column subset C = A[:, indices], draw the (t, n) Gaussian sketch from
    sketch_seed, evaluate the cited source's specialised Frobenius
    residual-bound factor on this column subset and this sketch, recover
    the middle matrix of the cited reconstruction from A, C and X, confirm
    that (C M C^T)[row, col] is finite, and return the residual-bound
    factor. The factor is finite and strictly greater than one when the
    source's rank assumption holds.

    Parameters
    ----------
    n : int
        Dimension, n >= 2.
    s : np.ndarray
        Eigenvalues of length n with both a positive and a negative entry.
    data_seed : int
        Seed for the matrix construction.
    sketch_seed : int
        Seed for the Gaussian sketch.
    indices : np.ndarray
        Distinct 0-based column indices, length r with 1 <= r < n.
    t : int
        Sketch height with r < t < n.
    row, col : int
        0-based indices in 0, ..., n-1.

    Returns
    -------
    value : float
        The specialised Frobenius residual-bound factor.

    Raises
    ------
    ValueError
        If t is not an integer or t <= r, if the residual-bound factor of
        the sketch on the column span is not finite and strictly greater
        than one, if the reconstruction entry is not finite, and any
        ValueError raised by the earlier steps for their invalid inputs
        (for example s with no negative entry, n < 2, repeated or
        out-of-range indices, t >= n, or a rank-deficient sketched
        column block).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_sketched_core_entry(
    n, s, data_seed, sketch_seed, indices, t, row, col
):
    if not isinstance(t, (int, np.integer)):
        raise ValueError("t must be an integer")
    t = int(t)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    r = indices.size
    if t <= r:
        raise ValueError("require t > r")
    A = _oracle_construct_indefinite_symmetric_matrix(n, s, data_seed)
    C = _oracle_extract_column_subset(A, indices)
    X = _oracle_draw_gaussian_sketch(t, n, sketch_seed)
    factor = _oracle_residual_bound_factor(C, X)
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError("sketch does not embed the column span")
    Mhat = _oracle_sketched_middle_matrix(A, C, X)
    entry = _oracle_reconstruction_entry(C, Mhat, row, col)
    if not np.isfinite(entry):
        raise ValueError("reconstruction entry is not finite")
    return float(factor)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, t, row, col = 12, 6, 2, 0
s = np.array([6.5, 5.2, 4.1, -3.8, -2.9, -1.6, 0.9, 0.55, 0.35, -0.25, 0.18, 0.12])
indices = np.array([5, 8, 9])
data_seed, sketch_seed = 7, 11
""",
            "call": "run_sketched_core_entry(n, s, data_seed, sketch_seed, indices, t, row, col)",
            "gold_call": "_oracle_run_sketched_core_entry(n, s, data_seed, sketch_seed, indices, t, row, col)",
        },
        {
            "setup": """import numpy as np
n, t, row, col = 4, 3, 1, 0
s = np.array([3.0, 1.0, -2.0, -0.4])
indices = np.array([0, 2])
data_seed, sketch_seed = 1, 4
""",
            "call": "run_sketched_core_entry(n, s, data_seed, sketch_seed, indices, t, row, col)",
            "gold_call": "_oracle_run_sketched_core_entry(n, s, data_seed, sketch_seed, indices, t, row, col)",
        },
        {
            "setup": """import numpy as np
n, t, row, col = 3, 2, 0, 1
s = np.array([2.0, -1.5, 0.5])
indices = np.array([1])
data_seed, sketch_seed = 0, 2
""",
            "call": "run_sketched_core_entry(n, s, data_seed, sketch_seed, indices, t, row, col)",
            "gold_call": "_oracle_run_sketched_core_entry(n, s, data_seed, sketch_seed, indices, t, row, col)",
        },
        {
            "setup": """import numpy as np
s = np.array([2.0, -1.0, 0.5, 0.2])
indices = np.array([0, 1])
def run_model():
    try:
        run_sketched_core_entry(4, s, 0, 1, indices, 2, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_sketched_core_entry(4, s, 0, 1, indices, 2, 0, 0)
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
s = np.array([2.0, 1.0, 0.5])
indices = np.array([0, 1])
def run_model():
    try:
        run_sketched_core_entry(3, s, 0, 1, indices, 2, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_sketched_core_entry(3, s, 0, 1, indices, 2, 0, 0)
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
