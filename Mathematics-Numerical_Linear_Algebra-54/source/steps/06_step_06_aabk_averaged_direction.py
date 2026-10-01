"""
Form the paper's block-averaged, weighted row-action direction for one iteration from a batch of drawn rows and the fresh noisy right-hand-side value returned for each draw.

At each iteration the paper draws a batch of tau row indices i.i.d. from its sampling distribution (with replacement, so a row may recur) and, for every draw separately, queries a fresh noisy value of that row's right-hand-side entry; a repeated row therefore receives independent noise on each of its draws. From these it forms a single direction d_k that combines the per-draw row-action terms using the per-row weights w_i and a specific aggregation across the batch (eq. (4) and Fig. 1); the paper stresses that this aggregation, rather than the block sum used by an earlier adaptive method, is what makes larger batches provably helpful. Consult eq. (4) of the paper for the exact per-draw term, its normalization, and the exact aggregation over the batch.

Returns
-------
np.ndarray of shape (n,), dtype float — the direction d_k for this iteration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aabk_averaged_direction(A: "np.ndarray", x: "np.ndarray", w: "np.ndarray",
                            batch: "np.ndarray",
                            b_noisy: "np.ndarray") -> "np.ndarray":
    '''Form the paper's direction d_k (eq. (4)) for one iteration.

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix with no zero row.
    x : np.ndarray
        (n,) current primal iterate.
    w : np.ndarray
        (m,) per-row weights, from noise_aware_coupling.
    batch : np.ndarray
        (tau,) integer row indices drawn for this iteration, in draw order;
        indices may repeat.
    b_noisy : np.ndarray
        (tau,) fresh noisy right-hand-side value returned for each draw, in
        the same draw order as batch (entry j belongs to row batch[j]).

    Returns
    -------
    d : np.ndarray
        (n,) the direction d_k defined in eq. (4) of the paper (not
        restated here).

    Raises
    ------
    ValueError
        If A is not a 2D array, if x does not have shape (n,) matching A's
        columns, if w does not have shape (m,) matching A's rows, if batch
        is empty, if batch and b_noisy do not have the same 1D shape, or if
        any index in batch is out of bounds for the rows of A.
    '''
    return d  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_aabk_averaged_direction(A: "np.ndarray", x: "np.ndarray", w: "np.ndarray",
                                    batch: "np.ndarray",
                                    b_noisy: "np.ndarray") -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    batch = np.asarray(batch)
    b_noisy = np.asarray(b_noisy, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if x.shape != (n,):
        raise ValueError("x must have shape (n,) matching A's columns")
    if w.shape != (m,):
        raise ValueError("w must have shape (m,) matching A's rows")
    if batch.ndim != 1 or batch.size == 0:
        raise ValueError("batch must be a non-empty 1D array")
    if b_noisy.shape != batch.shape:
        raise ValueError("b_noisy must have the same shape as batch")
    if np.any(batch < 0) or np.any(batch >= m):
        raise ValueError("batch contains an index out of bounds for A")
    tau = batch.size
    rows = A[batch]                                    # (tau, n)
    row_norms2 = np.sum(rows ** 2, axis=1)             # (tau,)
    # eq. (4): d_k = (1/tau) sum_j w_{i_j} (<a_{i_j}, x> - b~_j) / ||a_{i_j}||^2 a_{i_j}
    coeff = w[batch] * (rows @ x - b_noisy) / row_norms2
    return (coeff @ rows) / tau

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: iteration 0 of the exact problem instance (x = 0,
        #     batch (1, 2, 4) with its fresh noisy values) ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
x = np.zeros(4)
w = np.array([1.9739575001, 2.6727510900, 0.5698324470,
              3.0152698877, 0.2686215916, 2.4175943246])
batch = np.array([1, 2, 4])
b_noisy = np.array([0.7161151969, -5.3579478763, 0.3532085107])
""",
            "call": "aabk_averaged_direction(A, x, w, batch, b_noisy)",
            "gold_call": "_oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)",
        },
        # --- Normal: iteration 3 of the exact problem instance (nonzero
        #     iterate, batch (3, 2, 0)) ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
x = np.array([0.0, 0.0, -0.0859316817, 0.0])
w = np.array([1.9739575001, 2.6727510900, 0.5698324470,
              3.0152698877, 0.2686215916, 2.4175943246])
batch = np.array([3, 2, 0])
b_noisy = np.array([5.7729104256, -2.3646834319, 3.7473856322])
""",
            "call": "aabk_averaged_direction(A, x, w, batch, b_noisy)",
            "gold_call": "_oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)",
        },
        # --- Boundary: a repeated row inside the batch with two different
        #     fresh noisy values (the paper's Fig. 1 situation). ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
x = np.array([0.0, 0.0237511228, 0.0, 0.0])
w = np.array([1.9739575001, 2.6727510900, 0.5698324470,
              3.0152698877, 0.2686215916, 2.4175943246])
batch = np.array([2, 2, 4])
b_noisy = np.array([-5.1724029317, -2.5541638345, 0.2843391803])
""",
            "call": "aabk_averaged_direction(A, x, w, batch, b_noisy)",
            "gold_call": "_oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)",
        },
        # --- Boundary: batch size one (the method's single-row case) on
        #     rows of unequal norm with a non-unit weight. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x = np.array([1.0, -1.0])
w = np.array([0.5, 2.0, 1.5])
batch = np.array([1])
b_noisy = np.array([-3.0])
""",
            "call": "aabk_averaged_direction(A, x, w, batch, b_noisy)",
            "gold_call": "_oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)",
        },
        # --- Edge: the noisy values exactly satisfy the equations at x, so
        #     the direction must vanish for any weights. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x = np.array([1.0, -1.0])
w = np.array([0.5, 2.0, 1.5])
batch = np.array([0, 2, 2, 1])
b_noisy = A[batch] @ x
""",
            "call": "aabk_averaged_direction(A, x, w, batch, b_noisy)",
            "gold_call": "_oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)",
        },
        # --- Edge: a large batch drawn entirely from one row (the paper's
        #     variance-reduction case), with weights far from uniform. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x = np.array([0.2, 0.3])
w = np.array([0.1, 5.0, 1.0])
batch = np.array([1, 1, 1, 1, 1])
b_noisy = np.array([1.0, 3.0, -2.0, 0.5, 2.5])
""",
            "call": "aabk_averaged_direction(A, x, w, batch, b_noisy)",
            "gold_call": "_oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)",
        },
        # --- Invalid: empty batch -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x = np.array([1.0, -1.0])
w = np.array([0.5, 2.0, 1.5])
batch = np.array([], dtype=int)
b_noisy = np.array([])
def run_model():
    try:
        aabk_averaged_direction(A, x, w, batch, b_noisy)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: batch index out of bounds -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x = np.array([1.0, -1.0])
w = np.array([0.5, 2.0, 1.5])
batch = np.array([0, 3])
b_noisy = np.array([1.0, 2.0])
def run_model():
    try:
        aabk_averaged_direction(A, x, w, batch, b_noisy)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: b_noisy length differs from batch length -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x = np.array([1.0, -1.0])
w = np.array([0.5, 2.0, 1.5])
batch = np.array([0, 1])
b_noisy = np.array([1.0, 2.0, 3.0])
def run_model():
    try:
        aabk_averaged_direction(A, x, w, batch, b_noisy)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)
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
