"""
Advance the plant state by one step under the static gain belonging to the selected certificate index.

The plant only ever sees a single static gain at each instant; all of the supervisor's memory about which candidate is trusted lives elsewhere, in the controller state, not in how the plant itself is advanced.

Returns
-------
np.ndarray, shape (n,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def closed_loop_step(A: np.ndarray, B: np.ndarray, x: np.ndarray, index: int,
                     K_list: list) -> np.ndarray:
    """Advance the plant one step under the selected gain.

    Parameters
    ----------
    A : np.ndarray
        (n, n) realized state matrix.
    B : np.ndarray
        (n, m) realized input matrix.
    x : np.ndarray
        (n,) current plant state.
    index : int
        One-based index in 1..q of the gain selected at this step.
    K_list : list
        Length-q list of (m, n) gains, entry j - 1 belonging to index j.

    Returns
    -------
    x_next : np.ndarray
        (n,) plant state after one step.

    Raises
    ------
    ValueError
        If A is not a square 2-D array; if B is not 2-D with the same number of
        rows as A; if x is not one-dimensional of length n; if index is not an
        integer (booleans excluded) in 1..q with q the length of K_list; if
        K_list is empty or any entry does not have shape (m, n); or if any input
        contains non-finite entries.
    """
    return x_next

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_closed_loop_step(A: np.ndarray, B: np.ndarray, x: np.ndarray, index: int, K_list: list) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    xv = np.asarray(x, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2-D array")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must be 2-D with the same number of rows as A")
    m = B.shape[1]
    if xv.ndim != 1 or xv.shape[0] != n:
        raise ValueError("x must be one-dimensional of length n")
    q = len(K_list)
    if q < 1:
        raise ValueError("K_list must be non-empty")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer in 1..q")
    idx = int(index)
    if idx < 1 or idx > q:
        raise ValueError("index must be an integer in 1..q")
    Ks = []
    for Kj in K_list:
        Ka = np.asarray(Kj, dtype=float)
        if Ka.shape != (m, n):
            raise ValueError("each entry of K_list must have shape (m, n)")
        if not np.all(np.isfinite(Ka)):
            raise ValueError("inputs must be finite")
        Ks.append(Ka)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            and np.all(np.isfinite(xv))):
        raise ValueError("inputs must be finite")
    u = Ks[idx - 1] @ xv
    return A @ xv + B @ u

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
A = np.array([[1.10, 0.00, 0.00],
              [0.00, 0.95, 0.30],
              [-0.10, 0.00, 1.05]], dtype=float)
B = np.array([[1.0, 0.0],
              [0.0, 0.8],
              [0.3, 0.5]], dtype=float)
K1 = np.array([[-0.981340, -0.214471,  0.313087],
               [ 0.171413, -0.576659, -0.644444]], dtype=float)
K2 = np.array([[-0.894080,  0.235455, -0.527303],
               [-0.031926, -0.500010, -0.394591]], dtype=float)
K3 = np.array([[-0.998055, -0.677131,  0.093177],
               [-0.346396, -1.139076, -0.104864]], dtype=float)
K4 = np.array([[-0.599742,  0.020224, -0.514584],
               [-0.107856, -0.745123, -0.040444]], dtype=float)
K_list = [K1, K2, K3, K4]
x = np.array([0.3, 0.8, -0.6], dtype=float)
"""
    return [
        # --- Normal: the first gain of the benchmark family ---
        {
            "setup": setup,
            "call": "closed_loop_step(A.copy(), B.copy(), x.copy(), 1, [k.copy() for k in K_list])",
            "gold_call": "_oracle_closed_loop_step(A.copy(), B.copy(), x.copy(), 1, [k.copy() for k in K_list])",
        },
        # --- Boundary: the last gain, which differs most from the first ---
        {
            "setup": setup,
            "call": "closed_loop_step(A.copy(), B.copy(), x.copy(), 4, [k.copy() for k in K_list])",
            "gold_call": "_oracle_closed_loop_step(A.copy(), B.copy(), x.copy(), 4, [k.copy() for k in K_list])",
        },
        # --- Edge: the zero state stays at the origin under any gain ---
        {
            "setup": setup + "x0 = np.zeros(3)\n",
            "call": "closed_loop_step(A.copy(), B.copy(), x0.copy(), 2, [k.copy() for k in K_list])",
            "gold_call": "_oracle_closed_loop_step(A.copy(), B.copy(), x0.copy(), 2, [k.copy() for k in K_list])",
        },
        # --- Invalid: index beyond the family ---
        {
            "setup": setup + """def run_model():
    try:
        closed_loop_step(A.copy(), B.copy(), x.copy(), 9, [k.copy() for k in K_list])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_closed_loop_step(A.copy(), B.copy(), x.copy(), 9, [k.copy() for k in K_list])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: state of the wrong length ---
        {
            "setup": setup + """x_bad = np.array([0.3, 0.8])
def run_model():
    try:
        closed_loop_step(A.copy(), B.copy(), x_bad.copy(), 1, [k.copy() for k in K_list])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_closed_loop_step(A.copy(), B.copy(), x_bad.copy(), 1, [k.copy() for k in K_list])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a gain with transposed shape ---
        {
            "setup": setup + """K_bad = [K1.T, K2, K3, K4]
def run_model():
    try:
        closed_loop_step(A.copy(), B.copy(), x.copy(), 1, [k.copy() for k in K_bad])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_closed_loop_step(A.copy(), B.copy(), x.copy(), 1, [k.copy() for k in K_bad])
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
