"""
Realize an affinely parametrized uncertain linear system at a parameter vector.

The realized system at a parameter value is the nominal pair plus each direction matrix scaled by its own parameter entry. This affine form is the standard way to describe parametric uncertainty in robust control. The realized A and B are packed into one array, [A, B], with A the first n columns and B the rest.

Returns
-------
np.ndarray, shape (n, n + m), as [A, B]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def uncertain_system_matrices(A0: np.ndarray, B0: np.ndarray, A_dirs: list,
                              B_dirs: list, theta: np.ndarray) -> np.ndarray:
    """Realize an affinely parametrized uncertain system at a parameter vector.

    Parameters
    ----------
    A0 : np.ndarray
        (n, n) nominal state matrix.
    B0 : np.ndarray
        (n, m) nominal input matrix.
    A_dirs : list
        Length-L list of (n, n) float arrays, one state direction per parameter.
    B_dirs : list
        Length-L list of (n, m) float arrays, one input direction per parameter.
    theta : np.ndarray
        (L,) float array of parameter values.

    Returns
    -------
    AB : np.ndarray
        (n, n + m) array [A, B] with the realized state matrix in the first n
        columns and the realized input matrix in the remaining m columns.

    Raises
    ------
    ValueError
        If A0 is not a square 2-D array; if B0 is not 2-D with the same number
        of rows as A0; if len(A_dirs) or len(B_dirs) differs from the length of
        theta; if any entry of A_dirs does not have the shape of A0 or any entry
        of B_dirs does not have the shape of B0; if theta is not one-dimensional;
        or if any input contains non-finite entries.
    """
    return AB

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_uncertain_system_matrices(A0: np.ndarray, B0: np.ndarray, A_dirs: list, B_dirs: list, theta: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    A0 = np.asarray(A0, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    th = np.asarray(theta, dtype=float)
    if A0.ndim != 2 or A0.shape[0] != A0.shape[1] or A0.shape[0] < 1:
        raise ValueError("A0 must be a square 2-D array")
    if B0.ndim != 2 or B0.shape[0] != A0.shape[0]:
        raise ValueError("B0 must be 2-D with the same number of rows as A0")
    if th.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    if len(A_dirs) != th.shape[0] or len(B_dirs) != th.shape[0]:
        raise ValueError("A_dirs and B_dirs must have the same length as theta")
    if not (np.all(np.isfinite(A0)) and np.all(np.isfinite(B0))
            and np.all(np.isfinite(th))):
        raise ValueError("inputs must be finite")
    A = A0.copy()
    B = B0.copy()
    for j in range(th.shape[0]):
        Aj = np.asarray(A_dirs[j], dtype=float)
        Bj = np.asarray(B_dirs[j], dtype=float)
        if Aj.shape != A0.shape:
            raise ValueError("each entry of A_dirs must have the shape of A0")
        if Bj.shape != B0.shape:
            raise ValueError("each entry of B_dirs must have the shape of B0")
        if not (np.all(np.isfinite(Aj)) and np.all(np.isfinite(Bj))):
            raise ValueError("inputs must be finite")
        A = A + float(th[j]) * Aj
        B = B + float(th[j]) * Bj
    return np.hstack([A, B])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
A0 = np.array([[1.10, 0.20, 0.00],
               [0.00, 0.95, 0.30],
               [0.10, 0.00, 1.05]], dtype=float)
B0 = np.array([[1.0, 0.0],
               [0.0, 1.0],
               [0.5, 0.5]], dtype=float)
A1 = np.zeros((3, 3)); A1[0, 1] = 1.0
A2 = np.zeros((3, 3)); A2[2, 0] = 1.0
B1 = np.zeros((3, 2)); B1[2, 0] = 1.0
B2 = np.zeros((3, 2)); B2[1, 1] = 1.0
A_dirs = [A1, A2]
B_dirs = [B1, B2]
"""
    return [
        # --- Normal: the production parameter vector of the benchmark instance ---
        {
            "setup": setup + "theta = np.array([-0.20, -0.20])\n",
            "call": "uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())",
            "gold_call": "_oracle_uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())",
        },
        # --- Boundary: a corner of the parameter box, where both directions act fully ---
        {
            "setup": setup + "theta = np.array([0.25, -0.25])\n",
            "call": "uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())",
            "gold_call": "_oracle_uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())",
        },
        # --- Edge: the zero parameter vector returns the nominal pair unchanged ---
        {
            "setup": setup + "theta = np.zeros(2)\n",
            "call": "uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())",
            "gold_call": "_oracle_uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())",
        },
        # --- Invalid: theta length does not match the number of directions ---
        {
            "setup": setup + """theta = np.array([0.1, 0.2, 0.3])
def run_model():
    try:
        uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a direction array with the wrong shape ---
        {
            "setup": setup + """theta = np.array([-0.20, -0.20])
A_bad = [np.zeros((2, 2)), A2]
def run_model():
    try:
        uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_bad], [b.copy() for b in B_dirs], theta.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_uncertain_system_matrices(A0.copy(), B0.copy(), [a.copy() for a in A_bad], [b.copy() for b in B_dirs], theta.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite nominal entry ---
        {
            "setup": setup + """theta = np.array([-0.20, -0.20])
A_nan = A0.copy(); A_nan[0, 0] = np.nan
def run_model():
    try:
        uncertain_system_matrices(A_nan.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_uncertain_system_matrices(A_nan.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy())
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
