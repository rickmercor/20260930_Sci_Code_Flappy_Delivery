"""
Select the certificate index the supervisor will use at the current step from the current state and the current controller state.

A supervisor holding several candidates cannot tell which one is right except by watching whether each one's own promise is kept. Reconstructing the fixed order in which discarded candidates are retried, and the indexing convention the family uses, is part of the switching law itself.

Returns
-------
int, native Python int
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_active_index(x: np.ndarray, z: np.ndarray, P_list: list) -> int:
    """Choose the certificate index for the current step.

    Parameters
    ----------
    x : np.ndarray
        (n,) current state.
    z : np.ndarray
        (2,) controller state; z[0] is the promised level and z[1] is the
        previously active index stored as a float.
    P_list : list
        Length-q list of (n, n) symmetric positive definite certificate
        matrices, ordered so that entry j - 1 belongs to index j.

    Returns
    -------
    index : int
        One-based index in 1..q of the certificate to apply at this step.

    Raises
    ------
    ValueError
        If x is not one-dimensional; if z is not a one-dimensional array of
        length 2; if P_list is empty, or any entry does not have shape (n, n)
        with n the length of x, or any entry is not symmetric within atol 1e-10;
        or if any input contains non-finite entries.
    """
    return index

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_active_index(x: np.ndarray, z: np.ndarray, P_list: list) -> int:
    """Reference implementation."""
    import numpy as np
    xv = np.asarray(x, dtype=float)
    zv = np.asarray(z, dtype=float)
    if xv.ndim != 1 or xv.shape[0] < 1:
        raise ValueError("x must be a one-dimensional array")
    if zv.ndim != 1 or zv.shape[0] != 2:
        raise ValueError("z must be a one-dimensional array of length 2")
    q = len(P_list)
    if q < 1:
        raise ValueError("P_list must be non-empty")
    n = xv.shape[0]
    Ps = []
    for Pj in P_list:
        Pa = np.asarray(Pj, dtype=float)
        if Pa.shape != (n, n):
            raise ValueError("each entry of P_list must have shape (n, n)")
        if not np.allclose(Pa, Pa.T, rtol=0.0, atol=1e-10):
            raise ValueError("each entry of P_list must be symmetric within atol 1e-10")
        if not np.all(np.isfinite(Pa)):
            raise ValueError("inputs must be finite")
        Ps.append(Pa)
    if not (np.all(np.isfinite(xv)) and np.all(np.isfinite(zv))):
        raise ValueError("inputs must be finite")
    z1 = float(zv[0])
    z2 = int(round(float(zv[1])))
    if 1 <= z2 <= q:
        if float(xv @ Ps[z2 - 1] @ xv) <= z1:
            return int(z2)
        return int((z2 % q) + 1)
    return 1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
P1 = np.array([[ 4.109287,  1.923951, -3.168718],
               [ 1.923951,  4.508113, -2.980617],
               [-3.168718, -2.980617,  7.039886]], dtype=float)
P2 = np.array([[ 2.535466, -0.450140,  1.084729],
               [-0.450140,  2.357141, -1.468640],
               [ 1.084729, -1.468640,  5.922588]], dtype=float)
P3 = np.array([[10.369840,  8.842806, -6.611408],
               [ 8.842806, 11.281408, -6.344938],
               [-6.611408, -6.344938,  6.939830]], dtype=float)
P4 = np.array([[ 2.580081,  1.236551, -1.007780],
               [ 1.236551,  3.831790, -2.254034],
               [-1.007780, -2.254034,  4.587889]], dtype=float)
P_list = [P1, P2, P3, P4]
x = np.array([0.3, 0.8, -0.6], dtype=float)
"""
    return [
        # --- Normal: promise broken at an interior index, so the supervisor advances ---
        {
            "setup": setup + "z = np.array([0.5, 2.0])\n",
            "call": "select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
            "gold_call": "_oracle_select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
        },
        # --- Boundary: promise broken at the last index, so the cycle wraps to the first ---
        {
            "setup": setup + "z = np.array([0.0, 4.0])\n",
            "call": "select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
            "gold_call": "_oracle_select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
        },
        # --- Edge: recorded index outside 1..q, as at initialization ---
        {
            "setup": setup + "z = np.array([0.0, 0.0])\n",
            "call": "select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
            "gold_call": "_oracle_select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
        },
        # --- Advance from an untested interior transition: index 2 -> 3 ---
        {
            "setup": setup + "z = np.array([4.66221862, 2.0])\n",
            "call": "select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
            "gold_call": "_oracle_select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
        },
        # --- Advance from an untested interior transition: index 3 -> 4 ---
        {
            "setup": setup + "z = np.array([23.35751976, 3.0])\n",
            "call": "select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
            "gold_call": "_oracle_select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
        },
        # --- Boundary: promise met with exact equality, tests <= not < ---
        {
            "setup": setup + "z = np.array([10.71501439, 1.0])\n",
            "call": "select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
            "gold_call": "_oracle_select_active_index(x.copy(), z.copy(), [p.copy() for p in P_list])",
        },
        # --- Invalid: controller state of the wrong length ---
        {
            "setup": setup + """z_bad = np.array([0.0, 2.0, 1.0])
def run_model():
    try:
        select_active_index(x.copy(), z_bad.copy(), [p.copy() for p in P_list])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_active_index(x.copy(), z_bad.copy(), [p.copy() for p in P_list])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a certificate whose shape does not match the state ---
        {
            "setup": setup + """P_bad = [P1, np.eye(2)]
z = np.array([0.0, 1.0])
def run_model():
    try:
        select_active_index(x.copy(), z.copy(), [p.copy() for p in P_bad])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_active_index(x.copy(), z.copy(), [p.copy() for p in P_bad])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: empty certificate family ---
        {
            "setup": setup + """z = np.array([0.0, 1.0])
def run_model():
    try:
        select_active_index(x.copy(), z.copy(), [])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_active_index(x.copy(), z.copy(), [])
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
