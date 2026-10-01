"""
Compute the controller state carried into the next step from the current state and the index selected at the current step.

The next promised level has to reflect whichever certificate the supervisor is now relying on, not the one it just abandoned, or the falsification test at the next step would be checking a claim nobody is actually making anymore.

Returns
-------
np.ndarray, shape (2,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def controller_state_update(x: np.ndarray, index: int, P_list: list,
                            alpha: float) -> np.ndarray:
    """Build the controller state carried into the next step.

    Parameters
    ----------
    x : np.ndarray
        (n,) current state.
    index : int
        One-based index in 1..q of the certificate selected at this step.
    P_list : list
        Length-q list of (n, n) symmetric positive definite certificate
        matrices, ordered so that entry j - 1 belongs to index j.
    alpha : float
        Decay rate in (0, 1].

    Returns
    -------
    z_next : np.ndarray
        (2,) float array; entry 0 is the level the next state's quadratic form
        must not exceed, entry 1 is index stored as a float.

    Raises
    ------
    ValueError
        If x is not one-dimensional; if index is not an integer (booleans
        excluded) in 1..q with q the length of P_list; if P_list is empty, or
        any entry does not have shape (n, n) with n the length of x, or any
        entry is not symmetric within atol 1e-10; if alpha is not a finite real
        number in (0, 1]; or if any input contains non-finite entries.
    """
    return z_next

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_controller_state_update(x: np.ndarray, index: int, P_list: list, alpha: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    xv = np.asarray(x, dtype=float)
    if xv.ndim != 1 or xv.shape[0] < 1:
        raise ValueError("x must be a one-dimensional array")
    q = len(P_list)
    if q < 1:
        raise ValueError("P_list must be non-empty")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer in 1..q")
    idx = int(index)
    if idx < 1 or idx > q:
        raise ValueError("index must be an integer in 1..q")
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
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating, np.integer)):
        raise ValueError("alpha must be a real number in (0, 1]")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("alpha must be a finite real number in (0, 1]")
    if not np.all(np.isfinite(xv)):
        raise ValueError("inputs must be finite")
    level = a * a * float(xv @ Ps[idx - 1] @ xv) - float(xv @ xv)
    return np.array([level, float(idx)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
alpha = 0.9
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
        # --- Normal: the first index of the benchmark family ---
        {
            "setup": setup,
            "call": "controller_state_update(x.copy(), 1, [p.copy() for p in P_list], alpha)",
            "gold_call": "_oracle_controller_state_update(x.copy(), 1, [p.copy() for p in P_list], alpha)",
        },
        # --- Boundary: the last index, where the certificate differs most in scale ---
        {
            "setup": setup,
            "call": "controller_state_update(x.copy(), 3, [p.copy() for p in P_list], alpha)",
            "gold_call": "_oracle_controller_state_update(x.copy(), 3, [p.copy() for p in P_list], alpha)",
        },
        # --- Edge: the zero state, where the level is exactly zero ---
        {
            "setup": setup + "x0 = np.zeros(3)\n",
            "call": "controller_state_update(x0.copy(), 2, [p.copy() for p in P_list], alpha)",
            "gold_call": "_oracle_controller_state_update(x0.copy(), 2, [p.copy() for p in P_list], alpha)",
        },
        # --- Invalid: index outside 1..q ---
        {
            "setup": setup + """def run_model():
    try:
        controller_state_update(x.copy(), 5, [p.copy() for p in P_list], alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_controller_state_update(x.copy(), 5, [p.copy() for p in P_list], alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero index, which is outside the one-based range ---
        {
            "setup": setup + """def run_model():
    try:
        controller_state_update(x.copy(), 0, [p.copy() for p in P_list], alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_controller_state_update(x.copy(), 0, [p.copy() for p in P_list], alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: decay rate above one ---
        {
            "setup": setup + """def run_model():
    try:
        controller_state_update(x.copy(), 1, [p.copy() for p in P_list], 1.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_controller_state_update(x.copy(), 1, [p.copy() for p in P_list], 1.2)
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
