"""
Run the switched closed loop for a fixed number of steps and return the plant states and the index active at each step.

Composing the per-step pieces into a trajectory requires getting their order and their dependence on each other right, including how the loop behaves before any candidate has been tried yet. Before simulating, confirm the family is well-posed for this realization: use lyapunov_certificate_margin to check that at least one (K_i, P_i) pair has positive margin for the given (A, B) and alpha. This positive-margin check is a weaker condition than the stronger, unit-normalized margin needed to guarantee the switching scheme settles on a single candidate; passing this check permits the simulation to proceed without asserting that stronger guarantee holds.

Returns
-------
np.ndarray, shape (n_steps, n + 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def switched_trajectory(A: np.ndarray, B: np.ndarray, x0: np.ndarray,
                        K_list: list, P_list: list, alpha: float,
                        n_steps: int) -> np.ndarray:
    """Simulate the switched closed loop from the zero controller state.

    Parameters
    ----------
    A : np.ndarray
        (n, n) realized state matrix.
    B : np.ndarray
        (n, m) realized input matrix.
    x0 : np.ndarray
        (n,) initial plant state.
    K_list : list
        Length-q list of (m, n) gains, entry j - 1 belonging to index j.
    P_list : list
        Length-q list of (n, n) symmetric certificate matrices, entry j - 1
        belonging to index j.
    alpha : float
        Decay rate in (0, 1].
    n_steps : int
        Number of steps to simulate, at least 1.

    Returns
    -------
    trace : np.ndarray
        (n_steps, n + 1) float array. Row t holds the plant state after step
        t + 1 in its first n entries, and the one-based active index at step
        t + 1, stored as a float, in the last entry.

    Raises
    ------
    ValueError
        If A is not a square 2-D array; if B is not 2-D with the same number of
        rows as A; if x0 is not one-dimensional of length n; if K_list and
        P_list have different lengths or either is empty; if n_steps is not an
        integer (booleans excluded) or is less than 1; if alpha is not a finite
        real number in (0, 1]; if none of the (K_i, P_i) pairs in K_list,
        P_list has a positive certificate margin (via lyapunov_certificate_margin)
        for the given A, B, and alpha; or if any input contains non-finite entries.
    """
    return trace

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_switched_trajectory(A: np.ndarray, B: np.ndarray, x0: np.ndarray, K_list: list, P_list: list, alpha: float, n_steps: int) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    xv = np.asarray(x0, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2-D array")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must be 2-D with the same number of rows as A")
    if xv.ndim != 1 or xv.shape[0] != n:
        raise ValueError("x0 must be one-dimensional of length n")
    if len(K_list) < 1 or len(P_list) < 1:
        raise ValueError("K_list and P_list must be non-empty")
    if len(K_list) != len(P_list):
        raise ValueError("K_list and P_list must have the same length")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise ValueError("n_steps must be an integer >= 1")
    T = int(n_steps)
    if T < 1:
        raise ValueError("n_steps must be an integer >= 1")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating, np.integer)):
        raise ValueError("alpha must be a real number in (0, 1]")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("alpha must be a finite real number in (0, 1]")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            and np.all(np.isfinite(xv))):
        raise ValueError("inputs must be finite")
    margins = [_oracle_lyapunov_certificate_margin(A, B, K_list[j], P_list[j], a)
               for j in range(len(K_list))]
    if not any(m > 0.0 for m in margins):
        raise ValueError("no candidate certificate is valid for the realized system")
    x = xv.copy()
    z = np.array([0.0, 0.0], dtype=float)
    states = np.empty((T, n), dtype=float)
    indices = np.empty(T, dtype=int)
    for t in range(T):
        idx = _oracle_select_active_index(x, z, P_list)
        z_next = _oracle_controller_state_update(x, idx, P_list, a)
        x = _oracle_closed_loop_step(A, B, x, idx, K_list)
        z = z_next
        states[t, :] = x
        indices[t] = idx
    return np.column_stack([states, indices.astype(float)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
alpha = 0.9
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
K_list = [K1, K2, K3, K4]
P_list = [P1, P2, P3, P4]
x0 = np.array([0.3, 0.8, -0.6], dtype=float)
"""
    return [
        # --- Normal: a horizon long enough to show several falsifications ---
        {
            "setup": setup,
            "call": "switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 20)",
            "gold_call": "_oracle_switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 20)",
        },
        # --- Boundary: a single step, which must use the first certificate ---
        {
            "setup": setup,
            "call": "switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 1)",
            "gold_call": "_oracle_switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 1)",
        },
        # --- Edge: a family of one certificate, where the cycle returns to itself ---
        {
            "setup": setup + "K_one = [K1]\nP_one = [P1]\n",
            "call": "switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_one], [p.copy() for p in P_one], alpha, 12)",
            "gold_call": "_oracle_switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_one], [p.copy() for p in P_one], alpha, 12)",
        },
        # --- Invalid: zero steps ---
        {
            "setup": setup + """def run_model():
    try:
        switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: gain and certificate families of different lengths ---
        {
            "setup": setup + """P_short = [P1, P2]
def run_model():
    try:
        switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_short], alpha, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_switched_trajectory(A.copy(), B.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_short], alpha, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite initial state ---
        {
            "setup": setup + """x_bad = np.array([0.3, np.inf, -0.6])
def run_model():
    try:
        switched_trajectory(A.copy(), B.copy(), x_bad.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_switched_trajectory(A.copy(), B.copy(), x_bad.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 5)
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
