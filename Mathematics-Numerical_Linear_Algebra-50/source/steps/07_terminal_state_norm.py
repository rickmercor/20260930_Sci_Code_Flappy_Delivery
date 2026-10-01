"""
Run the whole pipeline on an affinely parametrized uncertain system and report the Euclidean norm of the plant state after a fixed horizon.

Realize the system, then run the switched loop: select an index, falsify it if the trajectory breaks its promise, and cycle until one holds. Two implementations that differ in when they advance the index, which certificate sets the promised level, or how the index wraps produce different index sequences and different norms.

Returns
-------
float, native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def terminal_state_norm(A0: np.ndarray, B0: np.ndarray, A_dirs: list,
                        B_dirs: list, theta: np.ndarray, x0: np.ndarray,
                        K_list: list, P_list: list, alpha: float,
                        n_steps: int) -> float:
    """Norm of the plant state after a fixed horizon of the switched loop.

    Parameters
    ----------
    A0 : np.ndarray
        (n, n) nominal state matrix.
    B0 : np.ndarray
        (n, m) nominal input matrix.
    A_dirs : list
        Length-L list of (n, n) state directions, one per parameter.
    B_dirs : list
        Length-L list of (n, m) input directions, one per parameter.
    theta : np.ndarray
        (L,) parameter vector naming the realization.
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
    norm : float
        Euclidean norm of the plant state after n_steps steps, as a native
        Python float.

    Raises
    ------
    ValueError
        If A0 is not a square 2-D array; if B0 is not 2-D with the same number
        of rows as A0; if len(A_dirs) or len(B_dirs) differs from the length of
        theta, or any direction does not match the shape of its nominal
        counterpart; if x0 is not one-dimensional of length n; if K_list and
        P_list have different lengths or either is empty; if n_steps is not an
        integer (booleans excluded) or is less than 1; if alpha is not a finite
        real number in (0, 1]; or if any input contains non-finite entries.
    """
    return norm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_terminal_state_norm(A0: np.ndarray, B0: np.ndarray, A_dirs: list, B_dirs: list, theta: np.ndarray, x0: np.ndarray, K_list: list, P_list: list, alpha: float, n_steps: int) -> float:
    """Reference implementation."""
    import numpy as np
    AB = _oracle_uncertain_system_matrices(A0, B0, A_dirs, B_dirs, theta)
    n = A0.shape[0]
    A, B = AB[:, :n], AB[:, n:]
    trace = _oracle_switched_trajectory(A, B, x0, K_list, P_list, alpha,
                                        n_steps)
    return float(np.linalg.norm(trace[-1, :-1]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
alpha = 0.9
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
theta = np.array([-0.20, -0.20], dtype=float)
"""
    return [
        # --- Normal: the benchmark configuration ---
        {
            "setup": setup,
            "call": "terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 20)",
            "gold_call": "_oracle_terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 20)",
        },
        # --- Boundary: a corner of the parameter box with a shorter horizon ---
        {
            "setup": setup + "theta_v = np.array([0.25, 0.25], dtype=float)\n",
            "call": "terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta_v.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 9)",
            "gold_call": "_oracle_terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta_v.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 9)",
        },
        # --- Interior corner with a horizon long enough to visit every candidate multiple times ---
        {
            "setup": setup + "theta_w = np.array([-0.18, -0.22], dtype=float)\n",
            "call": "terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta_w.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 15)",
            "gold_call": "_oracle_terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta_w.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 15)",
        },
        # --- Edge: the zero initial state stays at the origin for any horizon ---
        {
            "setup": setup + "x_zero = np.zeros(3)\n",
            "call": "terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy(), x_zero.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 20)",
            "gold_call": "_oracle_terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy(), x_zero.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 20)",
        },
        # --- Invalid: parameter vector longer than the direction lists ---
        {
            "setup": setup + """theta_bad = np.array([0.1, 0.2, 0.3])
def run_model():
    try:
        terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta_bad.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 20)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta_bad.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 20)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: horizon below one ---
        {
            "setup": setup + """def run_model():
    try:
        terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], alpha, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: decay rate of zero ---
        {
            "setup": setup + """def run_model():
    try:
        terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], 0.0, 20)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_terminal_state_norm(A0.copy(), B0.copy(), [a.copy() for a in A_dirs], [b.copy() for b in B_dirs], theta.copy(), x0.copy(), [k.copy() for k in K_list], [p.copy() for p in P_list], 0.0, 20)
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
