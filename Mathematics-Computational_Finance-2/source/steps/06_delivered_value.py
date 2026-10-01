"""
Return the worst-case expected utility of the portfolio `weights`.

The utility is U itself, not the surrogate; the ambiguity set and support are as in surrogate_worst_case.

Returns
-------
float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def delivered_value(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                    returns: np.ndarray, radius: float) -> float:
    '''Return the worst-case expected utility of a fixed portfolio.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n,). Portfolio weights.
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    returns : np.ndarray
        Shape (N, n). Observed returns, one per row.
    radius : float
        Wasserstein radius.

    Returns
    -------
    value : float
        To floating-point accuracy.

    Raises
    ------
    ValueError
        If radius is not a finite scalar > 0; if xmin and xmax are not finite
        one-dimensional arrays of a common length n >= 1 with
        -1 < xmin < xmax elementwise; if returns is not a finite (N, n) array
        with N >= 1 and every row inside the box; or if weights is not a
        finite nonnegative (n,) array summing to one within 1e-9.
    '''
    return value  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def _oracle_delivered_value(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                            returns: np.ndarray, radius: float) -> float:
    """Reference implementation."""
    radius = _ref_positive_radius(radius)
    xmin, xmax = _ref_check_box(xmin, xmax)
    X = _ref_observations(returns, xmin, xmax)
    w = _ref_unit_weights(weights, xmin.size)
    portfolio_return, distance = _ref_prefix_corners(w, X, xmin)
    N, K = portfolio_return.shape
    utility = np.log1p(portfolio_return).ravel()
    distance = distance.ravel()
    # Transport dual with the portfolio fixed: maximise -lam*radius + mean_j a_j
    # subject to a_j <= log(1 + y_jk) + lam * d_jk over every prefix corner k.
    rows = np.arange(N * K)
    A_ub = sparse.csr_matrix(
        (np.concatenate([-distance, np.ones(N * K)]),
         (np.concatenate([rows, rows]),
          np.concatenate([np.zeros(N * K, dtype=int), 1 + rows // K]))),
        shape=(N * K, 1 + N))
    c = np.concatenate([[radius], -np.ones(N) / N])
    res = linprog(c, A_ub=A_ub, b_ub=utility, bounds=[(0.0, None)] + [(None, None)] * N,
                  method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    lam = float(res.x[0])
    return float(-lam * radius + np.mean(_oracle_inner_worst_case(w, xmin, xmax, X, lam)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
def instance(n, N):
    i = np.arange(n)
    j = np.arange(N)
    xmin = -(150 + (37 * i) % 121) / 1000.0
    xmax = (120 + (53 * i) % 101) / 1000.0
    X = (((7 * j[:, None] + 13 * i[None, :] + 5 * j[:, None] * i[None, :]) % 181) - 90) / 1000.0
    return xmin, xmax, X
"""
    structured = """cap_idx = [17, 35, 53, 59, 62, 69, 70, 80, 83, 89, 107, 125, 147, 161, 162, 179, 198, 216,
           234, 240, 243, 250, 251, 261, 264, 270, 288, 306, 328, 342, 343, 360, 379, 397,
           415, 421, 424, 431, 432, 442, 445, 451, 469, 487, 508, 509, 523, 524, 541, 560,
           578, 596, 602, 605, 612, 613, 623, 626, 632, 650, 668, 690, 704, 705, 722, 741,
           759, 777, 783, 786, 793, 794, 804, 807, 813, 831, 849, 870, 871, 885, 886, 903,
           922, 940, 958, 964, 967, 974, 975, 985, 988, 994]
kink_idx = [38, 146, 219, 327, 400, 581, 689, 943]
w = np.zeros(1000)
w[cap_idx] = 0.01
w[kink_idx] = 0.008857762318920874
w[762] = 1.0 - w.sum()
"""
    spread = """held = np.arange(0, 1000, 5)
raw = 1.0 + ((held * 37) % 101) / 101.0
w = np.zeros(1000)
w[held] = raw / raw.sum()
"""
    invalid = """
def run_model():
    try:
        delivered_value(w.copy(), xmin.copy(), xmax.copy(), X.copy(), radius)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_delivered_value(w, xmin, xmax, X, radius)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    call = "delivered_value(w.copy(), xmin.copy(), xmax.copy(), X.copy(), radius)"
    gold = "_oracle_delivered_value(w, xmin, xmax, X, radius)"
    return [
        # --- Valid: production sample, a portfolio with the structure of the
        #     surrogate optimum ---
        {"setup": base + "xmin, xmax, X = instance(1000, 25)\n" + structured + "radius = 0.01\n",
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Valid: production sample, 200 held assets at 101 distinct weight
        #     levels, where the worst case moves only part of the held assets ---
        {"setup": base + "xmin, xmax, X = instance(1000, 25)\n" + spread + "radius = 0.05\n",
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Valid: ten assets, concentrated weights, a larger radius ---
        {"setup": base + """xmin, xmax, X = instance(10, 5)
w = np.array([0.30, 0.02, 0.25, 0.01, 0.12, 0.08, 0.0, 0.15, 0.04, 0.03])
radius = 0.2
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Valid: six-asset projection, distinct weights ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
radius = 0.01
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Boundary: radius equal to the l1 diameter of the box ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
radius = float(np.sum(xmax - xmin))
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Edge: a very small radius ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
radius = 1e-6
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Edge: a single asset ---
        {"setup": base + """xmin, xmax = np.array([-0.3]), np.array([0.4])
X = np.array([[0.1], [-0.2], [0.3]])
w = np.array([1.0])
radius = 0.05
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Invalid: nonpositive radius ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.full(6, 1.0 / 6.0)
radius = -0.01
""" + invalid, "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: weights that do not sum to one ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.full(6, 0.2)
radius = 0.01
""" + invalid, "call": "run_model()", "gold_call": "run_gold()"},
    ]
