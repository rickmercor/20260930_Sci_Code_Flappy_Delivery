"""
Return each row's inner minimisation value in the strong dual of the worst-case expected utility of `weights`, at transport multiplier `lam`.

The dual objective at `lam` is -lam * radius plus the mean of these values; the support is [xmin, xmax].

Returns
-------
np.ndarray of shape (N,), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def inner_worst_case(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                     returns: np.ndarray, lam: float) -> np.ndarray:
    '''Return the per-observation inner values of the strong dual.

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
    lam : float
        Transport multiplier.

    Returns
    -------
    values : np.ndarray
        Shape (N,), dtype float, in row order, to floating-point accuracy.

    Raises
    ------
    ValueError
        If lam is not a finite scalar >= 0; if xmin and xmax are not finite
        one-dimensional arrays of a common length n >= 1 with
        -1 < xmin < xmax elementwise; if returns is not a finite (N, n) array
        with N >= 1 and every row inside the box; or if weights is not a
        finite nonnegative (n,) array summing to one within 1e-9.
    '''
    return values  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ref_prefix_corners(w, X, xmin):
    """Portfolio return and l1 distance of each row's candidate corners."""
    order = np.argsort(-w, kind="stable")
    d = (X - xmin)[:, order]
    zero = np.zeros((X.shape[0], 1))
    distance = np.hstack([zero, np.cumsum(d, axis=1)])
    reduction = np.hstack([zero, np.cumsum(d * w[order], axis=1)])
    return (X @ w)[:, None] - reduction, distance


def _oracle_inner_worst_case(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                             returns: np.ndarray, lam: float) -> np.ndarray:
    """Reference implementation."""
    if not np.isscalar(lam) or not np.isfinite(float(lam)) or float(lam) < 0.0:
        raise ValueError("lam must be a finite nonnegative scalar")
    xmin, xmax = _ref_check_box(xmin, xmax)
    X = _ref_observations(returns, xmin, xmax)
    w = _ref_unit_weights(weights, xmin.size)
    portfolio_return, distance = _ref_prefix_corners(w, X, xmin)
    return np.min(np.log1p(portfolio_return) + float(lam) * distance, axis=1)

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
        inner_worst_case(w.copy(), xmin.copy(), xmax.copy(), X.copy(), lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_inner_worst_case(w, xmin, xmax, X, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    call = "inner_worst_case(w.copy(), xmin.copy(), xmax.copy(), X.copy(), lam)"
    gold = "_oracle_inner_worst_case(w, xmin, xmax, X, lam)"
    return [
        # --- Valid: production sample, 200 held assets at 101 distinct weight
        #     levels; the minimisers move between 72 and 76 assets ---
        {"setup": base + "xmin, xmax, X = instance(1000, 25)\n" + spread + "lam = 0.006\n",
         "call": call, "gold_call": gold},
        # --- Valid: production sample, a portfolio with the structure of the
        #     surrogate optimum, at a price where some rows stay put and others
        #     move every held asset ---
        {"setup": base + "xmin, xmax, X = instance(1000, 25)\n" + structured + "lam = 0.011\n",
         "call": call, "gold_call": gold},
        # --- Valid: six-asset projection, distinct weights, minimisers that
        #     move different numbers of assets in different rows ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
lam = 0.18
""", "call": call, "gold_call": gold},
        # --- Boundary: free transport ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
lam = 0.0
""", "call": call, "gold_call": gold},
        # --- Edge: transport too expensive to move anything ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
lam = 5.0
""", "call": call, "gold_call": gold},
        # --- Edge: ten assets, concentrated weights with a zero entry ---
        {"setup": base + """xmin, xmax, X = instance(10, 5)
w = np.array([0.30, 0.02, 0.25, 0.01, 0.12, 0.08, 0.0, 0.15, 0.04, 0.03])
lam = 0.12
""", "call": call, "gold_call": gold},
        # --- Invalid: negative price ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.full(6, 1.0 / 6.0)
lam = -0.1
""" + invalid, "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: weights that do not sum to one ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
w = np.full(6, 0.2)
lam = 0.1
""" + invalid, "call": "run_model()", "gold_call": "run_gold()"},
    ]
