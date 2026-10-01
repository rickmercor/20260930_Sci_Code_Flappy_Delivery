"""
Return the worst-case expected surrogate utility of the portfolio `weights`.

The ambiguity set has radius `radius` around the empirical distribution of `returns`, on the support [xmin, xmax]; `coeffs` is laid out as tangent_coefficients returns it.

Returns
-------
float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def surrogate_worst_case(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                         returns: np.ndarray, coeffs: np.ndarray,
                         radius: float) -> float:
    '''Return the worst-case expected surrogate utility of a fixed portfolio.

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
    coeffs : np.ndarray
        Shape (2, M). Affine pieces of the surrogate utility.
    radius : float
        Wasserstein radius.

    Returns
    -------
    value : float
        To floating-point accuracy.

    Raises
    ------
    ValueError
        If xmin and xmax are not finite one-dimensional arrays of a common
        length n >= 1 with -1 < xmin < xmax elementwise; if returns is not a
        finite (N, n) array with N >= 1 and every row inside the box; if
        weights is not a finite nonnegative (n,) array summing to one within
        1e-9; if coeffs is not a finite (2, M) array with M >= 1 and
        nonnegative slopes; or if radius is not a finite scalar > 0.
    '''
    return value  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def _ref_observations(returns, xmin, xmax):
    """Validate the observed returns against the box and return them as floats."""
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] != xmin.size:
        raise ValueError("returns must have shape (N, n) with N >= 1")
    if not np.all(np.isfinite(X)):
        raise ValueError("returns must be finite")
    if np.any(X < xmin) or np.any(X > xmax):
        raise ValueError("every observation must lie in the box")
    return X


def _ref_unit_weights(weights, n):
    """Validate a long-only, fully invested weight vector of length n."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be a finite nonnegative (n,) array")
    if abs(float(np.sum(w)) - 1.0) > 1e-9:
        raise ValueError("weights must sum to one")
    return w


def _ref_surrogate_pieces(coeffs):
    """Validate the affine pieces and return (slopes, intercepts)."""
    coeffs = np.asarray(coeffs, dtype=float)
    if (coeffs.ndim != 2 or coeffs.shape[0] != 2 or coeffs.shape[1] < 1
            or not np.all(np.isfinite(coeffs))):
        raise ValueError("coeffs must be a finite (2, M) array")
    if np.any(coeffs[0] < 0.0):
        raise ValueError("the slopes must be nonnegative")
    return coeffs[0], coeffs[1]


def _ref_positive_radius(radius):
    """Validate the radius of the Wasserstein ball."""
    if not np.isscalar(radius) or not np.isfinite(float(radius)) or float(radius) <= 0.0:
        raise ValueError("radius must be a finite positive scalar")
    return float(radius)


def _oracle_surrogate_worst_case(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                                 returns: np.ndarray, coeffs: np.ndarray,
                                 radius: float) -> float:
    """Reference implementation."""
    xmin, xmax = _ref_check_box(xmin, xmax)
    X = _ref_observations(returns, xmin, xmax)
    w = _ref_unit_weights(weights, xmin.size)
    alpha, beta = _ref_surrogate_pieces(coeffs)
    radius = _ref_positive_radius(radius)
    N, M = X.shape[0], alpha.size

    # Transport dual with the box support dualised: for each piece the inner
    # minimum over the box is alpha_m <w, x_j> + beta_m minus the l1 cost of
    # moving each held asset to its lower bound where alpha_m w_i exceeds lam.
    held = np.flatnonzero(w > 0.0)
    k = held.size
    D = (X - xmin)[:, held]
    base = alpha[None, :] * (X @ w)[:, None] + beta[None, :]
    nv = 1 + N + M * k
    c = np.zeros(nv)
    c[0] = radius
    c[1:1 + N] = -1.0 / N
    r1 = np.arange(N * M)
    jj, mm = np.divmod(r1, M)
    rows = [r1, np.repeat(r1, k)]
    cols = [1 + jj, (1 + N + mm[:, None] * k + np.arange(k)[None, :]).ravel()]
    vals = [np.ones(N * M), D[jj].ravel()]
    r2 = N * M + np.arange(M * k)
    rows += [r2, r2]
    cols += [np.zeros(M * k, dtype=int), 1 + N + np.arange(M * k)]
    vals += [-np.ones(M * k), -np.ones(M * k)]
    A_ub = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N * M + M * k, nv))
    b_ub = np.concatenate([base.ravel(), -(alpha[:, None] * w[held][None, :]).ravel()])
    bounds = [(0.0, None)] + [(None, None)] * N + [(0.0, None)] * (M * k)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    lam = float(res.x[0])
    excess = np.maximum(alpha[:, None] * w[held][None, :] - lam, 0.0)
    inner = base - D @ excess.T
    return float(-lam * radius + inner.min(axis=1).mean())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
def tangents(lo, hi, M):
    p = np.linspace(lo, hi, M)
    return np.vstack([1.0 / (1.0 + p), np.log1p(p) - p / (1.0 + p)])
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
        surrogate_worst_case(w.copy(), xmin.copy(), xmax.copy(), X.copy(), coeffs.copy(), radius)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_surrogate_worst_case(w, xmin, xmax, X, coeffs, radius)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    call = "surrogate_worst_case(w.copy(), xmin.copy(), xmax.copy(), X.copy(), coeffs.copy(), radius)"
    gold = "_oracle_surrogate_worst_case(w, xmin, xmax, X, coeffs, radius)"
    return [
        # --- Valid: production sample and majorant, a portfolio with the
        #     structure of the surrogate optimum ---
        {"setup": base + "xmin, xmax, X = instance(1000, 25)\n"
                         "coeffs = tangents(-0.26444, 0.21543, 9)\nradius = 0.01\n" + structured,
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Valid: production sample, 200 held assets at 101 distinct weight
        #     levels, a larger radius ---
        {"setup": base + "xmin, xmax, X = instance(1000, 25)\n"
                         "coeffs = tangents(-0.26444, 0.21543, 9)\nradius = 0.05\n" + spread,
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Valid: six-asset projection, distinct weights ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
coeffs = tangents(-0.2215, 0.166, 9)
radius = 0.01
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Boundary: radius equal to the l1 diameter of the box ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
coeffs = tangents(-0.2215, 0.166, 9)
radius = float(np.sum(xmax - xmin))
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Edge: two affine pieces only ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
coeffs = tangents(-0.2215, 0.166, 2)
radius = 0.05
w = np.array([0.24, 0.10, 0.26, 0.20, 0.05, 0.15])
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Edge: a single asset ---
        {"setup": base + """xmin, xmax = np.array([-0.3]), np.array([0.4])
X = np.array([[0.1], [-0.2], [0.3]])
coeffs = tangents(-0.3, 0.4, 5)
radius = 0.05
w = np.array([1.0])
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Invalid: nonpositive radius ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
coeffs = tangents(-0.2215, 0.166, 9)
radius = 0.0
w = np.full(6, 1.0 / 6.0)
""" + invalid, "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: weights that do not sum to one ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
coeffs = tangents(-0.2215, 0.166, 9)
radius = 0.01
w = np.full(6, 0.2)
""" + invalid, "call": "run_model()", "gold_call": "run_gold()"},
    ]
