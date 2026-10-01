"""
Return the admissible portfolio with per-asset cap `cap` that maximises the value returned by surrogate_worst_case.

The maximiser is unique for every instance used here.

Returns
-------
np.ndarray of shape (n,), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def surrogate_portfolio(xmin: np.ndarray, xmax: np.ndarray, returns: np.ndarray,
                        coeffs: np.ndarray, cap: float, radius: float) -> np.ndarray:
    '''Return the surrogate-optimal admissible portfolio.

    Parameters
    ----------
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    returns : np.ndarray
        Shape (N, n). Observed returns, one per row.
    coeffs : np.ndarray
        Shape (2, M). Affine pieces of the surrogate utility.
    cap : float
        Per-asset weight cap.
    radius : float
        Wasserstein radius.

    Returns
    -------
    weights : np.ndarray
        Shape (n,), dtype float, to floating-point accuracy.

    Raises
    ------
    ValueError
        If xmin and xmax are not finite one-dimensional arrays of a common
        length n >= 1 with -1 < xmin < xmax elementwise; if returns is not a
        finite (N, n) array with N >= 1 and every row inside the box; if
        coeffs is not a finite (2, M) array with M >= 1 and nonnegative
        slopes; if cap is not a finite scalar with cap > 0 and n * cap >= 1;
        or if radius is not a finite scalar > 0.
    '''
    return weights  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def _oracle_surrogate_portfolio(xmin: np.ndarray, xmax: np.ndarray, returns: np.ndarray,
                                coeffs: np.ndarray, cap: float, radius: float) -> np.ndarray:
    """Reference implementation."""
    xmin, xmax = _ref_check_box(xmin, xmax)
    X = _ref_observations(returns, xmin, xmax)
    alpha, beta = _ref_surrogate_pieces(coeffs)
    if not np.isscalar(cap) or not np.isfinite(float(cap)):
        raise ValueError("cap must be a finite scalar")
    cap = float(cap)
    N, n = X.shape
    if cap <= 0.0 or n * cap < 1.0:
        raise ValueError("the admissible portfolio set must be nonempty")
    radius = _ref_positive_radius(radius)
    M = alpha.size

    # Box-specialised hyperplane-dual linear programme. Variables
    # z = [w (n), lam, a (N), s^1..s^M (n each)] with s^m >= alpha_m w - lam,
    # s^m >= 0 shared by all observations.
    il, ia, i_s = n, n + 1, n + 1 + N
    nv = n + 1 + N + M * n
    c = np.zeros(nv)
    c[il] = radius
    c[ia:ia + N] = -1.0 / N
    D = X - xmin
    r1 = np.arange(N * M)
    jj, mm = np.divmod(r1, M)
    rows_w = np.repeat(r1, n)
    cols_w = np.tile(np.arange(n), N * M)
    vals_w = (-alpha[mm][:, None] * X[jj]).ravel()
    cols_s = (i_s + mm[:, None] * n + np.arange(n)[None, :]).ravel()
    vals_s = D[jj].ravel()
    k2 = np.arange(M * n)
    m2, i2 = np.divmod(k2, n)
    r2 = N * M + k2
    rows = np.concatenate([rows_w, r1, rows_w, r2, r2, r2])
    cols = np.concatenate([cols_w, ia + jj, cols_s, i2, np.full(M * n, il), i_s + k2])
    vals = np.concatenate([vals_w, np.ones(N * M), vals_s, alpha[m2],
                           -np.ones(M * n), -np.ones(M * n)])
    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(N * M + M * n, nv))
    b_ub = np.concatenate([beta[mm], np.zeros(M * n)])
    A_eq = sparse.csr_matrix((np.ones(n), (np.zeros(n, dtype=int), np.arange(n))),
                             shape=(1, nv))
    bounds = ([(0.0, cap)] * n + [(0.0, None)] + [(None, None)] * N
              + [(0.0, None)] * (M * n))
    # The optimum is flat (a small loss of value lets w move far), so the vertex is
    # located at the tightest feasibility tolerances HiGHS accepts.
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds,
                  method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    return np.clip(res.x[:n], 0.0, cap)

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
    invalid = """
def run_model():
    try:
        surrogate_portfolio(xmin.copy(), xmax.copy(), X.copy(), coeffs.copy(), cap, radius)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_surrogate_portfolio(xmin, xmax, X, coeffs, cap, radius)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    call = "surrogate_portfolio(xmin.copy(), xmax.copy(), X.copy(), coeffs.copy(), cap, radius)"
    gold = "_oracle_surrogate_portfolio(xmin, xmax, X, coeffs, cap, radius)"
    return [
        # --- Valid: the production 1000-asset, 25-observation instance ---
        {"setup": base + """xmin, xmax, X = instance(1000, 25)
coeffs = tangents(-0.26444, 0.21543, 9)
cap, radius = 0.01, 0.01
""", "call": call, "gold_call": gold},
        # --- Valid: the six-asset, four-observation projection ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
coeffs = tangents(-0.2215, 0.166, 9)
cap, radius = 0.25, 0.01
""", "call": call, "gold_call": gold},
        # --- Valid: a sixty-asset instance with an intermediate cap ---
        {"setup": base + """xmin, xmax, X = instance(60, 8)
lo = np.sort(xmin)[:20].mean()
hi = np.sort(xmax)[::-1][:20].mean()
coeffs = tangents(lo, hi, 9)
cap, radius = 0.05, 0.02
""", "call": call, "gold_call": gold},
        # --- Boundary: cap = 1/n, so the only admissible portfolio is equal weight ---
        {"setup": base + """xmin, xmax, X = instance(5, 3)
coeffs = tangents(xmin.mean(), xmax.mean(), 5)
cap, radius = 0.2, 0.05
""", "call": call, "gold_call": gold},
        # --- Edge: radius equal to the l1 diameter of the box ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
coeffs = tangents(-0.2215, 0.166, 9)
cap, radius = 0.25, float(np.sum(xmax - xmin))
""", "call": call, "gold_call": gold},
        # --- Edge: a single asset, the portfolio fixed at full weight ---
        {"setup": base + """xmin, xmax = np.array([-0.3]), np.array([0.4])
X = np.array([[0.1], [-0.2], [0.3]])
coeffs = tangents(-0.3, 0.4, 5)
cap, radius = 1.0, 0.05
""", "call": call, "gold_call": gold},
        # --- Invalid: nonpositive radius ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
coeffs = tangents(-0.2215, 0.166, 9)
cap, radius = 0.25, 0.0
""" + invalid, "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: an observation outside the box ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
X = X.copy()
X[2, 3] = xmax[3] + 0.01
coeffs = tangents(-0.2215, 0.166, 9)
cap, radius = 0.25, 0.01
""" + invalid, "call": "run_model()", "gold_call": "run_gold()"},
    ]
