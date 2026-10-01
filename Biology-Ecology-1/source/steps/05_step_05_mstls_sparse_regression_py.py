"""
Solve a sparse regression problem with modified sequential-thresholding least squares (MSTLS) and select the sparsity threshold by minimizing a fit-plus-sparsity loss over a grid.

Weak-form sparse identification seeks a coefficient vector $w$ with few nonzero entries such that $G\,w\approx b$, balancing the residual $\|b-G\,w\|_2$ against the number of active library terms. Sequential-thresholding least squares repeatedly discards coefficients whose magnitude is outside an admissible band and refits the remaining ones by least squares. The modified variant makes the band scale-aware: a column $G_j$ that is large relative to $b$ may carry a small coefficient, and a small column needs a large coefficient, so both a lower and an upper bound on $|w_j|$ are set from the ratio $\|b\|_2/\|G_j\|_2$. The threshold $\lambda$ itself is chosen by a loss that adds the relative distance of the sparse fit from the full least-squares fit to the fraction of retained terms; the smallest $\lambda$ that attains the minimum loss is selected.

Returns
-------
np.ndarray of shape (J + 1,): MSTLS coefficients w followed by the selected threshold lambda_hat
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mstls_sparse_regression(G: "np.ndarray", b: "np.ndarray", lambdas: "np.ndarray") -> "np.ndarray":
    '''Return the MSTLS coefficient vector followed by the selected threshold.
 
    Let J be the number of columns, w_LS the minimum-norm least-squares solution of
    G w ~ b, and G_j the j-th column (all norms are Euclidean). For a threshold lambda:
      * bounds  L_j = lambda * max(1, ||b|| / ||G_j||),
                U_j = (1 / lambda) * min(1, ||b|| / ||G_j||);
      * iterate from w^0 = w_LS and I^{-1} = {0..J-1}:
            I^l     = {j : L_j <= |w^l_j| <= U_j},
            w^{l+1} = minimum-norm least-squares solution of G w ~ b with w_j = 0 for j
                      not in I^l (w^{l+1} = 0 if I^l is empty),
        stopping at the first l with I^l = I^{l-1}; the result w^lambda is the current
        iterate w^l (restricted least squares on the stable index set).
    The loss is
        loss(lambda) = ||G (w^lambda - w_LS)|| / ||G w_LS|| + |I^lambda| / J,
    where |I^lambda| is the size of the final index set. The selected threshold is the
    smallest lambda in `lambdas` whose loss equals the minimum loss over `lambdas`.
 
    Parameters
    ----------
    G : np.ndarray
        Matrix of shape (R, J) with R >= 1, J >= 1 and no zero column.
    b : np.ndarray
        Right-hand side of shape (R,).
    lambdas : np.ndarray
        1-D nonempty array of positive thresholds (any order).
 
    Returns
    -------
    result : np.ndarray
        Float array of shape (J + 1,): result[:J] is w^lambda_hat, with exact zeros
        outside the final index set, and result[J] is lambda_hat.
 
    Raises
    ------
    ValueError
        If shapes are inconsistent, G has a zero column, lambdas is empty or contains a
        nonpositive value, or G w_LS = 0 (the loss is undefined).
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _restricted_lstsq(G: "np.ndarray", b: "np.ndarray", support: "np.ndarray") -> "np.ndarray":
    w = np.zeros(G.shape[1])
    if support.any():
        w[support] = np.linalg.lstsq(G[:, support], b, rcond=None)[0]
    return w
 
def _oracle_mstls_sparse_regression(G: "np.ndarray", b: "np.ndarray", lambdas: "np.ndarray") -> "np.ndarray":
    G = np.asarray(G, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    lams = np.asarray(lambdas, dtype=float).ravel()
    if G.ndim != 2 or G.shape[0] < 1 or G.shape[1] < 1 or b.size != G.shape[0]:
        raise ValueError("G must be (R, J) and b must have length R")
    if lams.size == 0 or np.any(~(lams > 0)):
        raise ValueError("lambdas must be a nonempty array of positive values")
    col = np.linalg.norm(G, axis=0)
    if np.any(col == 0):
        raise ValueError("G has a zero column")
    J = G.shape[1]
    w_ls = np.linalg.lstsq(G, b, rcond=None)[0]
    fit_ls = np.linalg.norm(G @ w_ls)
    if fit_ls == 0:
        raise ValueError("G w_LS vanishes; loss undefined")
    ratio = np.linalg.norm(b) / col
    best = None
    for lam in lams:
        lower = lam * np.maximum(1.0, ratio)
        upper = np.minimum(1.0, ratio) / lam
        w = w_ls.copy()
        prev = np.ones(J, dtype=bool)
        while True:
            cur = (np.abs(w) >= lower) & (np.abs(w) <= upper)
            if np.array_equal(cur, prev):
                break
            w = _restricted_lstsq(G, b, cur)
            prev = cur
        loss = np.linalg.norm(G @ (w - w_ls)) / fit_ls + np.count_nonzero(prev) / J
        if best is None or loss < best[0] or (loss == best[0] and lam < best[1]):
            best = (loss, float(lam), w)
    return np.append(best[2], best[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
lams = 10.0 ** (-4.0 + 4.0 * np.arange(50) / 49.0)
def make_system(ratio, seed, h, kt, ka):
    a = np.arange(int(round(25.0 / h)) + 1) * h
    nt = int(round(5.0 / h))
    t = np.arange(nt + 1) * h
    wa = np.full(a.size, h); wa[0] = wa[-1] = h / 2
    wt = np.full(t.size, h); wt[0] = wt[-1] = h / 2
    n = np.zeros((nt + 1, a.size))
    n[0] = np.where(a <= 15.0, 1.0 - np.cos(2.0 * np.pi * a / 15.0), 0.0)
    surv = np.exp(-0.1 * np.exp(0.08 * a[:-1]) * np.expm1(0.08 * h) / 0.08)
    beta = np.exp(-(a - 10.0) ** 2 / 50.0)
    for i in range(nt):
        n[i + 1, 1:] = n[i, :-1] * surv
        n[i + 1, 0] = np.dot(wa[1:] * beta[1:], n[i + 1, 1:]) / (1.0 - wa[0] * beta[0])
    x = 1.0 if ratio == 0 else brentq(lambda y: y ** 4 - 2 * y + 1 - ratio, 1.0, 10.0, xtol=1e-15)
    sig = np.sqrt(2.0 * np.log(x))
    n = n * np.exp(sig * np.random.default_rng(seed).standard_normal(n.shape))
    def bump(x, starts, ell):
        u = (x[None, :] - starts[:, None]) / ell
        ins = (u > 0) & (u < 1)
        q = np.where(ins, 4 * u * (1 - u), 0.0)
        return q ** 14, np.where(ins, 14 * q ** 13 * 4 * (1 - 2 * u) / ell, 0.0)
    ph, dph = bump(t, np.arange(kt) * 2.5 / (kt - 1), 2.5)
    ps, dps = bump(a, np.arange(ka) * 12.5 / (ka - 1), 12.5)
    W = wt[:, None] * wa[None, :]
    ip = lambda A, B, F: (A @ (F * W) @ B.T).ravel()
    rates = [0.08, 0.4, 0.72, 1.04, 1.36]
    b_pde = -ip(dph, ps, n) - ip(ph, dps, n)
    G_pde = np.column_stack([ip(ph, ps, np.exp(c * a) * n) for c in rates] + [np.zeros(kt * ka)] * 3)
    m = n / np.exp(sig ** 2 / 2)
    b_ode = -(dph * wt) @ (m @ wa)
    G_ode = np.column_stack([(ph * wt) @ (m @ (wa * np.exp(c * a))) for c in rates] + [(ph * wt) @ (m @ (wa * np.exp(-(a - mu) ** 2 / 50.0))) for mu in (5.0, 10.0, 15.0)])
    return np.vstack([G_pde, G_ode]), np.concatenate([b_pde, b_ode])
G, b = make_system(0.1, 5, 0.05, 26, 126)
""",
            "call": 'mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
            "gold_call": '_oracle_mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
        },
        {
            "setup": """import numpy as np
lams = 10.0 ** (-4.0 + 4.0 * np.arange(50) / 49.0)
rng = np.random.default_rng(0)
G = rng.standard_normal((200, 8))
w_true = np.array([0.0, 1.5, 0.0, 0.0, -0.7, 0.0, 0.0, 2.0])
b = G @ w_true + 0.02 * rng.standard_normal(200)
""",
            "call": 'mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
            "gold_call": '_oracle_mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
        },
        {
            "setup": """import numpy as np
lams = 10.0 ** (-4.0 + 4.0 * np.arange(50) / 49.0)
rng = np.random.default_rng(211)
G = rng.standard_normal((120, 6)) * 10.0 ** np.array([0.0, -2.0, -3.0, 0.0, 1.0, 2.0])
b = G @ np.array([-0.5, 40.0, 0.0, 0.0, 0.0, 0.01]) + 0.3 * rng.standard_normal(120)
""",
            "call": 'mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
            "gold_call": '_oracle_mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
        },
        {
            "setup": """import numpy as np
lams = 10.0 ** (-4.0 + 4.0 * np.arange(50) / 49.0)
s = np.linspace(0.0, 25.0, 60)
G = np.column_stack([np.exp(-(s - mu) ** 2 / 50.0) * (1.0 + 0.1 * np.sin(s)) for mu in (5.0, 10.0, 15.0)])
rng = np.random.default_rng(2)
b = G @ np.array([0.0, 1.0, 0.0]) + 0.05 * rng.standard_normal(60)
""",
            "call": 'mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
            "gold_call": '_oracle_mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3)
G = rng.standard_normal((30, 4))
b = G @ np.array([1.0, 0.5, 0.0, 0.2])
lams = np.array([0.9, 0.95, 1.0])
""",
            "call": 'mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
            "gold_call": '_oracle_mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
G = rng.standard_normal((80, 5))
b = G @ np.array([2.0, 0.0, 0.0, -1.0, 0.0]) + 0.01 * rng.standard_normal(80)
lams = np.array([0.05, 0.001, 0.02, 0.3, 0.0005, 0.01])
""",
            "call": 'mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
            "gold_call": '_oracle_mstls_sparse_regression(G.copy(), b.copy(), lams.copy())',
        },
        {
            "setup": """import numpy as np
G = np.array([[1.0, 0.0], [2.0, 0.0], [0.5, 0.0]])
b = np.array([1.0, 2.0, 0.4])
def run_model():
    try:
        mstls_sparse_regression(G.copy(), b.copy(), np.array([0.01]))
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": '1',
        },
    ]
