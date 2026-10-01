"""
Identify source and boundary (birth) coefficients from a stacked PDE/ODE weak-form system, using a cross-validated boundary-term selection that corrects the PDE-dominated joint sparse regression.

In the stacked weak-form system $G\,w=b$ the PDE block contains many more rows than the total-population ODE block, so a single sparse regression on the whole system is dominated by the PDE residual and tends to push errors into the boundary (birth) coefficients $w_\beta$, which then come out non-sparse. A cross-validation step repairs this: the source coefficients $w_f$ from the joint fit are held fixed and the birth coefficients are re-selected from the ODE block alone, $\Xi_\beta\,\hat w_\beta\approx b^{ode}-\Xi_f\,w_f$. The two birth supports $\operatorname{supp}(w_\beta)$ and $\operatorname{supp}(\hat w_\beta)$ are compared; if they disagree, the birth library is pruned to the terms they share (or to all terms either one selected when they share none) and the joint regression is repeated on the pruned library. If they agree, the candidate birth vector with the smaller relative residual $R(w)=\|b-G\,w\|_2/\|b\|_2$ of the full system is kept.

Returns
-------
np.ndarray of shape (J,): learned coefficients (w_f, w_beta) after boundary cross-validation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def boundary_bagging_regression(G: "np.ndarray", b: "np.ndarray", n_pde_rows: int, n_source_terms: int, lambdas: "np.ndarray") -> "np.ndarray":
    '''Return the learned coefficient vector w = (w_f, w_beta) after boundary cross-validation.
 
    Notation: G has J = M_f + M_beta columns (the first M_f = n_source_terms are source
    terms, the rest birth terms); rows 0..n_pde_rows-1 form the PDE block and the
    remaining rows form the ODE block, whose source and birth sub-matrices are Xi_f and
    Xi_beta and whose right-hand side is b_ode. MSTLS(A, y) denotes the modified
    sequential-thresholding least-squares estimate with threshold chosen from `lambdas`:
    minimum-norm least squares w_LS; bounds L_j = lambda max(1, ||y||/||A_j||) and
    U_j = min(1, ||y||/||A_j||)/lambda; iterate I = {j : L_j <= |w_j| <= U_j} and
    restricted least-squares refits (zeros off I) from w_LS until I stops changing;
    loss(lambda) = ||A (w^lambda - w_LS)|| / ||A w_LS|| + |I^lambda| / (number of columns);
    choose the smallest lambda attaining the minimum loss. supp(v) is the set of
    indices of nonzero entries and R(v) = ||b - G v|| / ||b|| is the relative residual of
    the full system.
 
    Procedure:
      1. (w_f, w_beta) = MSTLS(G, b).
      2. w_beta_hat = MSTLS(Xi_beta, b_ode - Xi_f w_f).
      3. If supp(w_beta_hat) != supp(w_beta): let S = supp(w_beta_hat) & supp(w_beta) if
         this intersection is nonempty, otherwise the union. Recompute
         (w_f, w_beta_S) = MSTLS(G restricted to all source columns plus the birth
         columns in S, b), and return w_f with w_beta_S placed at the birth positions in
         S and zeros at the other birth positions.
      4. Otherwise return (w_f, w_beta_hat) if R((w_f, w_beta_hat)) < R((w_f, w_beta)),
         else (w_f, w_beta).
 
    Parameters
    ----------
    G : np.ndarray
        Stacked matrix of shape (R, J) with no zero column.
    b : np.ndarray
        Stacked right-hand side of shape (R,), nonzero.
    n_pde_rows : int
        Number of PDE rows (0 < n_pde_rows < R).
    n_source_terms : int
        Number of source columns M_f (0 < M_f < J).
    lambdas : np.ndarray
        Threshold grid (positive values).
 
    Returns
    -------
    w : np.ndarray
        Learned coefficients of shape (J,), exact zeros for inactive terms.
 
    Raises
    ------
    ValueError
        If the block sizes are inconsistent with G, or any MSTLS subproblem is invalid
        (zero column, empty or nonpositive thresholds, vanishing least-squares fit).
    '''
    return w

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_boundary_bagging_regression(G: "np.ndarray", b: "np.ndarray", n_pde_rows: int, n_source_terms: int, lambdas: "np.ndarray") -> "np.ndarray":
    G = np.asarray(G, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    if G.ndim != 2 or b.size != G.shape[0]:
        raise ValueError("G must be (R, J) and b must have length R")
    R, J = G.shape
    p, mf = int(n_pde_rows), int(n_source_terms)
    if not (0 < p < R and 0 < mf < J):
        raise ValueError("block sizes inconsistent with G")
    w = _oracle_mstls_sparse_regression(G, b, lambdas)[:-1]
    w_f, w_beta = w[:mf], w[mf:]
    xi_f, xi_beta, b_ode = G[p:, :mf], G[p:, mf:], b[p:]
    w_hat = _oracle_mstls_sparse_regression(xi_beta, b_ode - xi_f @ w_f, lambdas)[:-1]
    s_joint, s_hat = w_beta != 0, w_hat != 0
    if not np.array_equal(s_joint, s_hat):
        keep = s_joint & s_hat
        if not keep.any():
            keep = s_joint | s_hat
        cols = np.concatenate([np.ones(mf, dtype=bool), keep])
        w_sub = _oracle_mstls_sparse_regression(G[:, cols], b, lambdas)[:-1]
        out = np.zeros(J)
        out[cols] = w_sub
        return out
    cand_joint = np.concatenate([w_f, w_beta])
    cand_hat = np.concatenate([w_f, w_hat])
    nb = np.linalg.norm(b)
    if np.linalg.norm(b - G @ cand_hat) / nb < np.linalg.norm(b - G @ cand_joint) / nb:
        return cand_hat
    return cand_joint

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
            "call": 'boundary_bagging_regression(G.copy(), b.copy(), 3276, 5, lams.copy())',
            "gold_call": '_oracle_boundary_bagging_regression(G.copy(), b.copy(), 3276, 5, lams.copy())',
        },
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
G, b = make_system(0.1, 7, 0.1, 26, 51)
""",
            "call": 'boundary_bagging_regression(G.copy(), b.copy(), 1326, 5, lams.copy())',
            "gold_call": '_oracle_boundary_bagging_regression(G.copy(), b.copy(), 1326, 5, lams.copy())',
        },
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
G, b = make_system(0.05, 3, 0.05, 11, 51)
""",
            "call": 'boundary_bagging_regression(G.copy(), b.copy(), 561, 5, lams.copy())',
            "gold_call": '_oracle_boundary_bagging_regression(G.copy(), b.copy(), 561, 5, lams.copy())',
        },
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
G, b = make_system(0.0, 0, 0.1, 11, 26)
""",
            "call": 'boundary_bagging_regression(G.copy(), b.copy(), 286, 5, lams.copy())',
            "gold_call": '_oracle_boundary_bagging_regression(G.copy(), b.copy(), 286, 5, lams.copy())',
        },
        {
            "setup": """import numpy as np
G = np.eye(4)[:, :3] + 0.1
b = np.ones(4)
def run_model():
    try:
        boundary_bagging_regression(G.copy(), b.copy(), 4, 1, np.array([0.01]))
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": '1',
        },
    ]
