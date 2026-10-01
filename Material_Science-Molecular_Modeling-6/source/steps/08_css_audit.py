"""
Build the two declared datasets from the integer constructors, run the full CSS pipeline on each with the declared cutoffs, Krylov dimensions and Fermi parameters, and report per dataset: the band energy Tr(rho H); the electron count Tr(rho S); the Frobenius error of the CSS density against the exact Fermi-Dirac density; the Frobenius error of the CSS overlap inverse square root against the exact one; the two color counts; and the probe quadratic form u^T rho u. Assemble by calling the earlier sub-problem functions.

The audit quantifies what the chromatic compression preserves and what it discards on a deliberately small chain, against exact dense references.

Returns
-------
return (2, 7) float64: audit rows per dataset
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def css_audit(v, v_s):
    """v, v_s: Krylov dimensions for the density and overlap branches.
    Builds the two declared datasets, runs the CSS pipeline with the declared
    parameters, and returns a float64 array (2, 7) with columns: band energy
    Tr(rho H); electron count Tr(rho S); Frobenius error of rho against the
    exact Fermi-Dirac density; Frobenius error of the CSS overlap inverse
    square root against the exact one; density color count; overlap color
    count; and the probe quadratic form u^T rho u. Assembled by calling the
    earlier sub-problem functions. Raises ValueError on invalid Krylov
    dimensions."""
    return np.zeros((2, 7))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8 (final orchestrator): CSS audit over the two declared datasets."""

import numpy as np

_N = 20
_DCUT_RHO = 3.0
_DCUT_S = 5.0
_BETA = 6.0


def _build(variant):
    a = 3 if variant == 1 else 5
    H = np.zeros((_N, _N))
    S = np.zeros((_N, _N))
    for i in range(_N):
        for j in range(_N):
            if abs(i - j) <= 3:
                H[i, j] = (((i + 1) * (j + 1) + a) % 13) / 13.0 - 0.5
            if abs(i - j) <= 2 and i != j:
                S[i, j] = (((i + 2) * (j + 2) + 2 * a) % 7) / 35.0 - 0.1
        S[i, i] = 1.4 + (((i + 1) * a) % 5) / 25.0
    H = (H + H.T) / 2.0
    S = (S + S.T) / 2.0
    mu = 0.15 if variant == 1 else -0.1
    return H, S, mu


def _distances():
    idx = np.arange(_N, dtype=float)
    return np.abs(idx[:, None] - idx[None, :])


def _probe():
    return np.array([((3 * (i + 1)) % 7 - 3) / 4.0 for i in range(_N)])


def _exact(H, S, beta, mu):
    lam, V = np.linalg.eigh(S)
    Sinvh = V @ np.diag(lam ** -0.5) @ V.T
    Hp = Sinvh @ H @ Sinvh
    Hp = (Hp + Hp.T) / 2.0
    eps, W = np.linalg.eigh(Hp)
    C = Sinvh @ W
    f = 1.0 / (1.0 + np.exp(beta * (eps - mu)))
    return C @ np.diag(f) @ C.T, Sinvh


def _oracle_css_audit(v, v_s):
    for x in (v, v_s):
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1:
            raise ValueError("invalid Krylov dimension")
    D = _distances()
    u = _probe()
    rows = []
    for variant in (1, 2):
        H, S, mu = _build(variant)
        rho = _oracle_css_density(H, S, D, _DCUT_RHO, _DCUT_S, v, v_s, _BETA, mu)
        P_exact, Sinvh_exact = _exact(H, S, _BETA, mu)
        Shat = _oracle_css_inv_sqrt(S, D, _DCUT_S, v_s)
        eband = float(np.trace(rho @ H))
        ne = float(np.trace(rho @ S))
        err_p = float(np.linalg.norm(rho - P_exact))
        err_s = float(np.linalg.norm(Shat - Sinvh_exact))
        nc_r = float(_oracle_greedy_multicoloring(D, _DCUT_RHO).max() + 1)
        nc_s = float(_oracle_greedy_multicoloring(D, _DCUT_S).max() + 1)
        sp = float(u @ rho @ u)
        rows.append([eband, ne, err_p, err_s, nc_r, nc_s, sp])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'v = 5\nv_s = 3', "call": "css_audit(v, v_s)", "gold_call": "_oracle_css_audit(v, v_s)", "tol": 1e-08},
        {"setup": 'v = 2\nv_s = 2', "call": "css_audit(v, v_s)", "gold_call": "_oracle_css_audit(v, v_s)", "tol": 1e-08},
        {"setup": 'v = 4\nv_s = 4', "call": "css_audit(v, v_s)", "gold_call": "_oracle_css_audit(v, v_s)", "tol": 1e-08},
        {"setup": 'v = 1\nv_s = 1', "call": "css_audit(v, v_s)", "gold_call": "_oracle_css_audit(v, v_s)", "tol": 1e-08},
        {"setup": 'v = 20\nv_s = 20', "call": "css_audit(v, v_s)", "gold_call": "_oracle_css_audit(v, v_s)", "tol": 1e-08},
    ]
