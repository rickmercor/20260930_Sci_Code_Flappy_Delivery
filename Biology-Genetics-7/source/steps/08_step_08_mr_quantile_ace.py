"""
MEI-gated ALasso valid set followed by MR-Quantile.

Call the prior step helpers: allele_orientations, rivw_slope, mei_z_score, mei_combined_stat, adaptive_alasso_weights, alasso_valid_flags, and ratio_invse_weights. Gate on the combined MEI statistic, then run Algorithm 1 on the chosen SNP set. Import numpy inside the function; np is not predefined.

Returns
-------
Scalar MR-Quantile causal effect.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mr_quantile_ace(
    bx: np.ndarray,
    sx: np.ndarray,
    by: np.ndarray,
    sy: np.ndarray,
    eaf: np.ndarray,
    z: np.ndarray,
    rivw_lam: float,
    eta: float,
    alasso_lam: float,
    alasso_nu: float,
    mei_crit: float,
) -> float:
    """MEI-gated MR-ALasso valid set followed by MR-Quantile.

    Parameters
    ----------
    bx, sx, by, sy, eaf, z : np.ndarray
        Length-p two-sample summary table and provided rerandomization draws.
    rivw_lam, eta, alasso_lam, alasso_nu, mei_crit : float
        Locked selection threshold, rerandomization scale, ALasso penalty,
        adaptive exponent, and two-sided normal critical value.

    Returns
    -------
    float
        MR-Quantile causal-effect estimate on the gated SNP set.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_mr_quantile_ace(bx, sx, by, sy, eaf, z, rivw_lam, eta, alasso_lam, alasso_nu, mei_crit):
    lib = __import__(type(bx).__module__)
    from math import sqrt

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    sx = lib.asarray(sx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    eaf = lib.asarray(eaf, dtype=lib.float64).reshape(-1)
    z = lib.asarray(z, dtype=lib.float64).reshape(-1)

    ori = _oracle_allele_orientations(bx, by, eaf)
    g_m, G_m, g_n, G_n = ori[0], ori[1], ori[2], ori[3]
    z_m = lib.where(eaf < 0.5, -z, z)
    z_n = lib.where(bx < 0.0, -z, z)

    _oracle_rivw_slope(g_n, G_n, sx, sy, z_n, rivw_lam, eta)
    _oracle_mei_z_score(g_m, G_m, sx, sy, z_m, rivw_lam, eta)
    _oracle_mei_z_score(g_n, G_n, sx, sy, z_n, rivw_lam, eta)
    zc = _oracle_mei_combined_stat(bx, by, sx, sy, eaf, z, rivw_lam, eta)

    sel = lib.abs(g_n / sx + z_n) - rivw_lam > 0.0
    if zc > mei_crit and lib.any(sel):
        _oracle_adaptive_alasso_weights(g_n[sel], G_n[sel], alasso_nu)
        flags = _oracle_alasso_valid_flags(
            g_n[sel], G_n[sel], sy[sel], alasso_lam, alasso_nu
        )
        mask = lib.zeros(bx.size, dtype=bool)
        mask[lib.where(sel)[0][flags > 0.5]] = True
    else:
        mask = sel
    if lib.count_nonzero(mask) < 2:
        mask = sel

    rw = _oracle_ratio_invse_weights(g_n[mask], G_n[mask], sx[mask], sy[mask])
    r, w = rw[0], rw[1]
    p = float(r.size)
    tau = 0.5
    prev_ll = -lib.inf
    theta = 0.0
    lam = 1.0
    for _ in range(80):
        order = lib.argsort(r, kind="mergesort")
        rs, ws = r[order], w[order]
        cw = lib.cumsum(ws)
        idx = int(lib.searchsorted(cw, tau * cw[-1], side="left"))
        idx = min(max(idx, 0), rs.size - 1)
        theta = float(rs[idx])
        u = r - theta
        rho = lib.where(u >= 0.0, tau * u, (tau - 1.0) * u)
        lam = p / max(lib.sum(w * rho), 1e-18)
        a = lam * lib.sum(w * (r - theta))
        tau = 0.5 - a / (2.0 * (2.0 * p + sqrt(a * a + 4.0 * p * p)))
        tau = min(max(tau, 1e-8), 1.0 - 1e-8)
        u = r - theta
        rho = lib.where(u >= 0.0, tau * u, (tau - 1.0) * u)
        ll = (
            lib.sum(lib.log(w))
            + p * lib.log(max(lam * tau * (1.0 - tau), 1e-300))
            - lam * lib.sum(w * rho)
        )
        if abs(ll - prev_ll) < 1e-12:
            break
        prev_ll = ll
    return float(theta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nbx = np.array([0.2, 0.3, 0.4])\nsx = np.full(3, 0.02)\nby = np.array([0.06, 0.09, 0.12])\nsy = np.full(3, 0.03)\neaf = np.array([0.6, 0.55, 0.7])\nz = np.zeros(3)",
            "call": "mr_quantile_ace(bx, sx, by, sy, eaf, z, 3.5, 0.5, 0.06, 1.0, 1.959963984540054)",
            "gold_call": "_oracle_mr_quantile_ace(bx, sx, by, sy, eaf, z, 3.5, 0.5, 0.06, 1.0, 1.959963984540054)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.2, 0.3, 0.4, 0.01])\nsx = np.array([0.02, 0.02, 0.02, 0.02])\nby = np.array([0.06, 0.09, 0.52, 0.4])\nsy = np.array([0.03, 0.03, 0.03, 0.05])\neaf = np.array([0.6, 0.55, 0.7, 0.5])\nz = np.zeros(4)",
            "call": "mr_quantile_ace(bx, sx, by, sy, eaf, z, 3.5, 0.5, 0.06, 1.0, 1.959963984540054)",
            "gold_call": "_oracle_mr_quantile_ace(bx, sx, by, sy, eaf, z, 3.5, 0.5, 0.06, 1.0, 1.959963984540054)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.08, -0.12, 0.16, 0.20, -0.24, 0.28, 0.32, -0.36, 0.40, -0.44, 0.48, -0.52, 0.012, -0.010])\nsx = np.array([0.015]*12+[0.020, 0.020])\nby = np.array([0.0252, -0.0368, 0.0486, 0.0590, -0.0716, 0.0835, 0.0969, -0.1083, 0.6720, -0.7335, 0.6450, -0.7372, 0.1636, -0.1150])\nsy = np.array([0.015]*12+[0.025, 0.025])\neaf = np.array([0.62, 0.31, 0.71, 0.44, 0.28, 0.58, 0.67, 0.39, 0.55, 0.41, 0.73, 0.36, 0.52, 0.48])\nz = np.array([0.08, -0.05, 0.06, 0.02, -0.10, 0.04, 0.03, -0.02, 0.05, -0.03, 0.04, 0.01, 0.18, -0.14])",
            "call": "mr_quantile_ace(bx, sx, by, sy, eaf, z, 3.5, 0.5, 0.06, 1.0, 1.959963984540054)",
            "gold_call": "_oracle_mr_quantile_ace(bx, sx, by, sy, eaf, z, 3.5, 0.5, 0.06, 1.0, 1.959963984540054)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([-0.25, 0.30])\nsx = np.full(2, 0.02)\nby = np.array([-0.05, 0.06])\nsy = np.full(2, 0.03)\neaf = np.array([0.4, 0.6])\nz = np.array([0.05, -0.02])",
            "call": "mr_quantile_ace(bx, sx, by, sy, eaf, z, 3.5, 0.5, 0.06, 1.0, 1.959963984540054)",
            "gold_call": "_oracle_mr_quantile_ace(bx, sx, by, sy, eaf, z, 3.5, 0.5, 0.06, 1.0, 1.959963984540054)",
        },
    ]
