"""
Combined MEI statistic over the two allele codings.

Compute one-coding MEI after each orientation, flipping z with the same SNPs, then take max abs. Import numpy inside the function; np is not predefined.

Returns
-------
max(|Z_major|, |Z_normal|).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mei_combined_stat(
    bx,
    by,
    sx,
    sy,
    eaf,
    z,
    lam,
    eta,
):
    """Combined MEI statistic over major-allele and normal/risk-allele coding.

    Parameters
    ----------
    bx, by, sx, sy, eaf, z
        Length-p two-sample summary table and rerandomization draws.
    lam, eta
        Selection threshold and rerandomization scale.

    Returns
    -------
    float
        max(|Z_major|, |Z_normal|).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_mei_combined_stat(bx, by, sx, sy, eaf, z, lam, eta):
    lib = __import__(type(bx).__module__)
    from math import erf, exp, pi, sqrt

    _SQRT2 = sqrt(2.0)
    _SQRT2PI = sqrt(2.0 * pi)
    _phi = lambda x: exp(-0.5 * x * x) / _SQRT2PI
    _Phi = lambda x: 0.5 * (1.0 + erf(x / _SQRT2))

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sx = lib.asarray(sx, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    eaf = lib.asarray(eaf, dtype=lib.float64).reshape(-1)
    z = lib.asarray(z, dtype=lib.float64).reshape(-1)
    g_m = lib.where(eaf < 0.5, -bx, bx)
    G_m = lib.where(eaf < 0.5, -by, by)
    z_m = lib.where(eaf < 0.5, -z, z)
    g_n = lib.where(bx < 0.0, -bx, bx)
    G_n = lib.where(bx < 0.0, -by, by)
    z_n = lib.where(bx < 0.0, -z, z)

    zs = []
    for gx, Gy, zx in ((g_m, G_m, z_m), (g_n, G_n, z_n)):
        sel = lib.abs(gx / sx + zx) - lam > 0.0
        g_rb = lib.zeros_like(gx)
        s2_rb = lib.zeros_like(gx)
        for j in range(gx.size):
            if not sel[j]:
                continue
            ap = -gx[j] / (sx[j] * eta) + lam / eta
            am = -gx[j] / (sx[j] * eta) - lam / eta
            den = 1.0 - _Phi(ap) + _Phi(am)
            if den < 1e-15:
                g_rb[j] = gx[j]
                s2_rb[j] = sx[j] ** 2
                continue
            num = _phi(ap) - _phi(am)
            g_rb[j] = gx[j] - (sx[j] / eta) * (num / den)
            s2_rb[j] = sx[j] ** 2 * (
                1.0
                - (1.0 / eta**2) * (ap * _phi(ap) - am * _phi(am)) / den
                + (1.0 / eta**2) * (num / den) ** 2
            )
            s2_rb[j] = max(s2_rb[j], 1e-18)
        idx = lib.where(sel)[0]
        if idx.size < 2:
            zs.append(0.0)
            continue
        w = 1.0 / lib.maximum(sy[idx] ** 2, 1e-30)
        g = g_rb[idx]
        G = Gy[idx]
        s2 = s2_rb[idx]
        ww = lib.zeros_like(gx)
        ww[sel] = 1.0 / lib.maximum(sy[sel] ** 2, 1e-30)
        den = lib.sum(ww * (g_rb**2 - s2_rb))
        beta_r = float(lib.sum(ww * g_rb * Gy) / den) if abs(den) > 1e-18 else 0.0
        sw_g2 = lib.sum(w * (g**2 - s2))
        sw_gG = lib.sum(w * g * G)
        lam_rc = sw_g2 * lib.sum(w * G) - sw_gG * lib.sum(w * g) + lib.sum((w**2) * s2 * G)
        u = (G - beta_r * g) * sw_g2 - (g * G - beta_r * (g**2 - s2)) * lib.sum(w * g)
        v = lib.sum((w**2) * u**2)
        if v <= 0.0:
            zs.append(0.0)
        else:
            zs.append(float(lam_rc / sqrt(v)))
    return float(max(abs(zs[0]), abs(zs[1])))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nbx = np.array([0.2, -0.3, 0.4])\nby = np.array([0.06, -0.09, 0.12])\nsx = np.full(3, 0.02)\nsy = np.full(3, 0.03)\neaf = np.array([0.6, 0.3, 0.7])\nz = np.zeros(3)",
            "call": "mei_combined_stat(bx, by, sx, sy, eaf, z, 3.5, 0.5)",
            "gold_call": "_oracle_mei_combined_stat(bx, by, sx, sy, eaf, z, 3.5, 0.5)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.2, -0.3, 0.4])\nby = np.array([0.06, 0.40, 0.12])\nsx = np.full(3, 0.02)\nsy = np.full(3, 0.03)\neaf = np.array([0.6, 0.3, 0.7])\nz = np.array([0.1, -0.1, 0.0])",
            "call": "mei_combined_stat(bx, by, sx, sy, eaf, z, 3.5, 0.5)",
            "gold_call": "_oracle_mei_combined_stat(bx, by, sx, sy, eaf, z, 3.5, 0.5)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.08, -0.12, 0.16, 0.20, -0.24, 0.28, 0.32, -0.36, 0.40, -0.44, 0.48, -0.52, 0.012, -0.010])\nby = np.array([0.0252, -0.0368, 0.0486, 0.0590, -0.0716, 0.0835, 0.0969, -0.1083, 0.6720, -0.7335, 0.6450, -0.7372, 0.1636, -0.1150])\nsx = np.array([0.015]*12+[0.020, 0.020])\nsy = np.array([0.015]*12+[0.025, 0.025])\neaf = np.array([0.62, 0.31, 0.71, 0.44, 0.28, 0.58, 0.67, 0.39, 0.55, 0.41, 0.73, 0.36, 0.52, 0.48])\nz = np.array([0.08, -0.05, 0.06, 0.02, -0.10, 0.04, 0.03, -0.02, 0.05, -0.03, 0.04, 0.01, 0.18, -0.14])",
            "call": "mei_combined_stat(bx, by, sx, sy, eaf, z, 3.5, 0.5)",
            "gold_call": "_oracle_mei_combined_stat(bx, by, sx, sy, eaf, z, 3.5, 0.5)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.25, 0.30])\nby = np.array([0.05, 0.06])\nsx = np.full(2, 0.02)\nsy = np.full(2, 0.03)\neaf = np.array([0.55, 0.60])\nz = np.zeros(2)",
            "call": "mei_combined_stat(bx, by, sx, sy, eaf, z, 3.5, 0.5)",
            "gold_call": "_oracle_mei_combined_stat(bx, by, sx, sy, eaf, z, 3.5, 0.5)",
        },
    ]
