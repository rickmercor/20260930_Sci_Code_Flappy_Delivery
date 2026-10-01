"""
Rerandomized IVW slope on S_lambda using Rao-Blackwell exposure effects.

Select SNPs with |bx/sx + z| > lam, replace bx by its Rao-Blackwell conditional mean, and form the RIVW slope. Import numpy inside the function; np is not predefined.

Returns
-------
Scalar RIVW slope on the selected set.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rivw_slope(
    bx: np.ndarray,
    by: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    z: np.ndarray,
    lam: float,
    eta: float,
) -> float:
    """Rerandomized IVW slope on the selected SNP set.

    Parameters
    ----------
    bx, by, sx, sy, z : np.ndarray
        Length-p summary statistics and provided rerandomization draws.
    lam, eta : float
        Selection threshold and rerandomization scale.

    Returns
    -------
    float
        RIVW slope on S_lambda.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rivw_slope(bx, by, sx, sy, z, lam, eta):
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
    z = lib.asarray(z, dtype=lib.float64).reshape(-1)
    sel = lib.abs(bx / sx + z) - lam > 0.0
    g_rb = lib.zeros_like(bx)
    s2_rb = lib.zeros_like(bx)
    for j in range(bx.size):
        if not sel[j]:
            continue
        ap = -bx[j] / (sx[j] * eta) + lam / eta
        am = -bx[j] / (sx[j] * eta) - lam / eta
        den = 1.0 - _Phi(ap) + _Phi(am)
        if den < 1e-15:
            g_rb[j] = bx[j]
            s2_rb[j] = sx[j] ** 2
            continue
        num = _phi(ap) - _phi(am)
        g_rb[j] = bx[j] - (sx[j] / eta) * (num / den)
        s2_rb[j] = sx[j] ** 2 * (
            1.0
            - (1.0 / eta**2) * (ap * _phi(ap) - am * _phi(am)) / den
            + (1.0 / eta**2) * (num / den) ** 2
        )
        s2_rb[j] = max(s2_rb[j], 1e-18)
    w = lib.zeros_like(bx)
    w[sel] = 1.0 / lib.maximum(sy[sel] ** 2, 1e-30)
    den = lib.sum(w * (g_rb**2 - s2_rb))
    if abs(den) < 1e-18:
        return 0.0
    return float(lib.sum(w * g_rb * by) / den)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nbx = np.array([0.2, 0.3])\nby = np.array([0.1, 0.15])\nsx = np.array([0.02, 0.02])\nsy = np.array([0.03, 0.03])\nz = np.array([0.0, 0.0])",
            "call": "rivw_slope(bx, by, sx, sy, z, 3.5, 0.5)",
            "gold_call": "_oracle_rivw_slope(bx, by, sx, sy, z, 3.5, 0.5)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.01, 0.25])\nby = np.array([0.4, 0.08])\nsx = np.array([0.02, 0.02])\nsy = np.array([0.03, 0.03])\nz = np.array([0.1, 0.0])",
            "call": "rivw_slope(bx, by, sx, sy, z, 3.5, 0.5)",
            "gold_call": "_oracle_rivw_slope(bx, by, sx, sy, z, 3.5, 0.5)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.012, 0.010])\nby = np.array([0.0252, 0.0368, 0.0486, 0.0590, 0.0716, 0.0835, 0.0969, 0.1083, 0.6720, 0.7335, 0.6450, 0.7372, 0.1636, 0.1150])\nsx = np.array([0.015]*12+[0.020, 0.020])\nsy = np.array([0.015]*12+[0.025, 0.025])\nz = np.array([0.08, 0.05, 0.06, 0.02, 0.10, 0.04, 0.03, 0.02, 0.05, 0.03, 0.04, 0.01, 0.18, 0.14])",
            "call": "rivw_slope(bx, by, sx, sy, z, 3.5, 0.5)",
            "gold_call": "_oracle_rivw_slope(bx, by, sx, sy, z, 3.5, 0.5)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.4])\nby = np.array([0.1])\nsx = np.array([0.02])\nsy = np.array([0.03])\nz = np.array([0.0])",
            "call": "rivw_slope(bx, by, sx, sy, z, 3.5, 0.5)",
            "gold_call": "_oracle_rivw_slope(bx, by, sx, sy, z, 3.5, 0.5)",
        },
    ]
