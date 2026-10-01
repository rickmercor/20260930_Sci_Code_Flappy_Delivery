"""
MR-ALasso valid-instrument flags after coordinate descent.

Penalize only the pleiotropic coefficients. A SNP is valid when the fitted pleiotropic coefficient is zero. Import numpy inside the function; np is not predefined.

Returns
-------
Length-p flags, 1.0 if valid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def alasso_valid_flags(
    bx: np.ndarray,
    by: np.ndarray,
    sy: np.ndarray,
    lam_n: float,
    nu: float,
) -> np.ndarray:
    """MR-ALasso valid-instrument flags after coordinate descent.

    Parameters
    ----------
    bx, by, sy : np.ndarray
        Length-p associations and outcome standard errors.
    lam_n, nu : float
        Locked penalty and adaptive-weight exponent.

    Returns
    -------
    np.ndarray
        Length-p array with 1.0 on SNPs whose fitted pleiotropic coefficient is zero.
    """
    return np.zeros(np.asarray(bx).reshape(-1).size, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_alasso_valid_flags(bx, by, sy, lam_n, nu):
    lib = __import__(type(bx).__module__)

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    w = 1.0 / lib.maximum(sy**2, 1e-30)
    theta0 = float(lib.median(by / bx))
    omega = 1.0 / lib.maximum(lib.abs(by - theta0 * bx), 1e-12) ** nu
    alpha = lib.zeros_like(bx)
    theta = theta0
    for _ in range(400):
        resid = by - theta * bx
        thresh = lam_n * omega / w
        alpha_new = lib.sign(resid) * lib.maximum(lib.abs(resid) - thresh, 0.0)
        den = lib.sum(w * bx**2)
        theta_new = float(lib.sum(w * bx * (by - alpha_new)) / den) if abs(den) > 1e-18 else theta
        if lib.max(lib.abs(alpha_new - alpha)) < 1e-12 and abs(theta_new - theta) < 1e-12:
            alpha = alpha_new
            break
        alpha, theta = alpha_new, theta_new
    return (lib.abs(alpha) <= 1e-10).astype(lib.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nbx = np.array([0.2, 0.3, 0.4])\nby = np.array([0.06, 0.09, 0.12])\nsy = np.full(3, 0.03)",
            "call": "alasso_valid_flags(bx, by, sy, 0.06, 1.0)",
            "gold_call": "_oracle_alasso_valid_flags(bx, by, sy, 0.06, 1.0)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.2, 0.3, 0.4])\nby = np.array([0.06, 0.09, 0.52])\nsy = np.full(3, 0.03)",
            "call": "alasso_valid_flags(bx, by, sy, 0.06, 1.0)",
            "gold_call": "_oracle_alasso_valid_flags(bx, by, sy, 0.06, 1.0)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52])\nby = np.array([0.0252, 0.0368, 0.0486, 0.0590, 0.0716, 0.0835, 0.0969, 0.1083, 0.6720, 0.7335, 0.6450, 0.7372])\nsy = np.full(12, 0.015)",
            "call": "alasso_valid_flags(bx, by, sy, 0.06, 1.0)",
            "gold_call": "_oracle_alasso_valid_flags(bx, by, sy, 0.06, 1.0)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.5, -0.4])\nby = np.array([0.1, 0.9])\nsy = np.array([0.02, 0.02])",
            "call": "alasso_valid_flags(bx, by, sy, 0.06, 1.0)",
            "gold_call": "_oracle_alasso_valid_flags(bx, by, sy, 0.06, 1.0)",
        },
    ]
