"""
Two-sample Wald ratios and inverse-standard-error weights.

Use the two-sample delta-method variance with no exposure-outcome covariance. Weights are 1/se, not 1/Var. Import numpy inside the function; np is not predefined.

Returns
-------
Array of shape (2, p): ratios and 1/se weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ratio_invse_weights(
    bx: np.ndarray,
    by: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
) -> np.ndarray:
    """Two-sample Wald ratios and inverse-standard-error weights.

    Parameters
    ----------
    bx, by, sx, sy : np.ndarray
        Length-p associations and standard errors.

    Returns
    -------
    np.ndarray
        Array of shape (2, p): row 0 is r_i = by/bx; row 1 is 1/se(r_i).
    """
    p = np.asarray(bx).reshape(-1).size
    return np.zeros((2, p), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ratio_invse_weights(bx, by, sx, sy):
    lib = __import__(type(bx).__module__)

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sx = lib.asarray(sx, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    r = by / bx
    var = (sy**2) / (bx**2) + (by**2) * (sx**2) / (bx**4)
    se = lib.sqrt(lib.maximum(var, 1e-30))
    return lib.vstack([r, 1.0 / se])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nbx = np.array([0.2, 0.4])\nby = np.array([0.1, 0.1])\nsx = np.array([0.02, 0.02])\nsy = np.array([0.03, 0.03])",
            "call": "ratio_invse_weights(bx, by, sx, sy)",
            "gold_call": "_oracle_ratio_invse_weights(bx, by, sx, sy)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([-0.2])\nby = np.array([-0.05])\nsx = np.array([0.01])\nsy = np.array([0.02])",
            "call": "ratio_invse_weights(bx, by, sx, sy)",
            "gold_call": "_oracle_ratio_invse_weights(bx, by, sx, sy)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36])\nby = np.array([0.0252, 0.0368, 0.0486, 0.0590, 0.0716, 0.0835, 0.0969, 0.1083])\nsx = np.full(8, 0.015)\nsy = np.full(8, 0.015)",
            "call": "ratio_invse_weights(bx, by, sx, sy)",
            "gold_call": "_oracle_ratio_invse_weights(bx, by, sx, sy)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.01, 0.50])\nby = np.array([0.02, 0.10])\nsx = np.array([0.02, 0.01])\nsy = np.array([0.05, 0.02])",
            "call": "ratio_invse_weights(bx, by, sx, sy)",
            "gold_call": "_oracle_ratio_invse_weights(bx, by, sx, sy)",
        },
    ]
