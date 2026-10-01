"""
Adaptive lasso weights from median-ratio residuals.

Start from the median Wald ratio, form residual pleiotropy, and raise its absolute value to -nu. Import numpy inside the function; np is not predefined.

Returns
-------
Length-p adaptive weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adaptive_alasso_weights(bx: np.ndarray, by: np.ndarray, nu: float) -> np.ndarray:
    """Adaptive lasso weights from median-ratio residuals.

    Parameters
    ----------
    bx, by : np.ndarray
        Length-p exposure and outcome associations.
    nu : float
        Exponent on the absolute median-ratio residual.

    Returns
    -------
    np.ndarray
        Length-p adaptive weights.
    """
    return np.zeros(np.asarray(bx).reshape(-1).size, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_adaptive_alasso_weights(bx, by, nu):
    lib = __import__(type(bx).__module__)

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    theta0 = float(lib.median(by / bx))
    alpha0 = by - theta0 * bx
    return 1.0 / lib.maximum(lib.abs(alpha0), 1e-12) ** nu

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nbx = np.array([0.2, 0.4, 0.1])\nby = np.array([0.1, 0.2, 0.15])",
            "call": "adaptive_alasso_weights(bx, by, 1.0)",
            "gold_call": "_oracle_adaptive_alasso_weights(bx, by, 1.0)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.2, -0.4])\nby = np.array([0.06, -0.12])",
            "call": "adaptive_alasso_weights(bx, by, 1.0)",
            "gold_call": "_oracle_adaptive_alasso_weights(bx, by, 1.0)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.012, 0.010])\nby = np.array([0.0252, 0.0368, 0.0486, 0.0590, 0.0716, 0.0835, 0.0969, 0.1083, 0.6720, 0.7335, 0.6450, 0.7372, 0.1636, 0.1150])",
            "call": "adaptive_alasso_weights(bx, by, 1.0)",
            "gold_call": "_oracle_adaptive_alasso_weights(bx, by, 1.0)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.5, 0.5, 0.5])\nby = np.array([0.1, 0.1, 0.4])",
            "call": "adaptive_alasso_weights(bx, by, 2.0)",
            "gold_call": "_oracle_adaptive_alasso_weights(bx, by, 2.0)",
        },
    ]
