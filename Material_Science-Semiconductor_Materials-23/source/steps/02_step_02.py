"""
Compute the SCAN metallic-branch exchange factor h1x.

*The metallic branch of the SCAN exchange enhancement is a bounded function of*

 *a single auxiliary variable built from the dimensionless density gradient and*

 *the bond-strength indicator. That variable collects the second-order gradient*

 *contribution together with a gradient-and-alpha term formed from the SCAN*

 *auxiliary constants, and the branch saturates at large argument in the way*

 *required by the Lieb-Oxford bound*

Returns
-------
float, SCAN metallic-branch exchange factor at (s, alpha)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_h1x_metallic_factor(
    s: float,
    alpha: float,
    kappa: float,
) -> float:
    """Compute the SCAN metallic-branch exchange factor at one grid point.

    Parameters
    ----------
    s : float
        Dimensionless density gradient (positive).
    alpha : float
        Bond-strength indicator in [0, 1].
    kappa : float
        SCAN exchange parameter kappa (positive).

    Returns
    -------
    h1x : float
        SCAN metallic-branch exchange factor at (s, alpha). Assembled by
        calling the earlier sub-problem functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_h1x_metallic_factor(
    s: float,
    alpha: float,
    kappa: float,
) -> float:
    import numpy as np

    if s <= 0.0:
        raise ValueError("s must be positive")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    b1, b2, b3, b4 = np.asarray(
        _oracle_compute_scan_auxiliary_constants(kappa), dtype=float
    )

    mu = 10.0 / 81.0
    x = mu * s**2 * (
        1.0 + (b4 * s**2 / mu) * np.exp(-abs(b4) * s**2 / mu)
    ) + (
        b1 * s**2 + b2 * (1.0 - alpha) * np.exp(-b3 * (1.0 - alpha) ** 2)
    ) ** 2
    return float(1.0 + kappa - kappa / (1.0 + x / kappa))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ns, alpha, kappa = 1.0, 0.30, 0.15",
            "call": "compute_h1x_metallic_factor(s, alpha, kappa)",
            "gold_call": "_oracle_compute_h1x_metallic_factor(s, alpha, kappa)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa = 1.0, 0.30, 0.065",
            "call": "compute_h1x_metallic_factor(s, alpha, kappa)",
            "gold_call": "_oracle_compute_h1x_metallic_factor(s, alpha, kappa)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa = 0.75, 0.0, 0.15",
            "call": "compute_h1x_metallic_factor(s, alpha, kappa)",
            "gold_call": "_oracle_compute_h1x_metallic_factor(s, alpha, kappa)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa = 1.25, 1.0, 0.027",
            "call": "compute_h1x_metallic_factor(s, alpha, kappa)",
            "gold_call": "_oracle_compute_h1x_metallic_factor(s, alpha, kappa)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa = 0.001, 0.5, 0.15",
            "call": "compute_h1x_metallic_factor(s, alpha, kappa)",
            "gold_call": "_oracle_compute_h1x_metallic_factor(s, alpha, kappa)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa = 50.0, 0.5, 0.15",
            "call": "compute_h1x_metallic_factor(s, alpha, kappa)",
            "gold_call": "_oracle_compute_h1x_metallic_factor(s, alpha, kappa)",
            "tol": 1e-10,
        },
    ]
