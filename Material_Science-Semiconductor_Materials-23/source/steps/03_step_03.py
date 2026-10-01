"""
Compute the SCAN exchange enhancement factor

*The SCAN exchange enhancement at a point is set by the dimensionless density*

 *gradient s and the bond-strength indicator alpha, given the exchange*

 *parameters kappa and c1x. Its published construction interpolates between the*

 *metallic branch and a covalent limit and applies a gradient factor.*

Returns
-------
float, SCAN exchange enhancement factor at (s, alpha)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_fx_enhancement(
    s: float,
    alpha: float,
    kappa: float,
    c1x: float,
) -> float:
    """Compute the SCAN exchange enhancement factor at one grid point.

    Parameters
    ----------
    s : float
        Dimensionless density gradient (positive).
    alpha : float
        Bond-strength indicator in [0, 1].
    kappa : float
        SCAN exchange parameter kappa (positive).
    c1x : float
        SCAN exchange interpolation parameter c1x.

    Returns
    -------
    fx_enhancement : float
        SCAN exchange enhancement factor at (s, alpha).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_fx_enhancement(
    s: float,
    alpha: float,
    kappa: float,
    c1x: float,
) -> float:
    import numpy as np

    if s <= 0.0:
        raise ValueError("s must be positive")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    h1x = float(_oracle_compute_h1x_metallic_factor(s, alpha, kappa))

    if alpha >= 1.0:
        fx = 0.0
    else:
        fx = float(np.exp(-c1x * alpha / (1.0 - alpha)))

    gx = 1.0 - np.exp(-4.9479 * s ** (-0.5))
    h0x = 1.174
    return float((h1x + fx * (h0x - h1x)) * gx)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ns, alpha, kappa, c1x = 1.0, 0.30, 0.15, 0.20",
            "call": "compute_fx_enhancement(s, alpha, kappa, c1x)",
            "gold_call": "_oracle_compute_fx_enhancement(s, alpha, kappa, c1x)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa, c1x = 1.0, 0.30, 0.065, 0.667",
            "call": "compute_fx_enhancement(s, alpha, kappa, c1x)",
            "gold_call": "_oracle_compute_fx_enhancement(s, alpha, kappa, c1x)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa, c1x = 0.75, 0.0, 0.15, 0.20",
            "call": "compute_fx_enhancement(s, alpha, kappa, c1x)",
            "gold_call": "_oracle_compute_fx_enhancement(s, alpha, kappa, c1x)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa, c1x = 1.25, 0.45, 0.027, 0.15",
            "call": "compute_fx_enhancement(s, alpha, kappa, c1x)",
            "gold_call": "_oracle_compute_fx_enhancement(s, alpha, kappa, c1x)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa, c1x = 2.0, 1.0, 0.065, 0.667",
            "call": "compute_fx_enhancement(s, alpha, kappa, c1x)",
            "gold_call": "_oracle_compute_fx_enhancement(s, alpha, kappa, c1x)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa, c1x = 0.001, 0.0, 0.15, 0.20",
            "call": "compute_fx_enhancement(s, alpha, kappa, c1x)",
            "gold_call": "_oracle_compute_fx_enhancement(s, alpha, kappa, c1x)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa, c1x = 50.0, 0.5, 0.15, 0.20",
            "call": "compute_fx_enhancement(s, alpha, kappa, c1x)",
            "gold_call": "_oracle_compute_fx_enhancement(s, alpha, kappa, c1x)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ns, alpha, kappa, c1x = 1.0, 0.999, 0.5, 0.05",
            "call": "compute_fx_enhancement(s, alpha, kappa, c1x)",
            "gold_call": "_oracle_compute_fx_enhancement(s, alpha, kappa, c1x)",
            "tol": 1e-10,
        },
    ]
