"""
Compute the SCAN auxiliary exchange constants implied by kappa.

**The SCAN exchange construction fixes a small set of auxiliary constants from**



 **exact-constraint and norm conditions. Three of them are pure numbers, while**



 **the fourth is tied to the exchange parameter kappa through the second-order**



 **gradient coefficient, so among these auxiliary constants only b4 changes**



** when kappa is reparameterized.

Returns
-------
np.ndarray, shape (4,) SCAN auxiliary exchange constants [b1, b2, b3, b4]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_scan_auxiliary_constants(
    kappa: float,
) -> "np.ndarray":
    """Compute the SCAN auxiliary exchange constants [b1, b2, b3, b4].

    Parameters
    ----------
    kappa : float
        SCAN exchange parameter kappa (positive).

    Returns
    -------
    constants : "np.ndarray"
        Shape (4,) array [b1, b2, b3, b4].
    """
    return constants  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_scan_auxiliary_constants(
    kappa: float,
) -> "np.ndarray":
    import numpy as np

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    mu = 10.0 / 81.0
    b2 = np.sqrt(5913.0 / 405000.0)
    b1 = (511.0 / 13500.0) / (2.0 * b2)
    b3 = 0.5
    b4 = mu**2 / kappa - 1606.0 / 18225.0 - b1**2
    return np.array([b1, b2, b3, b4], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nkappa = 0.15",
            "call": "compute_scan_auxiliary_constants(kappa)",
            "gold_call": "_oracle_compute_scan_auxiliary_constants(kappa)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nkappa = 0.065",
            "call": "compute_scan_auxiliary_constants(kappa)",
            "gold_call": "_oracle_compute_scan_auxiliary_constants(kappa)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nkappa = 0.027",
            "call": "compute_scan_auxiliary_constants(kappa)",
            "gold_call": "_oracle_compute_scan_auxiliary_constants(kappa)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nkappa = 0.5",
            "call": "compute_scan_auxiliary_constants(kappa)",
            "gold_call": "_oracle_compute_scan_auxiliary_constants(kappa)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nkappa = 1e-4",
            "call": "compute_scan_auxiliary_constants(kappa)",
            "gold_call": "_oracle_compute_scan_auxiliary_constants(kappa)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nkappa = 100.0",
            "call": "compute_scan_auxiliary_constants(kappa)",
            "gold_call": "_oracle_compute_scan_auxiliary_constants(kappa)",
            "tol": 1e-12,
        },
    ]
