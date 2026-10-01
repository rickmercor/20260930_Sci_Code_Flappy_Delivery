"""
Build the binary force active-learning keep mask from disagreement scores.

DFT labeling budgets are finite, so only MD frames whose force-ensemble disagreement exceeds a physically motivated threshold (often tied to DFT force-convergence practice) enter the labeled set.

Returns
-------
np.ndarray shape (n_configs,), float mask in {0.0, 1.0}
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_force_labeling_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return a 0/1 mask for force-AL labeling eligibility.

    Parameters
    ----------
    scores : np.ndarray
        Shape (n_configs,). Force disagreement scores in eV/Å.
    threshold : float
        Labeling threshold in eV/Å.

    Returns
    -------
    mask : np.ndarray
        Shape (n_configs,) with values in {0.0, 1.0}.
    """
    return np.asarray([], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_force_labeling_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    import numpy as np
    u = np.asarray(scores, dtype=float).reshape(-1)
    if u.size < 1:
        raise ValueError("scores must be non-empty")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be a finite non-negative float")
    if np.any(~np.isfinite(u)) or np.any(u < 0.0):
        raise ValueError("scores must be finite and non-negative")
    return (u > float(threshold)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "scores = np.array([0.0, 0.024494897427831, 0.004082482904638])\n"
                "threshold = 0.01\n"
            ),
            "call": "compute_force_labeling_mask(scores, threshold)",
            "gold_call": "_oracle_compute_force_labeling_mask(scores, threshold)",
        },
        {
            "setup": "import numpy as np\nscores = np.array([0.0])\nthreshold = 0.01\n",
            "call": "compute_force_labeling_mask(scores, threshold)",
            "gold_call": "_oracle_compute_force_labeling_mask(scores, threshold)",
        },
        {
            "setup": "import numpy as np\nscores = np.array([0.01, 0.0100001])\nthreshold = 0.01\n",
            "call": "compute_force_labeling_mask(scores, threshold)",
            "gold_call": "_oracle_compute_force_labeling_mask(scores, threshold)",
        },
    ]
