"""
Compute the discrete confidence-gap quantity associated with a finite scenario probability vector.

The probability vector represents a finite discrete distribution, and the requested quantity depends only on its subset masses and the supplied confidence level.

Returns
-------
The probability support is small and exact combinatorial subset masses determine the returned gap.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_scenario_gap(probabilities: np.ndarray, confidence: float) -> float:
    """Return the discrete confidence gap for a finite probability distribution.

    Parameters
    ----------
    probabilities : np.ndarray
        One-dimensional positive scenario probabilities that sum to one.
    confidence : float
        Confidence level in the open interval (0, 1).

    Returns
    -------
    gap : float
        Difference between the confidence level and the largest subset probability
        strictly below it.

    Raises
    ------
    ValueError
        If probabilities are invalid or confidence is outside (0, 1).
    """
    return gap

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_scenario_gap(probabilities: np.ndarray, confidence: float) -> float:
    p = np.asarray(probabilities, dtype=float)
    a = float(confidence)
    if p.ndim != 1 or p.size == 0 or np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("probabilities must be a nonempty positive finite vector")
    if not np.isclose(p.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    if not (0.0 < a < 1.0):
        raise ValueError("confidence must lie in (0,1)")
    best = 0.0
    for mask in range(1 << p.size):
        mass = float(sum(p[j] for j in range(p.size) if (mask >> j) & 1))
        if mass < a and mass > best:
            best = mass
    return float(a - best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent finite-scenario cases."""
    return [
        {"setup": "import numpy as np\np=np.array([0.05,0.10,0.15,0.20,0.22,0.28])\na=0.65", "call": "compute_scenario_gap(p,a)", "gold_call": "_oracle_compute_scenario_gap(p,a)", "tol": 1e-12},
        {"setup": "import numpy as np\np=np.array([0.2,0.3,0.5])\na=0.5", "call": "compute_scenario_gap(p,a)", "gold_call": "_oracle_compute_scenario_gap(p,a)", "tol": 1e-12},
        {"setup": "import numpy as np\np=np.array([0.125]*8)\na=0.73", "call": "compute_scenario_gap(p,a)", "gold_call": "_oracle_compute_scenario_gap(p,a)", "tol": 1e-12},
    ]
