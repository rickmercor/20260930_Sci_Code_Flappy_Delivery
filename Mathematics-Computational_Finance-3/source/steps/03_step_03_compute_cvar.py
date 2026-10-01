"""
Compute a probability-weighted upper-tail conditional risk value from finite scenario losses.

The scenario probabilities weight the discrete loss support, and the risk value is determined from the specified confidence level.

Returns
-------
The function accepts an arbitrary confidence level in the open unit interval.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_cvar(losses: np.ndarray, probabilities: np.ndarray, beta: float) -> float:
    """Return the finite-scenario upper-tail conditional risk value.

    Parameters
    ----------
    losses : np.ndarray
        One-dimensional scenario loss values.
    probabilities : np.ndarray
        Positive scenario probabilities summing to one.
    beta : float
        Tail confidence level in (0, 1).

    Returns
    -------
    value : float
        Minimum Rockafellar-Uryasev finite-scenario objective value.

    Raises
    ------
    ValueError
        If dimensions, probabilities, or beta are invalid.
    """
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_cvar(losses: np.ndarray, probabilities: np.ndarray, beta: float) -> float:
    L = np.asarray(losses, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    b = float(beta)
    if L.ndim != 1 or p.ndim != 1 or L.size == 0 or L.size != p.size:
        raise ValueError("losses and probabilities must be compatible vectors")
    if np.any(~np.isfinite(L)) or np.any(~np.isfinite(p)) or np.any(p <= 0.0) or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("invalid scenario probabilities or losses")
    if not (0.0 < b < 1.0):
        raise ValueError("beta must lie in (0,1)")
    vals = np.unique(np.sort(L))
    return float(min(float(u + p @ np.maximum(L-u, 0.0)/(1.0-b)) for u in vals))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent CVaR cases."""
    return [
        {"setup": "import numpy as np\nL=np.array([-1.,0.,2.,3.]); p=np.array([0.1,0.2,0.3,0.4]); b=0.65", "call": "compute_cvar(L,p,b)", "gold_call": "_oracle_compute_cvar(L,p,b)", "tol": 1e-12},
        {"setup": "import numpy as np\nL=np.array([1.,1.,1.,1.]); p=np.array([0.05,0.15,0.30,0.50]); b=0.99", "call": "compute_cvar(L,p,b)", "gold_call": "_oracle_compute_cvar(L,p,b)", "tol": 1e-12},
        {"setup": "import numpy as np\nL=np.array([-2.,-1.,0.,1.,2.,4.]); p=np.array([0.08,0.09,0.12,0.16,0.24,0.31]); b=0.50", "call": "compute_cvar(L,p,b)", "gold_call": "_oracle_compute_cvar(L,p,b)", "tol": 1e-12},
    ]
