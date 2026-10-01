"""
Compute the source's relative root mean squared error on a held-out reference panel.

The two matched vectors are predictions and independent reference labels on a held-out evaluation population. No pseudo-training rows enter this metric. Report the source-defined relative error as a percentage.

Returns
-------
float, the held-out relative RMSE in percent as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relative_rmse(reference: "np.ndarray", prediction: "np.ndarray") -> float:
    """Return the source Eq. 1 relative-error metric as a percentage.

    Inputs are matched, finite, one-dimensional test vectors of equal length
    at least two. Raise ValueError for invalid shapes, nonfinite entries, or
    a zero reference normalization denominator.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_relative_rmse(reference: "np.ndarray", prediction: "np.ndarray") -> float:
    """Compute source Eq. 1 in percent on an explicitly chosen test panel."""
    y, p = np.asarray(reference, float), np.asarray(prediction, float)
    if y.shape != p.shape or y.ndim != 1 or y.size < 2:
        raise ValueError("one-dimensional matched test panels are required")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("nonfinite label")
    reference_square = np.dot(y, y)
    if reference_square <= 0:
        raise ValueError("reference RMS must be positive")
    return float(100 * np.sqrt(np.dot(y - p, y - p) / reference_square))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\na=np.array([3.,4.]); b=np.array([0.,4.]); ag=a.copy(); bg=b.copy()", "call": "relative_rmse(a,b)", "gold_call": "_oracle_relative_rmse(ag,bg)", "tol": 1e-10},
        {"setup": "import numpy as np\na=np.array([-1.,1.]); b=a.copy(); ag=a.copy(); bg=b.copy()", "call": "relative_rmse(a,b)", "gold_call": "_oracle_relative_rmse(ag,bg)", "tol": 1e-10},
        {"setup": "import numpy as np\na=np.array([.1,.2,.3,1.]); b=np.array([.2,.1,.5,.8]); ag=a.copy(); bg=b.copy()", "call": "relative_rmse(a,b)", "gold_call": "_oracle_relative_rmse(ag,bg)", "tol": 1e-10},
        {"setup": 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\na=np.zeros(3); b=np.ones(3); ag=a.copy(); bg=b.copy()', "call": '_raises(lambda: relative_rmse(a,b))', "gold_call": '_raises(lambda: _oracle_relative_rmse(ag,bg))', "tol": 0},
    ]
