"""
Aggregate a weighted conditional average of a quantity over uniform bins.

cv is a finite one-dimensional array of coordinate values.



values is a finite array aligned with cv, holding the quantity to average.



logw is a finite array aligned with cv, holding natural-log weights.



edges is a monotone one-dimensional array of length n_bins + 1.



Return the weight-normalised average of values within each bin. Weights arrive as logarithms and must be exponentiated without overflowing for large spreads. No further reweighting is applied.

Returns
-------
np.ndarray of shape (n_bins,), float: the weighted average in each bin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Aggregate a weighted conditional average of a quantity over uniform bins."""

import numpy as np
from math import erf


def aggregate_binned_averages(cv: np.ndarray, values: np.ndarray, logw: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Aggregate a weighted conditional average of a quantity over uniform bins.

    Parameters
    ----------
    cv
        Coordinate values with shape ``(n_points,)``.
    values
        Quantity to average, aligned with ``cv``.
    logw
        Natural-log weights, aligned with ``cv``.
    edges
        Monotone bin edges with shape ``(n_bins + 1,)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_bins,)`` holding the weighted average in each bin.

    Raises
    ------
    ValueError
        If ``cv``, ``values`` and ``logw`` do not share one shape, or if
        ``edges`` is not one-dimensional with at least two entries.
    """
    return np.empty(np.asarray(edges).size - 1, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_aggregate_binned_averages(
    cv,
    values,
    logw,
    edges,
):
    """Reference implementation for aggregate_binned_averages."""
    import numpy as np

    y = np.asarray(cv, float); v = np.asarray(values, float); a = np.asarray(logw, float)
    e = np.asarray(edges, float)
    if not (y.shape == v.shape == a.shape): raise ValueError("cv, values and logw must match in shape")
    if e.ndim != 1 or e.size < 2: raise ValueError("edges must be 1-D with >= 2 entries")
    idx = np.digitize(y, e) - 1
    out = np.full(e.size - 1, np.nan)
    for b in range(e.size - 1):
        sel = idx == b
        if not np.any(sel): continue
        aa = a[sel]; w = np.exp(aa - aa.max())
        out[b] = float(np.sum(w*v[sel])/np.sum(w))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for aggregate_binned_averages."""
    return [
            {
                    "setup": "import numpy as np\ncv = np.array([-0.9, -0.2, 0.1, 0.8])\nv = np.array([1.0, 2.0, 3.0, 4.0])\nlw = np.array([0.0, -1.0, -2.0, -0.5])\ne = np.linspace(-1.0, 1.0, 3)\n",
                    "call": "aggregate_binned_averages(cv, v, lw, e)",
                    "gold_call": "_oracle_aggregate_binned_averages(cv, v, lw, e)"
            },
            {
                    "setup": "import numpy as np\ncv = np.array([0.0, 0.0])\nv = np.array([5.0, 7.0])\nlw = np.zeros(2)\ne = np.array([-1.0, 1.0])\n",
                    "call": "aggregate_binned_averages(cv, v, lw, e)",
                    "gold_call": "_oracle_aggregate_binned_averages(cv, v, lw, e)"
            },
            {
                    "setup": "import numpy as np\ncv = np.array([-0.5, 0.5])\nv = np.array([1.0, 3.0])\nlw = np.array([700.0, 700.0])\ne = np.linspace(-1.0, 1.0, 3)\n",
                    "call": "aggregate_binned_averages(cv, v, lw, e)",
                    "gold_call": "_oracle_aggregate_binned_averages(cv, v, lw, e)"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        aggregate_binned_averages(np.zeros(3), np.zeros(2), np.zeros(3), np.linspace(0, 1, 3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_aggregate_binned_averages(np.zeros(3), np.zeros(2), np.zeros(3), np.linspace(0, 1, 3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        aggregate_binned_averages(np.zeros(3), np.zeros(3), np.zeros(3), np.array([0.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_aggregate_binned_averages(np.zeros(3), np.zeros(3), np.zeros(3), np.array([0.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            }
    ]
