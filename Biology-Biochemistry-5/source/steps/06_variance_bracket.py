"""
Combine the four moment bounds into the guaranteed bracket [V_plus, V_minus] of the transient variance E[X^2] - (E[X])^2, pairing the second-moment and mean bounds the way the source does so that each side of the bracket is a valid bound.

The variance is not a monomial moment, so its bounds must be assembled from the bounds on the first two moments. Which upper bound is paired with which lower bound decides whether the result is a bound at all.

Returns
-------
ndarray of float64, shape (2,), [V_plus, V_minus].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variance_bracket(y1_plus: float, y1_minus: float, y2_plus: float, y2_minus: float) -> "np.ndarray":
    """Combine the four moment bounds into the guaranteed bracket [V_plus, V_minus] of the transient variance E[X^2] - (E[X])^2, pairing the second-moment and mean bounds the way the source does so that each side of the bracket is a valid bound.

    Parameters
    ----------
    y1_plus : float
        Upper bound on E[X(t)].
    y1_minus : float
        Lower bound on E[X(t)].
    y2_plus : float
        Upper bound on E[X(t)^2].
    y2_minus : float
        Lower bound on E[X(t)^2].

    Returns
    -------
    v : np.ndarray
        Array [V_plus, V_minus].

    Raises
    ------
    ValueError
        If any bound is not finite, or an upper bound lies below its lower bound.
    """
    return v

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import comb
from scipy.linalg import expm


def _oracle_variance_bracket(y1_plus: float, y1_minus: float, y2_plus: float, y2_minus: float) -> "np.ndarray":
    """Sec. IV-A: guaranteed bracket [V-, V+] of V[X(t)] = E[X^2] - (E[X])^2 from the moment bounds.

    The upper variance bound pairs the UPPER second-moment bound with the LOWER mean bound,
    and the lower variance bound pairs the opposite pair.
    """
    vals = np.array([y1_plus, y1_minus, y2_plus, y2_minus], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError("moment bounds must be finite")
    if y1_plus < y1_minus or y2_plus < y2_minus:
        raise ValueError("each upper bound must be at least its lower bound")
    v_plus = y2_plus - y1_minus ** 2
    v_minus = y2_minus - y1_plus ** 2
    return np.array([v_plus, v_minus], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ny1_plus, y1_minus, y2_plus, y2_minus = 17.7686533166, 8.8501308318, 828.0557828036, 101.6466168274\n",
            "call": "np.asarray(variance_bracket(y1_plus, y1_minus, y2_plus, y2_minus))",
            "gold_call": "np.asarray(_oracle_variance_bracket(y1_plus, y1_minus, y2_plus, y2_minus))",
        },
        {
            "setup": "import numpy as np\ny1_plus, y1_minus, y2_plus, y2_minus = 3.0, 3.0, 9.0, 9.0\n",
            "call": "np.asarray(variance_bracket(y1_plus, y1_minus, y2_plus, y2_minus))",
            "gold_call": "np.asarray(_oracle_variance_bracket(y1_plus, y1_minus, y2_plus, y2_minus))",
        },
        {
            "setup": "import numpy as np\ny1_plus, y1_minus, y2_plus, y2_minus = 2.5, 0.0, 6.0, 0.0\n",
            "call": "np.asarray(variance_bracket(y1_plus, y1_minus, y2_plus, y2_minus))",
            "gold_call": "np.asarray(_oracle_variance_bracket(y1_plus, y1_minus, y2_plus, y2_minus))",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        variance_bracket(1.0, 2.0, 4.0, 3.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_variance_bracket(1.0, 2.0, 4.0, 3.0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
