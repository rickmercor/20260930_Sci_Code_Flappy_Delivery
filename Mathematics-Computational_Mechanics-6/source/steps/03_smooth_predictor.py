"""
Advance the state one step using only the non-impulsive part of the dynamics, producing the configuration the contact treatment is built on.

The scheme separates smooth bulk dynamics from the nonsmooth contact correction. This step is the smooth half alone.

Returns
-------
ndarray, the smooth configuration at the end of the step, same shape as u.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def smooth_predictor(u, v, a, dt):
    """Advance the state one step using only the non-impulsive part of the dynamics, producing the configuration the contact treatment is built on.

    Returns
    -------
    ndarray, the smooth configuration at the end of the step, same shape as u.

    Raises
    ------
    ValueError: if u, v and a do not share a shape, or dt is not positive.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_smooth_predictor(u, v, a, dt):
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float); a = np.asarray(a, dtype=float)
    if not (u.shape == v.shape == a.shape):
        raise ValueError("u, v and a must have the same shape")
    if dt <= 0:
        raise ValueError("dt must be positive")
    dt = float(dt)
    return u + dt * v + 0.5 * dt * dt * a

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "smooth_predictor(np.array([1.0,2.0,3.0]), np.array([-1.0,0.0,1.0]), np.array([0.5,-0.5,0.0]), 0.01)",
         "gold_call": "_oracle_smooth_predictor(np.array([1.0,2.0,3.0]), np.array([-1.0,0.0,1.0]), np.array([0.5,-0.5,0.0]), 0.01)"},   # normal
        {"setup": "import numpy as np",
         "call": "smooth_predictor(np.zeros(3), np.zeros(3), np.zeros(3), 1e-9)",
         "gold_call": "_oracle_smooth_predictor(np.zeros(3), np.zeros(3), np.zeros(3), 1e-9)"},   # boundary
        {"setup": "import numpy as np",
         "call": "smooth_predictor(np.linspace(-1,1,64), np.full(64,-5.0), np.zeros(64), 2.5e-7)",
         "gold_call": "_oracle_smooth_predictor(np.linspace(-1,1,64), np.full(64,-5.0), np.zeros(64), 2.5e-7)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        smooth_predictor(np.zeros(3), np.zeros(2), np.zeros(3), 0.01)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_smooth_predictor(np.zeros(3), np.zeros(2), np.zeros(3), 0.01)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
