"""
Compute the propagated mean of the stochastic update for a supplied force.

p_prev is a finite array of shape (n_points, 2) holding the incoming value per point.



force is a finite array aligned with p_prev.



dt is a finite strictly positive step size.



gamma is a finite strictly positive relaxation rate.



Apply the source-defined update rule for the block in which the force enters, and return the mean it produces. Only the mean is required; the associated spread is fixed elsewhere.

Returns
-------
np.ndarray of shape (n_points, 2), float: the propagated mean for each point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Compute the propagated mean of the stochastic update for a supplied force."""

import numpy as np
from math import erf


def compute_propagated_means(p_prev: np.ndarray, force: np.ndarray, dt: float, gamma: float) -> np.ndarray:
    """Compute the propagated mean of the stochastic update for a supplied force.

    Parameters
    ----------
    p_prev
        Incoming values with shape ``(n_points, 2)``.
    force
        Applied force aligned with ``p_prev``.
    dt
        Strictly positive step size.
    gamma
        Strictly positive relaxation rate.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 2)`` holding the propagated mean for each point.

    Raises
    ------
    ValueError
        If ``p_prev`` and ``force`` differ in shape, if either is not of
        shape ``(n_points, 2)``, or if ``dt`` or ``gamma`` is not strictly
        positive.
    """
    return np.empty_like(np.asarray(p_prev, dtype=float))  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_propagated_means(
    p_prev,
    force,
    dt,
    gamma,
):
    """Reference implementation for compute_propagated_means."""
    import numpy as np

    p = np.asarray(p_prev, float); F = np.asarray(force, float)
    if p.shape != F.shape: raise ValueError("p_prev and force must match in shape")
    if p.ndim != 2 or p.shape[1] != 2: raise ValueError("inputs must be (n_points, 2)")
    if dt <= 0 or gamma <= 0: raise ValueError("dt and gamma must be > 0")
    a = np.exp(-gamma*dt)
    return a*p + (1.0 + a)*(dt/2.0)*F

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_propagated_means."""
    return [
            {
                    "setup": "import numpy as np\np = np.zeros((3, 2))\nF = np.array([[1.0, -2.0], [0.5, 0.25], [-3.0, 4.0]])\n",
                    "call": "compute_propagated_means(p, F, 0.02, 1.0)",
                    "gold_call": "_oracle_compute_propagated_means(p, F, 0.02, 1.0)"
            },
            {
                    "setup": "import numpy as np\np = np.full((2, 2), 0.7)\nF = np.zeros((2, 2))\n",
                    "call": "compute_propagated_means(p, F, 0.02, 1.0)",
                    "gold_call": "_oracle_compute_propagated_means(p, F, 0.02, 1.0)"
            },
            {
                    "setup": "import numpy as np\np = np.ones((2, 2))\nF = np.ones((2, 2))\n",
                    "call": "compute_propagated_means(p, F, 1.0, 10.0)",
                    "gold_call": "_oracle_compute_propagated_means(p, F, 1.0, 10.0)"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        compute_propagated_means(np.zeros((2, 2)), np.zeros((3, 2)), 0.02, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_propagated_means(np.zeros((2, 2)), np.zeros((3, 2)), 0.02, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        compute_propagated_means(np.zeros((2, 2)), np.zeros((2, 2)), 0.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_propagated_means(np.zeros((2, 2)), np.zeros((2, 2)), 0.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            }
    ]
