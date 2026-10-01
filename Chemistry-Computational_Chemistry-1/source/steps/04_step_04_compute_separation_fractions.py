"""
Compute the separation fraction between two propagated mean fields.

mean_target is a finite array of shape (n_points, 2) holding the reference mean per point.



mean_draft is a finite array aligned with mean_target.



dt is a finite strictly positive step size.



gamma is a finite strictly positive relaxation rate.



mass is a finite strictly positive inertia scale.



kbt is a finite strictly positive thermal energy scale.



Apply the source-defined optimal-coupling result for two distributions that differ in their mean and share a spread set by dt, gamma, mass and kbt, and return one finite fraction in [0, 1] per point.

Returns
-------
np.ndarray of shape (n_points,), float: a fraction in [0, 1] per point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Compute the separation fraction between two propagated mean fields."""

import numpy as np
from math import erf


def compute_separation_fractions(mean_target: np.ndarray, mean_draft: np.ndarray, dt: float, gamma: float, mass: float, kbt: float) -> np.ndarray:
    """Compute the separation fraction between two propagated mean fields.

    Parameters
    ----------
    mean_target
        Reference means with shape ``(n_points, 2)``.
    mean_draft
        Comparison means aligned with ``mean_target``.
    dt
        Strictly positive step size.
    gamma
        Strictly positive relaxation rate.
    mass
        Strictly positive inertia scale.
    kbt
        Strictly positive thermal energy scale.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points,)`` holding a fraction in ``[0, 1]`` per point.

    Raises
    ------
    ValueError
        If ``mean_target`` and ``mean_draft`` differ in shape, if either is
        not of shape ``(n_points, 2)``, or if any of ``dt``, ``gamma``,
        ``mass`` or ``kbt`` is not strictly positive.
    """
    return np.empty(np.asarray(mean_target).shape[0], dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_separation_fractions(
    mean_target,
    mean_draft,
    dt,
    gamma,
    mass,
    kbt,
):
    """Reference implementation for compute_separation_fractions."""
    import numpy as np
    from math import erf

    mt = np.asarray(mean_target, float); md = np.asarray(mean_draft, float)
    if mt.shape != md.shape: raise ValueError("mean arrays must match in shape")
    if mt.ndim != 2 or mt.shape[1] != 2: raise ValueError("inputs must be (n_points, 2)")
    if min(dt, gamma, mass, kbt) <= 0: raise ValueError("dt, gamma, mass and kbt must be > 0")
    sig2 = mass*kbt*(1.0 - np.exp(-2.0*gamma*dt))
    dn = np.sqrt(np.sum((md - mt)**2, axis=1)/sig2)
    return np.array([erf(v/np.sqrt(8.0)) for v in dn])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_separation_fractions."""
    return [
            {
                    "setup": "import numpy as np\nmt = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]])\nmd = np.array([[0.0, 0.0], [0.0, 0.0], [-1.0, 0.5]])\n",
                    "call": "compute_separation_fractions(mt, md, 0.02, 1.0, 1.0, 1.0)",
                    "gold_call": "_oracle_compute_separation_fractions(mt, md, 0.02, 1.0, 1.0, 1.0)"
            },
            {
                    "setup": "import numpy as np\nmt = np.full((3, 2), 0.3)\nmd = np.full((3, 2), 0.3)\n",
                    "call": "compute_separation_fractions(mt, md, 0.02, 1.0, 1.0, 1.0)",
                    "gold_call": "_oracle_compute_separation_fractions(mt, md, 0.02, 1.0, 1.0, 1.0)"
            },
            {
                    "setup": "import numpy as np\nmt = np.zeros((2, 2))\nmd = np.array([[50.0, 0.0], [0.0, 50.0]])\n",
                    "call": "compute_separation_fractions(mt, md, 0.02, 1.0, 1.0, 1.0)",
                    "gold_call": "_oracle_compute_separation_fractions(mt, md, 0.02, 1.0, 1.0, 1.0)"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        compute_separation_fractions(np.zeros((2, 2)), np.zeros((3, 2)), 0.02, 1.0, 1.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_separation_fractions(np.zeros((2, 2)), np.zeros((3, 2)), 0.02, 1.0, 1.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        compute_separation_fractions(np.zeros((2, 2)), np.zeros((2, 2)), 0.02, 1.0, 1.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_separation_fractions(np.zeros((2, 2)), np.zeros((2, 2)), 0.02, 1.0, 1.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            }
    ]
