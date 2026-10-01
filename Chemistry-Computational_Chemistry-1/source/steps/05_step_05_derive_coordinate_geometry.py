"""
Derive the reduced coordinate and the geometric factors of its level sets.

X is a finite array of shape (n_points, 2), one planar point per row.



c is a finite amplitude of the coordinate's oscillatory term.



k is a finite angular frequency of the coordinate's oscillatory term.



Return the coordinate value, the two components of the Moore-Penrose pseudo-inverse of its Jacobian, and the divergence of that pseudo-inverse, in that column order. The divergence is required in full; it is not zero for this coordinate.

Returns
-------
np.ndarray of shape (n_points, 4), float: the coordinate value and the accompanying geometric components.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Derive the reduced coordinate and the geometric factors of its level sets."""

import numpy as np
from math import erf


def derive_coordinate_geometry(X: np.ndarray, c: float, k: float) -> np.ndarray:
    """Derive the reduced coordinate and the geometric factors of its level sets.

    Parameters
    ----------
    X
        Finite points with shape ``(n_points, 2)``.
    c
        Amplitude of the coordinate's oscillatory term.
    k
        Angular frequency of the coordinate's oscillatory term.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 4)`` holding the coordinate value, the two
                pseudo-inverse components, and their divergence, in that column order.

    Raises
    ------
    ValueError
        If ``X`` is not a two-dimensional array of shape ``(n_points, 2)``.
    """
    return np.empty((np.asarray(X).shape[0], 4), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_derive_coordinate_geometry(
    X,
    c,
    k,
):
    """Reference implementation for derive_coordinate_geometry."""
    import numpy as np

    P = np.asarray(X, float)
    if P.ndim != 2 or P.shape[1] != 2: raise ValueError("X must be (n_points, 2)")
    xi = P[:, 0] + c*np.sin(k*P[:, 1])
    g  = c*k*np.cos(k*P[:, 1]); s = 1.0 + g**2
    gp = -c*k*k*np.sin(k*P[:, 1])
    return np.stack([xi, 1.0/s, g/s, gp*(1.0 - g**2)/s**2], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for derive_coordinate_geometry."""
    return [
            {
                    "setup": "import numpy as np\nX = np.array([[-0.5, 1.5], [0.6, 0.02], [-0.05, 0.47]])\n",
                    "call": "derive_coordinate_geometry(X, 0.35, 1.5)",
                    "gold_call": "_oracle_derive_coordinate_geometry(X, 0.35, 1.5)"
            },
            {
                    "setup": "import numpy as np\nX = np.array([[0.0, 0.0]])\n",
                    "call": "derive_coordinate_geometry(X, 0.0, 1.5)",
                    "gold_call": "_oracle_derive_coordinate_geometry(X, 0.0, 1.5)"
            },
            {
                    "setup": "import numpy as np\nX = np.array([[0.3, np.pi/3.0]])\n",
                    "call": "derive_coordinate_geometry(X, 0.35, 1.5)",
                    "gold_call": "_oracle_derive_coordinate_geometry(X, 0.35, 1.5)"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        derive_coordinate_geometry(np.zeros(4), 0.35, 1.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_derive_coordinate_geometry(np.zeros(4), 0.35, 1.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            }
    ]
