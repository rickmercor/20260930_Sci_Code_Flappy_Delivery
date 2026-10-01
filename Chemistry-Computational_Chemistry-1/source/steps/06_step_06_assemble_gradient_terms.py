"""
Assemble the reduced gradient observable from a gradient and the geometry.

grad_U is a finite array of shape (n_points, 2) holding a planar gradient per point.



cv_geom is a finite array of shape (n_points, 4) aligned with grad_U, laid out as produced by the geometry step.



Combine the two inputs into the source-defined reduced observable and return one finite value per point. The observable is not the gradient projected along the coordinate; the geometry contributes a further term.

Returns
-------
np.ndarray of shape (n_points,), float: the assembled observable per point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Assemble the reduced gradient observable from a gradient and the geometry."""

import numpy as np
from math import erf


def assemble_gradient_terms(grad_U: np.ndarray, cv_geom: np.ndarray) -> np.ndarray:
    """Assemble the reduced gradient observable from a gradient and the geometry.

    Parameters
    ----------
    grad_U
        Surface gradients with shape ``(n_points, 2)``.
    cv_geom
        Geometry table with shape ``(n_points, 4)`` aligned with
        ``grad_U``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points,)`` holding the assembled observable per point.

    Raises
    ------
    ValueError
        If ``grad_U`` is not of shape ``(n_points, 2)``, or if ``cv_geom`` is
        not of shape ``(n_points, 4)`` with the same number of rows.
    """
    return np.empty(np.asarray(grad_U).shape[0], dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_gradient_terms(
    grad_U,
    cv_geom,
):
    """Reference implementation for assemble_gradient_terms."""
    import numpy as np

    G = np.asarray(grad_U, float); C = np.asarray(cv_geom, float)
    if G.ndim != 2 or G.shape[1] != 2: raise ValueError("grad_U must be (n_points, 2)")
    if C.shape[0] != G.shape[0] or C.shape[1] != 4: raise ValueError("cv_geom must be (n_points, 4)")
    return C[:, 1]*G[:, 0] + C[:, 2]*G[:, 1] - C[:, 3]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for assemble_gradient_terms."""
    return [
            {
                    "setup": "import numpy as np\nG = np.array([[1.0, 2.0], [-3.0, 0.5]])\nC = np.array([[0.1, 0.9, -0.3, -0.44], [0.2, 0.78, 0.41, -0.01]])\n",
                    "call": "assemble_gradient_terms(G, C)",
                    "gold_call": "_oracle_assemble_gradient_terms(G, C)"
            },
            {
                    "setup": "import numpy as np\nG = np.zeros((2, 2))\nC = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])\n",
                    "call": "assemble_gradient_terms(G, C)",
                    "gold_call": "_oracle_assemble_gradient_terms(G, C)"
            },
            {
                    "setup": "import numpy as np\nG = np.array([[5.0, -5.0]])\nC = np.array([[0.0, 1.0, 0.0, 2.5]])\n",
                    "call": "assemble_gradient_terms(G, C)",
                    "gold_call": "_oracle_assemble_gradient_terms(G, C)"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        assemble_gradient_terms(np.zeros((2, 2)), np.zeros((2, 3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_assemble_gradient_terms(np.zeros((2, 2)), np.zeros((2, 3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            }
    ]
