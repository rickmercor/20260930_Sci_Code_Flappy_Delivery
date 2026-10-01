"""
Evaluate the reference surface and its gradient on the supplied points.

X is a finite array of shape (n_points, 2), one planar point per row.



scale is a finite strictly positive amplitude applied to the source-defined terms.



kappa is a finite strictly positive confinement stiffness.



xc holds the two finite components of the confinement centre.



Apply the source-defined reference construction and return the value together with its two partial derivatives, in that column order.

Returns
-------
np.ndarray of shape (n_points, 3), float: the surface value and its two partial derivatives, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Evaluate the reference surface and its gradient on the supplied points."""

import numpy as np


def evaluate_reference_surface(X: np.ndarray, scale: float = 0.05, kappa: float = 2.0, xc: tuple = (-0.5, 0.75)) -> np.ndarray:
    """Evaluate the reference surface and its gradient on the supplied points.

    Parameters
    ----------
    X
        Finite points with shape ``(n_points, 2)``.
    scale
        Strictly positive amplitude applied to the Gaussian terms.
    kappa
        Strictly positive confinement stiffness.
    xc
        Two-component confinement centre.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 3)`` holding the surface value and its two
                partial derivatives, in that column order.

    Raises
    ------
    ValueError
        If ``X`` is not a two-dimensional array of shape ``(n_points, 2)``
        holding finite values, if ``scale`` or ``kappa`` is not a finite
        strictly positive number, or if ``xc`` does not hold two finite
        components.
    """
    return np.empty((np.asarray(X).shape[0], 3), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_reference_surface(
    X,
    scale=0.05,
    kappa=2.0,
    xc=(-0.5, 0.75),
):
    """Reference implementation for evaluate_reference_surface."""
    import numpy as np

    amplitude = np.array([-200.0, -100.0, -170.0, 15.0])
    a_coeff = np.array([-1.0, -1.0, -6.5, 0.7])
    b_coeff = np.array([0.0, 0.0, 11.0, 0.6])
    c_coeff = np.array([-10.0, -10.0, -6.5, 0.7])
    x_centre = np.array([1.0, 0.0, -0.5, -1.0])
    y_centre = np.array([0.0, 0.5, 1.5, 1.0])
    terms = np.arange(4)

    points = np.asarray(X, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "X must be a two-dimensional array of shape (n_points, 2)"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("X must contain only finite values")
    if not (np.isfinite(scale) and float(scale) > 0.0):
        raise ValueError("scale must be a finite number > 0")
    if not (np.isfinite(kappa) and float(kappa) > 0.0):
        raise ValueError("kappa must be a finite number > 0")

    centre = np.asarray(xc, dtype=float)
    if centre.shape != (2,) or not np.all(np.isfinite(centre)):
        raise ValueError("xc must hold two finite components")

    dx = points[:, [0]] - x_centre[None, terms]
    dy = points[:, [1]] - y_centre[None, terms]
    gauss = amplitude[None, terms] * np.exp(
        a_coeff[terms] * dx**2 + b_coeff[terms] * dx * dy + c_coeff[terms] * dy**2
    )
    offset = points - centre[None, :]

    value = float(scale) * gauss.sum(axis=1) + 0.5 * float(kappa) * np.sum(
        offset**2, axis=1
    )
    grad_x = float(scale) * (
        gauss * (2.0 * a_coeff[terms] * dx + b_coeff[terms] * dy)
    ).sum(axis=1) + float(kappa) * offset[:, 0]
    grad_y = float(scale) * (
        gauss * (b_coeff[terms] * dx + 2.0 * c_coeff[terms] * dy)
    ).sum(axis=1) + float(kappa) * offset[:, 1]

    return np.stack([value, grad_x, grad_y], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for evaluate_reference_surface."""
    return [
            {
                    "setup": "import numpy as np\nX = np.array([[-0.5, 1.5], [0.6, 0.02], [-0.05, 0.47]])\n",
                    "call": "evaluate_reference_surface(X)",
                    "gold_call": "_oracle_evaluate_reference_surface(X)"
            },
            {
                    "setup": "import numpy as np\nX = np.array([[-0.5, 0.75]])\n",
                    "call": "evaluate_reference_surface(X)",
                    "gold_call": "_oracle_evaluate_reference_surface(X)"
            },
            {
                    "setup": "import numpy as np\nX = np.array([[-0.5, 1.5], [0.6, 0.02], [-0.05, 0.47]])\n",
                    "call": "evaluate_reference_surface(X, 0.02, 5.0, (0.0, 1.0))",
                    "gold_call": "_oracle_evaluate_reference_surface(X, 0.02, 5.0, (0.0, 1.0))"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        evaluate_reference_surface(np.zeros((3, 3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_evaluate_reference_surface(np.zeros((3, 3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        evaluate_reference_surface(np.zeros((2, 2)), 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_evaluate_reference_surface(np.zeros((2, 2)), 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()"
            }
    ]
