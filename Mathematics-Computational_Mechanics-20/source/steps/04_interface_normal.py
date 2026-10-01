"""
Return the interface normal of a two-dimensional periodic field, computed from the interface distance-like function rather than from the volume fraction itself, as the source recommends. The gradient is taken with second-order central differences along each axis on the periodic grid, and the normal is that gradient divided by its own magnitude; where the magnitude is exactly zero the normal is the zero vector.

The distance-like function is smoother than the volume fraction across the interface, so a normal built from it is more accurate, and the source states that the two are analytically equivalent on the equilibrium profile.

Returns
-------
A (2, N, N) float64 array: entry 0 the x component and entry 1 the y component of the unit normal, the zero vector wherever the gradient magnitude is exactly zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_normal(psi: "np.ndarray", h: float) -> "np.ndarray":
    """Return the interface normal of a two-dimensional periodic field from the distance-like
    function, using second-order central differences along each axis and normalising the
    gradient by its own magnitude.

    Args:
        psi: Interface distance-like function on the (N, N) periodic grid; index [i, j]
            refers to cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        h: Grid spacing, positive, the same along both axes.

    Returns:
        A (2, N, N) float64 array: entry 0 the x component and entry 1 the y component of the
        unit normal, the zero vector wherever the gradient magnitude is exactly zero.

    Raises:
        ValueError: If psi is not two-dimensional or h is not positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ddx(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * h)


def _ddy(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * h)


def _oracle_interface_normal(psi: "np.ndarray", h: float) -> "np.ndarray":
    psi = np.asarray(psi, dtype=np.float64)
    if psi.ndim != 2:
        raise ValueError("psi must be a two-dimensional field")
    if h <= 0.0:
        raise ValueError("h must be positive")
    gx = _ddx(psi, h); gy = _ddy(psi, h)
    g = np.sqrt(gx * gx + gy * gy)
    g = np.where(g > 0.0, g, 1.0)
    return np.stack([gx / g, gy / g])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 48\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nd = 0.2 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\npsi = eps * np.log((np.maximum(phi - delta, 0.0) + 1e-100) / (np.maximum(1 - delta - phi, 0.0) + 1e-100))\n',
         'call': 'interface_normal(psi, h)',
         'gold_call': '_oracle_interface_normal(psi, h)', 'tol': 1e-10},
        {'setup': 'import numpy as np\nN = 32\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\npsi = 0.3 * np.sin(2 * np.pi * X) + 0.1 * np.cos(4 * np.pi * Y)\n',
         'call': 'interface_normal(psi, h)',
         'gold_call': '_oracle_interface_normal(psi, h)', 'tol': 1e-10},
        {'setup': 'import numpy as np\nh = 0.05\npsi = np.zeros((6, 6))\npsi[2, 3] = 1.0\n',
         'call': 'interface_normal(psi, h)',
         'gold_call': '_oracle_interface_normal(psi, h)', 'tol': 1e-12},
        {'setup': 'import numpy as np\n# invalid input: a one-dimensional field is not a two-dimensional field and must raise ValueError\nh = 0.05\npsi = np.linspace(-1.0, 1.0, 20)\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: interface_normal(psi, h))',
         'gold_call': '_catches_value_error(lambda: _oracle_interface_normal(psi, h))'},
    ]
