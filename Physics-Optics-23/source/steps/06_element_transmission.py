"""
Sample the complex transmission of the rotated super-Gaussian phase element on the pixel grid.

The element is a thin transmission mask placed in the plane of the modes. In the rotated coordinates ``x' = x cos(angle) + y sin(angle)`` and ``y' = -x sin(angle) + y cos(angle)`` its transmission is ``exp(-(x'/w_a)**4 - (y'/w_b)**4) * exp(1j * phase_coupling * x' * y')`$with half-widths$$(w_a, w_b) = widths$`. Coordinates are the unit-spaced pixel offsets from the grid center, `$x = j - (nx - 1) / 2$` for column `$j$` and `$y = i - (ny - 1) / 2$` for row `$i$`, in the `$xy$` convention, exactly as for the modes. The result is a complex128 array of shape ``(ny, nx)``.

Returns
-------
np.ndarray, transmission: Complex128 array of shape ``(ny, nx)``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def element_transmission(
    shape: tuple,
    widths: tuple,
    angle: float,
    phase_coupling: float,
) -> "np.ndarray":
    """Return the sampled complex transmission of the element.

    Parameters
    ----------
    shape : tuple
        ``(ny, nx)``, both integers at least 2.
    widths : tuple
        Positive finite half-widths ``(w_a, w_b)`` along the rotated axes.
    angle : float
        Rotation angle of the element axes in radians, finite.
    phase_coupling : float
        Finite coefficient of the bilinear phase in the rotated coordinates.

    Returns
    -------
    transmission : np.ndarray
        Complex128 array of shape ``(ny, nx)``.

    Raises
    ------
    ValueError
        If a shape is wrong, a width is not positive, or a coefficient
        is not finite.
    """
    return transmission

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_element_transmission(
    shape: tuple,
    widths: tuple,
    angle: float,
    phase_coupling: float,
) -> "np.ndarray":
    """Reference implementation."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("shape must contain two integers of at least 2.")
    half = np.asarray(widths, dtype=np.float64)
    if half.shape != (2,) or not np.all(np.isfinite(half)) or np.any(half <= 0):
        raise ValueError("widths must be two positive finite half-widths.")
    theta = float(angle)
    kappa = float(phase_coupling)
    if not (np.isfinite(theta) and np.isfinite(kappa)):
        raise ValueError("angle and phase_coupling must be finite.")
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    x = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    rot_x = grid_x * np.cos(theta) + grid_y * np.sin(theta)
    rot_y = -grid_x * np.sin(theta) + grid_y * np.cos(theta)
    amplitude = np.exp(-((rot_x / half[0]) ** 4) - ((rot_y / half[1]) ** 4))
    return (amplitude * np.exp(1j * kappa * rot_x * rot_y)).astype(np.complex128)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Original cases with independent candidate/reference input graphs."""
    return [{'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'shape = (6, 8)\n'
               'widths = (2.5, 1.5)',
      'call': '_review_independent(element_transmission, shape, widths, 0.3, 0.04)',
      'gold_call': '_review_independent(_oracle_element_transmission, shape, widths, 0.3, 0.04)',
      'tol': 1e-10},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'shape = (5, 7)\n'
               'widths = (3.0, 3.0)',
      'call': '_review_independent(element_transmission, shape, widths, 0.0, -0.1)',
      'gold_call': '_review_independent(_oracle_element_transmission, shape, widths, 0.0, -0.1)',
      'tol': 1e-10},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               '\n'
               'def invalid(function, args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'shape = (6, 6)\n'
               'widths = (2.0, 0.0)',
      'call': '_review_independent(invalid, element_transmission, (shape, widths, 0.2, 0.01))',
      'gold_call': '_review_independent(invalid, _oracle_element_transmission, (shape, widths, 0.2, 0.01))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np',
      'call': '_review_independent(element_transmission, (6, 8), (2.5, 1.5), 0.0, 0.0)',
      'gold_call': '_review_independent(_oracle_element_transmission, (6, 8), (2.5, 1.5), 0.0, 0.0)',
      'tol': 1e-10}]
