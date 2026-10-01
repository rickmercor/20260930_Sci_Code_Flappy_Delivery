"""
Return the right-hand side of the source's bounded volume-fraction transport equation on a two-dimensional periodic grid, written for a time-stepper as the time derivative of the volume fraction: the negative divergence of the volume fraction times the interface velocity, plus the extra non-conservative term the source's equation carries, plus the divergence of the regularisation flux. Take every divergence with second-order central differences along each axis on the periodic grid.

The volume-fraction equation of a compressible two-phase model is not in conservative form: the source's equation carries a term proportional to the volume fraction times the divergence of the interface velocity, which is what keeps a uniform volume fraction uniform under a velocity field that is not divergence free.

Returns
-------
An (N, N) float64 array holding the right-hand side.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def volume_fraction_rhs(phi: "np.ndarray", uI: "np.ndarray", eps: float, delta: float, Gamma: float, h: float) -> "np.ndarray":
    """Return the time derivative of the volume fraction of phase one given by the source's
    bounded transport equation on the (N, N) periodic grid, including its non-conservative
    term and the divergence of the regularisation flux.

    Args:
        phi: Volume fraction of phase one on the (N, N) periodic grid; index [i, j] refers to
            cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        uI: Interface velocity as a (2, N, N) array, entry 0 the x component and entry 1 the y
            component.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5).
        Gamma: Regularisation velocity scale, non-negative.
        h: Grid spacing, positive, the same along both axes.

    Returns:
        An (N, N) float64 array holding the right-hand side.

    Raises:
        ValueError: If phi is not two-dimensional or uI is not a (2, N, N) array matching it,
            or if eps, delta, Gamma or h lies outside the domain accepted by the earlier steps.
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


def _oracle_volume_fraction_rhs(phi: "np.ndarray", uI: "np.ndarray", eps: float, delta: float, Gamma: float, h: float) -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64); uI = np.asarray(uI, dtype=np.float64)
    if phi.ndim != 2 or uI.shape != (2,) + phi.shape:
        raise ValueError("phi must be a two-dimensional field and uI its (2, N, N) velocity")
    a = _oracle_interface_regularization_flux(phi, eps, delta, Gamma, h)
    div_u = _ddx(uI[0], h) + _ddy(uI[1], h)
    return -(_ddx(phi * uI[0], h) + _ddy(phi * uI[1], h)) + phi * div_u + _ddx(a[0], h) + _ddy(a[1], h)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 64\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nGamma = 1.0\nd = 0.2 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\nuI = np.stack([np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)])\n',
         'call': 'volume_fraction_rhs(phi, uI, eps, delta, Gamma, h)',
         'gold_call': '_oracle_volume_fraction_rhs(phi, uI, eps, delta, Gamma, h)', 'tol': 1e-9},
        {'setup': 'import numpy as np\nN = 48\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nGamma = 0.8\nd = 0.15 - np.sqrt((X - 0.4) ** 2 + (Y - 0.6) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\nuI = np.stack([0.3 + 0.2 * np.sin(2 * np.pi * Y), 0.5 * np.cos(2 * np.pi * X)])\n',
         'call': 'volume_fraction_rhs(phi, uI, eps, delta, Gamma, h)',
         'gold_call': '_oracle_volume_fraction_rhs(phi, uI, eps, delta, Gamma, h)', 'tol': 1e-9},
        {'setup': 'import numpy as np\nN = 32\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nGamma = 1.0\nphi = np.full((N, N), 0.3)\nuI = np.stack([np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)])\n',
         'call': 'volume_fraction_rhs(phi, uI, eps, delta, Gamma, h)',
         'gold_call': '_oracle_volume_fraction_rhs(phi, uI, eps, delta, Gamma, h)', 'tol': 1e-10},
        {'setup': 'import numpy as np\n# invalid input: an interface velocity with a single component is not a (2, N, N) array and must raise ValueError\nN = 16\nh = 1.0 / N\neps = 2.0 * h\ndelta = 1e-2\nGamma = 1.0\nphi = np.full((N, N), 0.3)\nuI = np.ones((N, N))\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: volume_fraction_rhs(phi, uI, eps, delta, Gamma, h))',
         'gold_call': '_catches_value_error(lambda: _oracle_volume_fraction_rhs(phi, uI, eps, delta, Gamma, h))'},
    ]
