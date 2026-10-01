"""
Return the source's volumetric interface-regularisation flux for the bounded phase field on a two-dimensional periodic grid: a vector field with a diffusion term proportional to the gradient of the volume fraction and a sharpening term proportional to one minus the squared hyperbolic tangent of the distance-like function over twice the thickness, directed along the interface normal. Both terms carry a prefactor built from the bound, and the two prefactors are different powers of the same quantity; the sharpening term also carries a fixed denominator. Recover the prefactors from the source's bounded transport equation. Take gradients with second-order central differences along each axis on the periodic grid, and take the normal from the distance-like function.

The flux is built so that the bounded equilibrium profile is an exact fixed point: on that profile the diffusion and sharpening terms cancel identically, so the regularisation acts only where numerical transport has moved the interface off its equilibrium shape.

Returns
-------
A (2, N, N) float64 array: entry 0 the x component and entry 1 the y component of the flux.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_regularization_flux(phi: "np.ndarray", eps: float, delta: float, Gamma: float, h: float) -> "np.ndarray":
    """Return the source's volumetric interface-regularisation flux of the bounded phase
    field on the (N, N) periodic grid, as a vector field with a diffusion term and a
    sharpening term directed along the normal taken from the distance-like function.

    Args:
        phi: Volume fraction of phase one on the (N, N) periodic grid; index [i, j] refers to
            cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5).
        Gamma: Regularisation velocity scale, non-negative.
        h: Grid spacing, positive, the same along both axes.

    Returns:
        A (2, N, N) float64 array: entry 0 the x component and entry 1 the y component of the
        flux.

    Raises:
        ValueError: If phi is not two-dimensional, if Gamma is negative, if eps is not
            positive, if delta does not lie in [0, 0.5), or if h is not positive.
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


def _oracle_interface_regularization_flux(phi: "np.ndarray", eps: float, delta: float, Gamma: float, h: float) -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional field")
    if Gamma < 0.0:
        raise ValueError("the regularisation velocity cannot be negative")
    psi = _oracle_interface_distance(phi, eps, delta)
    n = _oracle_interface_normal(psi, h)
    sharp = ((1.0 - 2.0 * delta) ** 2 / 4.0) * (1.0 - np.tanh(psi / (2.0 * eps)) ** 2)
    ax = Gamma * (eps * (1.0 - 2.0 * delta) * _ddx(phi, h) - sharp * n[0])
    ay = Gamma * (eps * (1.0 - 2.0 * delta) * _ddy(phi, h) - sharp * n[1])
    return np.stack([ax, ay])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 64\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nGamma = 1.0\nd = 0.2 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\n',
         'call': 'interface_regularization_flux(phi, eps, delta, Gamma, h)',
         'gold_call': '_oracle_interface_regularization_flux(phi, eps, delta, Gamma, h)', 'tol': 1e-10},
        {'setup': 'import numpy as np\nN = 48\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 5e-2\nGamma = 1.6\nd = 0.15 - np.sqrt((X - 0.4) ** 2 + (Y - 0.6) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (1.2 * eps)))\n',
         'call': 'interface_regularization_flux(phi, eps, delta, Gamma, h)',
         'gold_call': '_oracle_interface_regularization_flux(phi, eps, delta, Gamma, h)', 'tol': 1e-10},
        {'setup': 'import numpy as np\nh = 0.02\neps = 0.04\ndelta = 1e-2\nGamma = 0.0\nphi = np.full((8, 8), 0.5)\nphi[3:5, 3:5] = 0.9\n',
         'call': 'interface_regularization_flux(phi, eps, delta, Gamma, h)',
         'gold_call': '_oracle_interface_regularization_flux(phi, eps, delta, Gamma, h)', 'tol': 1e-12},
        {'setup': 'import numpy as np\n# invalid input: a negative regularisation velocity must raise ValueError\nh = 0.02\neps = 0.04\ndelta = 1e-2\nGamma = -1.0\nphi = np.full((8, 8), 0.5)\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: interface_regularization_flux(phi, eps, delta, Gamma, h))',
         'gold_call': '_catches_value_error(lambda: _oracle_interface_regularization_flux(phi, eps, delta, Gamma, h))'},
    ]
