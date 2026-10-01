"""
Return the right-hand side of one phase's mass equation on a two-dimensional periodic grid, written for a time-stepper as the time derivative of the phasic mass per unit volume: the negative divergence of the phasic mass flux, plus the divergence of that phase's mass-regularisation flux. The source forms the mass-regularisation flux from the volumetric regularisation flux and the phase's own explicitly recovered density, and the two phases carry it with opposite signs. Take every divergence with second-order central differences along each axis on the periodic grid.

Adding the regularisation to the phasic mass equations in divergence form is what makes the mixture mass discretely conservative on a periodic grid irrespective of the scheme, and weighting it by the local phasic density rather than a reference density is what the source changes relative to its predecessor.

Returns
-------
An (N, N) float64 array holding the right-hand side.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phasic_mass_rhs(phi: "np.ndarray", rho: "np.ndarray", u: "np.ndarray", eps: float, delta: float, Gamma: float, h: float, phase_one: bool) -> "np.ndarray":
    """Return the time derivative of one phase's mass per unit volume on the (N, N) periodic
    grid: the convective term of that phase plus the divergence of its mass-regularisation
    flux, signed as the source assigns it to the phase.

    Args:
        phi: Volume fraction of phase one on the (N, N) periodic grid; index [i, j] refers to
            cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        rho: Density of the phase being updated, shaped like phi, positive.
        u: Velocity of the phase being updated as a (2, N, N) array, entry 0 the x component
            and entry 1 the y component.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5).
        Gamma: Regularisation velocity scale, non-negative.
        h: Grid spacing, positive, the same along both axes.
        phase_one: True when the phase being updated is phase one, whose volume fraction is
            phi; False for phase two, whose volume fraction is 1 - phi.

    Returns:
        An (N, N) float64 array holding the right-hand side.

    Raises:
        ValueError: If phi is not two-dimensional or u is not a (2, N, N) array matching it,
            if any density is not positive, or if eps, delta, Gamma or h lies outside the
            domain accepted by the earlier steps.
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


def _oracle_phasic_mass_rhs(phi: "np.ndarray", rho: "np.ndarray", u: "np.ndarray", eps: float, delta: float, Gamma: float, h: float, phase_one: bool) -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64); rho = np.asarray(rho, dtype=np.float64); u = np.asarray(u, dtype=np.float64)
    if phi.ndim != 2 or u.shape != (2,) + phi.shape:
        raise ValueError("phi must be a two-dimensional field and u its (2, N, N) velocity")
    if np.any(rho <= 0.0):
        raise ValueError("density must be positive")
    a = _oracle_interface_regularization_flux(phi, eps, delta, Gamma, h)
    vf = phi if phase_one else (1.0 - phi)
    sgn = 1.0 if phase_one else -1.0
    conv = _ddx(vf * rho * u[0], h) + _ddy(vf * rho * u[1], h)
    return -conv + sgn * (_ddx(rho * a[0], h) + _ddy(rho * a[1], h))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 64\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nGamma = 1.0\nd = 0.2 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\nrho = np.full((N, N), 1.0e3)\nu = 0.4 * np.stack([np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)])\nphase_one = True\n',
         'call': 'phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one)',
         'gold_call': '_oracle_phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one)', 'tol': 1e-8},
        {'setup': 'import numpy as np\nN = 64\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nGamma = 1.0\nd = 0.2 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\nrho = 1.2 * (1.0 + 0.1 * np.sin(2 * np.pi * Y))\nu = np.stack([np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)])\nphase_one = False\n',
         'call': 'phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one)',
         'gold_call': '_oracle_phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one)', 'tol': 1e-10},
        {'setup': 'import numpy as np\nN = 32\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nGamma = 0.0\nphi = 0.5 + 0.3 * np.sin(2 * np.pi * X)\nrho = np.full((N, N), 2.0)\nu = np.stack([np.ones((N, N)), np.zeros((N, N))])\nphase_one = True\n',
         'call': 'phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one)',
         'gold_call': '_oracle_phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one)', 'tol': 1e-10},
        {'setup': 'import numpy as np\n# invalid input: a zero density is not positive and must raise ValueError\nN = 16\nh = 1.0 / N\neps = 2.0 * h\ndelta = 1e-2\nGamma = 1.0\nphi = np.full((N, N), 0.3)\nrho = np.zeros((N, N))\nu = np.zeros((2, N, N))\nphase_one = True\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one))',
         'gold_call': '_catches_value_error(lambda: _oracle_phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one))'},
    ]
