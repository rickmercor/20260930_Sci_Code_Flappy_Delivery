"""
Take one time step of the coupled volume-fraction and phasic-mass system on a two-dimensional periodic grid with a two-stage strong-stability-preserving Runge-Kutta method, given the current volume fraction, the two phasic masses per unit volume, the two prescribed phasic velocity fields and the two uniform phasic pressures. Within each stage recover each phasic density by dividing its mass by its volume fraction floored at the bound, form the interface velocity from the source's closure, and evaluate the three right-hand sides; do not clip the volume fraction at any point.

The source advances the volume fraction and the phasic masses together so that the mixture mass stays discretely conserved; recovering the phasic densities explicitly at every stage is what the finite bound makes well posed.

Returns
-------
A (5, N, N) float64 array: the updated volume fraction, the updated mass per unit volume of phase one, the updated mass per unit volume of phase two, and the x and y components of the interface velocity evaluated at the start of the step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance(phi: "np.ndarray", m1: "np.ndarray", m2: "np.ndarray", u1: "np.ndarray", u2: "np.ndarray", p1: "np.ndarray", p2: "np.ndarray", eps: float, delta: float, Gamma: float, h: float, dt: float) -> "np.ndarray":
    """Take one two-stage strong-stability-preserving Runge-Kutta step of the coupled
    volume-fraction and phasic-mass system on the (N, N) periodic grid, recovering each
    phasic density within a stage as its mass over its volume fraction floored at the bound,
    without clipping the volume fraction.

    Args:
        phi: Volume fraction of phase one on the (N, N) periodic grid; index [i, j] refers to
            cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        m1: Mass per unit volume of phase one, shaped like phi, positive.
        m2: Mass per unit volume of phase two, shaped like phi, positive.
        u1: Velocity of phase one as a (2, N, N) array, entry 0 the x component and entry 1
            the y component.
        u2: Velocity of phase two as a (2, N, N) array laid out like u1.
        p1: Pressure of phase one, shaped like phi.
        p2: Pressure of phase two, shaped like phi.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5).
        Gamma: Regularisation velocity scale, non-negative.
        h: Grid spacing, positive, the same along both axes.
        dt: Time step, positive.

    Returns:
        A (5, N, N) float64 array: the updated volume fraction, the updated mass per unit
        volume of phase one, the updated mass per unit volume of phase two, and the x and y
        components of the interface velocity evaluated at the start of the step.

    Raises:
        ValueError: If dt is not positive, if phi is not two-dimensional, if u1 or u2 is not
            a (2, N, N) array matching it, if m1 or m2 is not a positive field shaped like
            phi, or if eps, delta, Gamma or h lies outside the domain accepted by the earlier
            steps.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_advance(phi: "np.ndarray", m1: "np.ndarray", m2: "np.ndarray", u1: "np.ndarray", u2: "np.ndarray", p1: "np.ndarray", p2: "np.ndarray", eps: float, delta: float, Gamma: float, h: float, dt: float) -> "np.ndarray":
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    phi = np.asarray(phi, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64); m2 = np.asarray(m2, dtype=np.float64)
    u1 = np.asarray(u1, dtype=np.float64); u2 = np.asarray(u2, dtype=np.float64)
    if phi.ndim != 2 or u1.shape != (2,) + phi.shape or u2.shape != (2,) + phi.shape:
        raise ValueError("phi must be a two-dimensional field and u1, u2 its (2, N, N) velocities")
    if m1.shape != phi.shape or m2.shape != phi.shape or np.any(m1 <= 0.0) or np.any(m2 <= 0.0):
        raise ValueError("the phasic masses must be positive fields shaped like phi")

    def _rhs(ph, a1, a2):
        r1 = a1 / np.maximum(ph, delta)
        r2 = a2 / np.maximum(1.0 - ph, delta)
        uIx = _oracle_interface_closures(ph, p1, p2, r1, r2, u1[0], u2[0])[1]
        uIy = _oracle_interface_closures(ph, p1, p2, r1, r2, u1[1], u2[1])[1]
        uI = np.stack([uIx, uIy])
        return (_oracle_volume_fraction_rhs(ph, uI, eps, delta, Gamma, h),
                _oracle_phasic_mass_rhs(ph, r1, u1, eps, delta, Gamma, h, True),
                _oracle_phasic_mass_rhs(ph, r2, u2, eps, delta, Gamma, h, False), uI)

    k = _rhs(phi, m1, m2)
    p_1 = phi + dt * k[0]
    a_1, b_1 = m1 + dt * k[1], m2 + dt * k[2]
    k2 = _rhs(p_1, a_1, b_1)
    p_n = 0.5 * (phi + p_1 + dt * k2[0])
    a_n = 0.5 * (m1 + a_1 + dt * k2[1])
    b_n = 0.5 * (m2 + b_1 + dt * k2[2])
    return np.stack([p_n, a_n, b_n, k[3][0], k[3][1]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 48\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nd = 0.2 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\nm1 = phi * 1.0e3\nm2 = (1 - phi) * 1.2\nu2 = np.stack([np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)])\nu1 = 0.4 * u2\np1 = np.full((N, N), 1.0e5)\np2 = np.full((N, N), 1.0e5)\nGamma = 1.0\ndt = 0.25 * h / Gamma\n',
         'call': 'advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt)',
         'gold_call': '_oracle_advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt)', 'tol': 1e-9},
        {'setup': 'import numpy as np\nN = 40\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 2e-2\nd = 0.15 - np.sqrt((X - 0.4) ** 2 + (Y - 0.6) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\nm1 = phi * 8.0e2 * (1.0 + 0.05 * np.cos(2 * np.pi * X))\nm2 = (1 - phi) * 1.0\nu2 = np.stack([0.5 * np.ones((N, N)), 0.2 * np.sin(2 * np.pi * X)])\nu1 = np.stack([0.5 * np.ones((N, N)), np.zeros((N, N))])\np1 = np.full((N, N), 2.0e5)\np2 = np.full((N, N), 1.0e5)\nGamma = 0.5\ndt = 0.2 * h / Gamma\n',
         'call': 'advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt)',
         'gold_call': '_oracle_advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt)', 'tol': 1e-9},
        {'setup': 'import numpy as np\nN = 24\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\neps = 2.0 * h\ndelta = 1e-2\nphi = np.full((N, N), 0.4)\nm1 = phi * 1.0e3\nm2 = (1 - phi) * 1.2\nu2 = np.stack([np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)])\nu1 = 0.7 * u2\np1 = np.full((N, N), 1.0e5)\np2 = np.full((N, N), 1.0e5)\nGamma = 1.0\ndt = 0.25 * h / Gamma\n',
         'call': 'advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt)',
         'gold_call': '_oracle_advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt)', 'tol': 1e-10},
        {'setup': 'import numpy as np\n# invalid input: a zero time step is not positive and must raise ValueError\nN = 16\nh = 1.0 / N\neps = 2.0 * h\ndelta = 1e-2\nphi = np.full((N, N), 0.4)\nm1 = phi * 1.0e3\nm2 = (1 - phi) * 1.2\nu1 = np.zeros((2, N, N))\nu2 = np.zeros((2, N, N))\np1 = np.full((N, N), 1.0e5)\np2 = np.full((N, N), 1.0e5)\nGamma = 1.0\ndt = 0.0\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt))',
         'gold_call': '_catches_value_error(lambda: _oracle_advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt))'},
    ]
