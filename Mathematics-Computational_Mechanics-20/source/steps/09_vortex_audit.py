"""
Run the source's bounded interface regularisation through a prescribed two-dimensional vortex on three fixed configurations and audit the interface it produces, assembling the run from the earlier steps: initialise a circular blob of phase one with the bounded equilibrium profile, advance the coupled volume-fraction and phasic-mass system with the two-stage Runge-Kutta step, and record one row of diagnostics per configuration. The interface-length measure in the last column, the space integral of the volume fraction times its complement, is the quantity the audit is built around.

The two-phase Taylor-Green vortex is one of the source's own test flows; it stretches and folds an initially circular interface, and the regularisation must hold the interface at its prescribed thickness while the flow lengthens it. Because the two phases move with different prescribed velocities, the interface follows the source's mass-weighted interface velocity, which is not divergence free, so the non-conservative term of the volume-fraction equation and the density-weighted mass regularisation both act throughout the run.

Returns
-------
A (3, 6) float64 array with one row per configuration in the order listed: the largest magnitude of the regularisation flux on the initial profile in units of 1e-6, the ratio of the space integral of the volume fraction at the end to its initial value, the largest magnitude of the volume-fraction gradient at the end (central differences), the smallest volume fraction at the end, the space integral of the absolute difference between the final and initial volume fraction in units of 1e-3, and the space integral of the volume fraction times its complement at the end in units of 1e-3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vortex_audit(vel_scale: float) -> "np.ndarray":
    """Run the three fixed vortex configurations of the audit and return one row of
    diagnostics per configuration, assembling each run from the earlier steps.

    The domain is the unit periodic square with N x N cells of spacing h = 1 / N, cell
    centres at ((i + 1/2) h, (j + 1/2) h), index [i, j] with x along axis 0 and y along
    axis 1. The finite bound is delta = 1e-2 and the interface thickness is eps = 2 h. Phase
    one has density 1.0e3 and stiffened-gas constants 4.4 and 6.0e8, phase two has density
    1.2 and constants 1.4 and 0; both phasic pressures are 1.0e5 everywhere and the
    equation of state must invert consistently for both phases. The three configurations
    are (N, R0, alpha, cx, cy, steps) = (96, 0.15, 0.4, 0.5, 0.5, 160), (128, 0.12, 0.7,
    0.25, 0.5, 100) and (80, 0.18, 0.25, 0.3, 0.3, 107). The initial volume fraction of
    phase one is the bounded equilibrium profile delta + (1 - 2 delta) (1 + tanh(d / (2
    eps))) / 2 of the signed distance d = R0 - r, where r is the distance from the cell
    centre to (cx, cy) measured on the periodic square (each coordinate difference reduced
    into [-1/2, 1/2)). The velocity of phase two is vel_scale times (sin(2 pi x) cos(2 pi
    y), -cos(2 pi x) sin(2 pi y)) evaluated at the cell centres, and the velocity of phase
    one is alpha times that of phase two. The regularisation velocity Gamma is the largest
    velocity magnitude over both phases and all cells, and the time step is 0.25 h /
    Gamma. The phasic masses start as the volume fractions times the phasic densities, and
    the system is advanced for the configuration's number of steps with the earlier
    Runge-Kutta step, without clipping.

    Args:
        vel_scale: Positive finite factor multiplying both phasic velocity fields.

    Returns:
        A (3, 6) float64 array with one row per configuration in the order listed: the
        largest magnitude of the regularisation flux on the initial profile in units of 1e-6,
        the ratio of the space integral of the volume fraction at the end to its initial
        value, the largest magnitude of the volume-fraction gradient at the end (central
        differences), the smallest volume fraction at the end, the space integral of the
        absolute difference between the final and initial volume fraction in units of 1e-3,
        and the space integral of the volume fraction times its complement at the end in
        units of 1e-3.

    Raises:
        ValueError: If vel_scale is not a positive finite number, if the equation of state
            does not invert consistently for both phases, if the mixture mass drifts by more
            than 1e-10 relative over a run, or if the final volume fraction is not a finite
            field in [0, 1].
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


def _configuration() -> "tuple[dict, list]":
    """Material data and the three (N, R0, alpha, cx, cy, steps) configurations."""
    pars = dict(delta=1.0e-2, ceps=2.0, cfl=0.25, rho1=1.0e3, rho2=1.2, p0=1.0e5,
                gamma1=4.4, pi1=6.0e8, gamma2=1.4, pi2=0.0)
    cfgs = [(96, 0.15, 0.4, 0.5, 0.5, 160), (128, 0.12, 0.7, 0.25, 0.5, 100), (80, 0.18, 0.25, 0.3, 0.3, 107)]
    return pars, cfgs


def _oracle_vortex_audit(vel_scale: float) -> "np.ndarray":
    if isinstance(vel_scale, bool) or not np.isfinite(vel_scale) or vel_scale <= 0.0:
        raise ValueError("vel_scale must be positive and finite")
    P, cfgs = _configuration()
    delta = P["delta"]
    rows = []
    for (N, R0, alpha, cx, cy, nstep) in cfgs:
        h = 1.0 / N
        eps = P["ceps"] * h
        x = (np.arange(N) + 0.5) * h
        X, Y = np.meshgrid(x, x, indexing="ij")
        wx = (X - cx + 0.5) % 1.0 - 0.5
        wy = (Y - cy + 0.5) % 1.0 - 0.5
        d = R0 - np.sqrt(wx * wx + wy * wy)
        phi = delta + (1.0 - 2.0 * delta) * 0.5 * (1.0 + np.tanh(d / (2.0 * eps)))
        phi0 = phi.copy()
        ux = np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
        uy = -np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
        u2 = float(vel_scale) * np.stack([ux, uy])
        u1 = alpha * u2
        Gamma = float(max(np.max(np.hypot(u1[0], u1[1])), np.max(np.hypot(u2[0], u2[1]))))
        dt = P["cfl"] * h / max(Gamma, 1.0e-30)
        p1 = np.full((N, N), P["p0"]); p2 = np.full((N, N), P["p0"])
        r1 = np.full((N, N), P["rho1"]); r2 = np.full((N, N), P["rho2"])
        e1 = (p1 + P["gamma1"] * P["pi1"]) / ((P["gamma1"] - 1.0) * r1)
        e2 = (p2 + P["gamma2"] * P["pi2"]) / ((P["gamma2"] - 1.0) * r2)
        if (np.max(np.abs(_oracle_stiffened_gas_pressure(r1, e1, P["gamma1"], P["pi1"]) - p1)) > 1e-6 * P["p0"]
                or np.max(np.abs(_oracle_stiffened_gas_pressure(r2, e2, P["gamma2"], P["pi2"]) - p2)) > 1e-6 * P["p0"]):
            raise ValueError("the equation of state does not invert consistently")
        a0 = _oracle_interface_regularization_flux(phi, eps, delta, Gamma, h)
        res = float(np.max(np.hypot(a0[0], a0[1])))
        m1 = phi * r1; m2 = (1.0 - phi) * r2
        m0 = float(np.sum(m1 + m2)) * h * h
        for _ in range(nstep):
            out = _oracle_advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt)
            phi, m1, m2 = out[0], out[1], out[2]
        if abs(float(np.sum(m1 + m2)) * h * h - m0) / m0 > 1.0e-10:
            raise ValueError("the regularisation is not in divergence form: mixture mass drifted")
        if not np.all(np.isfinite(phi)) or np.min(phi) < 0.0 or np.max(phi) > 1.0:
            raise ValueError("the volume fraction is not a finite field in [0, 1]")
        gx = _ddx(phi, h); gy = _ddy(phi, h)
        area = float(np.sum(phi)) * h * h
        area0 = float(np.sum(phi0)) * h * h
        L = float(np.sum(phi * (1.0 - phi))) * h * h
        moved = float(np.sum(np.abs(phi - phi0))) * h * h
        rows.append([res * 1.0e6, area / area0, float(np.max(np.hypot(gx, gy))), float(np.min(phi)), moved * 1.0e3, L * 1.0e3])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nvel_scale = 1.0\n',
         'call': 'vortex_audit(vel_scale)',
         'gold_call': '_oracle_vortex_audit(vel_scale)', 'tol': 1e-7},
        {'setup': 'import numpy as np\nvel_scale = 0.5\n',
         'call': 'vortex_audit(vel_scale)',
         'gold_call': '_oracle_vortex_audit(vel_scale)', 'tol': 1e-7},
        {'setup': 'import numpy as np\nvel_scale = 1.5\n',
         'call': 'vortex_audit(vel_scale)',
         'gold_call': '_oracle_vortex_audit(vel_scale)', 'tol': 1e-7},
        {'setup': 'import numpy as np\n# invalid input: a zero velocity scale is not positive and must raise ValueError\nvel_scale = 0.0\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: vortex_audit(vel_scale))',
         'gold_call': '_catches_value_error(lambda: _oracle_vortex_audit(vel_scale))'},
    ]
