"""
Construct the starting trajectory on a uniform grid of the evolution parameter. SPECIFICATION: well is the array returned by the previous step. The trajectory is built at the energy E equal to the potential at the second well, that is well[1, 4]. Let f(x) be one eighth of the square of x squared minus one plus a times x cubed, let f'(x) be its derivative, and let g(x) be the square root of one plus four c squared x squared. Let xr be well[1, 0] and xl be well[0, 0]. Define xR as the root of f(x) minus E minus 1e-10 lying between zero and xr, found by a bracketing root finder to machine precision, that is with absolute tolerance 1e-16 and relative tolerance 8.9e-16 on x. The parameter accumulates as dt = g(x) dx divided by the square root of twice f(x) minus E, the mass being one. If the absolute value of f'(xl) is below 1e-13 and the absolute value of f(xl) minus E is below 1e-18, define xL as the root of f(x) minus E minus 1e-10 lying between xl and zero, found to the same precision, integrate on 200001 equally spaced points from xL to xR by the trapezoidal rule, centre the resulting interval inside tau_inst, and complete both ends with x equal to the nearby minimum minus the offset of the interval end from it times the exponential of minus w(x) times the elapsed parameter, where w(x) is the square root of the absolute value of (3 x^2 - 1)/2 + 6 a x, divided by g(x). Otherwise define xb as the root of f(x) minus E lying between xl and zero, found to the same precision, substitute x = xb + u squared and integrate on 200001 equally spaced points in u from zero to the square root of xR minus xb by the trapezoidal rule, taking the integrand at u equal to zero to be twice g(xb) divided by the square root of twice the absolute value of f'(xb); the trajectory then starts at xb and the parameter values beyond the end of that integral are completed with the same exponential approach to xr. Values inside the integrated interval are obtained by linear interpolation of x against the accumulated parameter. The second coordinate is set to minus c times x squared minus one at every point. The result is a real array of shape (n_beads + 1, 2) sampled at parameter values i times tau_inst over n_beads for i from zero to n_beads.

The quantity being computed is dominated by an exponential, so the starting guess has to be close enough that the later refinement does not wander onto a different stationary configuration. The configuration wanted here is the one whose two halves carry equal energy, and in the low-temperature regime that shared energy is pinned at the depth of the shallower well: the motion lingers there for most of the available parameter range and turns around on the far side at the point where the potential climbs back to the same value. The elapsed parameter grows only logarithmically as the end point is approached, which is why the approach is appended in closed form instead of being integrated, and the square-root zero at the far turning point is why the integration variable is changed there.

Returns
-------
numpy.ndarray of shape (n_beads + 1, 2) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def initial_path(a: float, c: float, wy: float, well: "np.ndarray", n_beads: int, tau_inst: float) -> "np.ndarray":
    """Construct the starting trajectory on a uniform grid of the evolution parameter. SPECIFICATION: well is the array returned by the previous step. The trajectory is built at the energy E equal to the potential at the second well, that is well[1, 4]. Let f(x) be one eighth of the square of x squared minus one plus a times x cubed, let f'(x) be its derivative, and let g(x) be the square root of one plus four c squared x squared. Let xr be well[1, 0] and xl be well[0, 0]. Define xR as the root of f(x) minus E minus 1e-10 lying between zero and xr, found by a bracketing root finder to machine precision, that is with absolute tolerance 1e-16 and relative tolerance 8.9e-16 on x. The parameter accumulates as dt = g(x) dx divided by the square root of twice f(x) minus E, the mass being one. If the absolute value of f'(xl) is below 1e-13 and the absolute value of f(xl) minus E is below 1e-18, define xL as the root of f(x) minus E minus 1e-10 lying between xl and zero, found to the same precision, integrate on 200001 equally spaced points from xL to xR by the trapezoidal rule, centre the resulting interval inside tau_inst, and complete both ends with x equal to the nearby minimum minus the offset of the interval end from it times the exponential of minus w(x) times the elapsed parameter, where w(x) is the square root of the absolute value of one half times three x squared minus one plus six a x, divided by g(x). Otherwise define xb as the root of f(x) minus E lying between xl and zero, found to the same precision, substitute x = xb + u squared and integrate on 200001 equally spaced points in u from zero to the square root of xR minus xb by the trapezoidal rule, taking the integrand at u equal to zero to be twice g(xb) divided by the square root of twice the absolute value of f'(xb); the trajectory then starts at xb and the parameter values beyond the end of that integral are completed with the same exponential approach to xr. Values inside the integrated interval are obtained by linear interpolation of x against the accumulated parameter. The second coordinate is set to minus c times x squared minus one at every point. The result is a real array of shape (n_beads + 1, 2) sampled at parameter values i times tau_inst over n_beads for i from zero to n_beads.

    Returns
    -------
    numpy.ndarray of shape (n_beads + 1, 2) and real dtype.

    Raises
    ------
    ValueError: if well does not have shape (2, 6), if n_beads is less than two, or if tau_inst is not positive and finite.
    """
    return path  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _oracle_initial_path(a: float, c: float, wy: float, well: "np.ndarray", n_beads: int, tau_inst: float) -> "np.ndarray":
    n_quad = 200001
    eta = 1e-10
    mass = 1.0
    well = np.asarray(well, dtype=float)
    if well.shape != (2, 6):
        raise ValueError("well must have shape (2, 6)")
    n_beads = int(n_beads)
    if n_beads < 2:
        raise ValueError("n_beads must be at least 2")
    if not np.isfinite(tau_inst) or tau_inst <= 0.0:
        raise ValueError("tau_inst must be positive and finite")
    x_l, x_r, E = well[0, 0], well[1, 0], well[1, 4]
    Veff = lambda z: 0.125 * (z * z - 1.0) ** 2 + a * z ** 3
    dVeff = lambda z: 0.5 * z * (z * z - 1.0) + 3.0 * a * z * z
    arc = lambda z: np.sqrt(1.0 + 4.0 * c * c * z * z)
    om_eff = lambda z: np.sqrt(abs(0.5 * (3.0 * z * z - 1.0) + 6.0 * a * z)) / arc(z)
    x_endR = brentq(lambda z: Veff(z) - E - eta, 0.0, x_r, xtol=1e-16, rtol=8.9e-16)
    tg = np.arange(n_beads + 1) * (tau_inst / n_beads)
    if abs(dVeff(x_l)) < 1e-13 and abs(Veff(x_l) - E) < 1e-18:
        x_endL = brentq(lambda z: Veff(z) - E - eta, x_l, 0.0, xtol=1e-16, rtol=8.9e-16)
        xs = np.linspace(x_endL, x_endR, n_quad)
        integ = arc(xs) / np.sqrt(2.0 * np.maximum(Veff(xs) - E, 1e-300) / mass)
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(xs))])
        lead = 0.5 * (tau_inst - tau[-1])
        xx = np.empty(n_beads + 1)
        m1 = tg < lead
        xx[m1] = x_l + (x_endL - x_l) * np.exp(-om_eff(x_l) * (lead - tg[m1]))
        m2 = (tg >= lead) & (tg <= lead + tau[-1])
        xx[m2] = np.interp(tg[m2] - lead, tau, xs)
        m3 = tg > lead + tau[-1]
        xx[m3] = x_r - (x_r - x_endR) * np.exp(-om_eff(x_r) * (tg[m3] - lead - tau[-1]))
    else:
        x_b = brentq(lambda z: Veff(z) - E, x_l, 0.0, xtol=1e-16, rtol=8.9e-16)
        u = np.linspace(0.0, np.sqrt(x_endR - x_b), n_quad)
        xs = x_b + u * u
        vv = np.maximum(Veff(xs) - E, 0.0)
        integ = np.zeros_like(u)
        nz = vv > 0
        integ[nz] = 2.0 * u[nz] * arc(xs[nz]) / np.sqrt(2.0 * vv[nz] / mass)
        integ[0] = 2.0 * arc(x_b) / np.sqrt(2.0 * abs(dVeff(x_b)) / mass)
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(u))])
        xx = np.empty(n_beads + 1)
        near = tg <= tau[-1]
        xx[near] = np.interp(tg[near], tau, xs)
        xx[~near] = x_r - (x_r - x_endR) * np.exp(-om_eff(x_r) * (tg[~near] - tau[-1]))
    return np.stack([xx, -c * (xx * xx - 1.0)], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nW = np.array([[-1.0000000015, -4.800000014399997e-09, 0.2643057316378634, 2.080923470725479, -5.00000001125e-10, 0.04690458354726685], [0.9999999985, 4.799999985599993e-09, 0.2643057319044678, 2.0809234655050735, 4.999999988750001e-10, 0.04690458444819083]])',
         "call": 'initial_path(5e-10, 1.6, 0.55, W.copy(), 64, 120.0)',
         "gold_call": '_oracle_initial_path(5e-10, 1.6, 0.55, W.copy(), 64, 120.0)'},   # normal: task couplings, 64 intervals over 120
        {"setup": 'import numpy as np\nW = np.array([[-1.0000000015, -4.800000014399997e-09, 0.2643057316378634, 2.080923470725479, -5.00000001125e-10, 0.04690458354726685], [0.9999999985, 4.799999985599993e-09, 0.2643057319044678, 2.0809234655050735, 4.999999988750001e-10, 0.04690458444819083]])',
         "call": 'initial_path(5e-10, 1.6, 0.55, W.copy(), 24, 60.0)',
         "gold_call": '_oracle_initial_path(5e-10, 1.6, 0.55, W.copy(), 24, 60.0)'},   # edge: coarser grid, shorter range
        {"setup": 'import numpy as np\nW0 = np.array([[-1.0, 0.0, 0.3979336830493747, 2.0103852327090848, 0.0, 0.04816637831516919], [1.0, 0.0, 0.3979336830493747, 2.0103852327090848, 0.0, 0.04816637831516919]])',
         "call": 'initial_path(0.0, 1.0, 0.8, W0.copy(), 40, 120.0)',
         "gold_call": '_oracle_initial_path(0.0, 1.0, 0.8, W0.copy(), 40, 120.0)'},   # boundary: symmetric limit takes the degenerate branch
        {"setup": 'import numpy as np\nW = np.array([[-1.0000000015, -4.800000014399997e-09, 0.2643057316378634, 2.080923470725479, -5.00000001125e-10, 0.04690458354726685], [0.9999999985, 4.799999985599993e-09, 0.2643057319044678, 2.0809234655050735, 4.999999988750001e-10, 0.04690458444819083]])\ndef _c():\n    try:\n        initial_path(5e-10, 1.6, 0.55, W.copy(), 1, 120.0)\n        return 0\n    except ValueError:\n        return 1\ndef _g():\n    try:\n        _oracle_initial_path(5e-10, 1.6, 0.55, W.copy(), 1, 120.0)\n        return 0\n    except ValueError:\n        return 1',
         "call": '_c()',
         "gold_call": '_g()'},   # invalid: fewer than two intervals
    ]
