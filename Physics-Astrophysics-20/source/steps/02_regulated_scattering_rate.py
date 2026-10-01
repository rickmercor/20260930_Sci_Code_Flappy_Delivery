"""
Firehose-regulated pitch-angle scattering.

Firehose-regulated pitch-angle scattering.

When the parallel pressure exceeds the perpendicular pressure by more than the
firehose threshold, P_par - P_perp > c_th B^2/2 (c_th = 1.4 when P_perp and P_par
are the total pair pressures), kinetic instabilities scatter particles in pitch
angle. The task models this as a relaxation of the pressure anisotropy that
holds the combination 2 P_perp + P_par of a fluid element fixed while
P_par - P_perp decays at rate 3 nu, with an effective rate nu switched by the
threshold: the full rate
nubar = tau_e^(-1/3) Omega_e^(2/3) above the threshold and no scattering below
it. Here tau_e = m_e^2 c^3 / (2 r_e^2 B^2 T) and Omega_e = e B c / (3 T) are the
synchrotron cooling time and the relativistic gyrofrequency evaluated with the
local field B and the local temperature T = (2 P_perp + P_par) / (3 n); the
reference quantities tau0 and Omega0 of step 01 are the same expressions at B0
and T0. The switch is taken in its zero-width limit: on the threshold itself the
rate is the one that keeps a fluid element on the threshold, limited to
[0, nubar].

Schedule: for t < t_d there is no scattering and the radiative closure constants
of step 01 are (alpha_perp, alpha_par) = (16/15, 8/15); for t >= t_d they are
(0.9, 0.48) in every cell and scattering is enabled, with no later reset.

Returns
-------
The effective scattering rate of every cell.

Returns
-------
The effective scattering rate of every cell.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regulated_scattering_rate(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    du_dx: ArrayLike,
    t: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    """Return the effective pitch-angle scattering rate nu of every cell.

    Parameters
    ----------
    n : float or array_like
        Number density in units of n0. Strictly positive.
    B : float or array_like
        Magnetic field strength in units of sqrt(4 pi n0 m_e c^2). Strictly
        positive.
    p_perp : float or array_like
        Perpendicular pressure in units of n0 m_e c^2. Strictly positive.
    p_par : float or array_like
        Parallel pressure in units of n0 m_e c^2. Strictly positive.
    du_dx : float or array_like
        Local gradient of the bulk velocity along x (units Omega0), i.e. the
        rate of perpendicular expansion felt by the fluid element (negative in
        compression). Finite.
    t : float
        Time (units 1/Omega0) at which the schedule is evaluated.
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float
        Onset time of scattering (units 1/Omega0).
    c_th : float, optional
        Firehose threshold constant. Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``S`` (broadcast shape of ``n``, ``B``, ``p_perp``,
        ``p_par`` and ``du_dx``) holding nu in units of Omega0. For t < t_d
        every entry is 0. For t >= t_d, with D = P_par - P_perp and
        D_th = c_th B^2/2: nu = nubar where D > D_th (1 + 1e-10); nu = 0 where
        D < D_th (1 - 1e-10); otherwise (on the threshold) nu is the rate at
        which D - D_th of the fluid element, whose velocity gradient is
        ``du_dx``, stays constant in time under the full local model of the
        task, clipped to [0, nubar].

    Raises
    ------
    ValueError
        If ``n``, ``B``, ``p_perp`` or ``p_par`` has a non-finite or
        non-positive entry, if ``du_dx`` has a non-finite entry, if the five
        arrays do not broadcast together, if ``t`` or ``t_d`` is not a finite
        scalar, or if ``beta0``, ``theta0``, ``chi`` or ``c_th`` is not a
        finite, strictly positive scalar.
    """
    return rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike


def _fr_post_constants():
    """Radiative closure constants (alpha_perp, alpha_par) at and after the onset time."""
    return 0.9, 0.48


def _fr_finite(name, x):
    """Finite real scalar."""
    if np.ndim(x) != 0:
        raise ValueError(f"{name} must be a scalar")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _fr_rate_prefactor(n, B, beta0, theta0, chi):
    """Factor f(n, B) with nubar = f / (2 P_perp + P_par)^(1/3) (no validation)."""
    b0 = math.sqrt(2.0 * theta0 / beta0)
    return chi ** (-1.0 / 3.0) * (B / b0) ** (4.0 / 3.0) * np.cbrt(3.0 * n * theta0)


def _fr_full_rate(n, B, p, q, beta0, theta0, chi):
    """Full scattering rate nubar at the local field and temperature (no validation)."""
    return _fr_rate_prefactor(n, B, beta0, theta0, chi) / np.cbrt(2.0 * p + q)


def _fr_exponents():
    """Fixed adiabatic exponents of (P_perp, P_par) under perpendicular compression."""
    return 8.0 / 5.0, 4.0 / 5.0


def _fr_drive(p, q, B, ux, rp, rq, c_th):
    """Scattering-free rate of change of the threshold excess of a fluid element,
    given the radiative sinks (rp, rq) (no validation)."""
    gp, gq = _fr_exponents()
    return (gp * p - gq * q + c_th * (B * B)) * ux + (rq - rp)


def _fr_switch(d, thr, g, nub):
    """Zero-width switch: nub above the band, 0 below it, clipped balance rate on it."""
    band = 1e-10
    above = d > thr * (1.0 + band)
    below = d < thr * (1.0 - band)
    nu = np.where(above, nub, 0.0)
    on = ~(above | below)
    if np.any(on):
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = np.where(on, np.clip(g / (3.0 * d), 0.0, nub), nu)
    return nu


def _oracle_regulated_scattering_rate(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    du_dx: ArrayLike,
    t: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    n = _rs_positive("n", n)
    B = _rs_positive("B", B)
    p = _rs_positive("p_perp", p_perp)
    q = _rs_positive("p_par", p_par)
    ux = np.asarray(du_dx, dtype=float)
    if not np.all(np.isfinite(ux)):
        raise ValueError("du_dx must be finite")
    tt = _fr_finite("t", t)
    td = _fr_finite("t_d", t_d)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    cth = _rs_scalar("c_th", c_th)
    try:
        n, B, p, q, ux = np.broadcast_arrays(n, B, p, q, ux)
    except ValueError:
        raise ValueError("n, B, p_perp, p_par and du_dx must broadcast together") from None
    if tt < td:
        return np.zeros(n.shape)
    ap, aq = _fr_post_constants()
    sinks = _oracle_radiative_pressure_sinks(n, B, p, q, ap, aq, beta0, theta0, chi)
    g = _fr_drive(p, q, B, ux, sinks[0], sinks[1], cth)
    nub = _fr_full_rate(n, B, p, q, beta0, theta0, chi)
    return _fr_switch(q - p, 0.5 * cth * (B * B), g, nub)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cases with independent candidate and oracle argument graphs."""
    return [{'setup': 'import numpy as np\n'
               'b = np.full(5, 0.2236067977499790)\n'
               'p = np.array([1.0, 1.0, 1.0, 1.2, 0.7])\n'
               'q = p + 0.7 * b * b\n'
               'q[0] = 1.1\n'
               'q[1] = 1.0\n'
               'ux = np.array([3.0e-3, -3.0e-3, 1.0e-4, -1.0e-2, 1.0])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(regulated_scattering_rate, 1.0, b, p, q, ux, 400.0, 40.0, 1.0, '
              '2000.0, 340.0)',
      'gold_call': '_independent_call(_oracle_regulated_scattering_rate, 1.0, b, p, q, ux, 400.0, '
                   '40.0, 1.0, 2000.0, 340.0)'},
     {'setup': 'import numpy as np\n'
               'b = np.full(5, 0.2236067977499790)\n'
               'p = np.array([1.0, 1.0, 1.0, 1.2, 0.7])\n'
               'q = p + 0.7 * b * b\n'
               'q[0] = 1.1\n'
               'q[1] = 1.0\n'
               'ux = np.array([3.0e-3, -3.0e-3, 1.0e-4, -1.0e-2, 1.0])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(regulated_scattering_rate, 1.0, b, p, q, ux, 339.0, 40.0, 1.0, '
              '2000.0, 340.0)',
      'gold_call': '_independent_call(_oracle_regulated_scattering_rate, 1.0, b, p, q, ux, 339.0, '
                   '40.0, 1.0, 2000.0, 340.0)'},
     {'setup': 'import numpy as np\n'
               'bb = 9.0\n'
               'thr = 0.5 * 1.4 * bb * bb\n'
               'pp = 300.0\n'
               'qq = pp + thr * np.array([1.0 + 2.0e-10, 1.0 - 2.0e-10, 1.0 + 5.0e-11, 1.0 - '
               '5.0e-11])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(regulated_scattering_rate, 2.0, bb, pp, qq, 0.0002, 0.0, 40.0, '
              '1000.0, 2000.0, 0.0)',
      'gold_call': '_independent_call(_oracle_regulated_scattering_rate, 2.0, bb, pp, qq, 0.0002, 0.0, '
                   '40.0, 1000.0, 2000.0, 0.0)'},
     {'setup': 'import numpy as np\n'
               'def _probe_call():\n'
               '    try:\n'
               '        regulated_scattering_rate(1.0, 1.0, 1.0, 2.0, 0.0, 5.0, 40.0, 1.0, 2000.0, '
               '1.0, c_th=0.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def _probe_gold():\n'
               '    try:\n'
               '        _oracle_regulated_scattering_rate(1.0, 1.0, 1.0, 2.0, 0.0, 5.0, 40.0, 1.0, '
               '2000.0, 1.0, c_th=0.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_call()',
      'gold_call': '_probe_gold()'}]
