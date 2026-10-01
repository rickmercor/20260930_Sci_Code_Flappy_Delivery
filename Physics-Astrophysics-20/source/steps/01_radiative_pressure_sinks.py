"""
Synchrotron radiation-reaction sinks of the two pressures.

Synchrotron radiation-reaction sinks of the two pressures.

The medium is an ultra-relativistic, gyrotropic electron-positron pair plasma in
a magnetic field of strength B pointing along z. Synchrotron emission, through
the classical radiation-reaction force that accompanies it, drains the total
perpendicular pressure P_perp and the total parallel pressure P_par. Taking
pressure moments of the kinetic equation brings in moments of the momentum
distribution beyond the two pressures; this task closes them with the two
dimensionless constants alpha_perp and alpha_par defined in the function
docstring. Nothing else acts in this step: no compression and no pitch-angle
scattering.

Normalised units shared by every step of the task: time in 1/Omega0, length in
rho_e0 = c/Omega0, velocity in c, density in n0 (so n0 = 1), pressure in
n0 m_e c^2 and magnetic field in sqrt(4 pi n0 m_e c^2). The reference state has
density n0, temperature T0 = theta0 m_e c^2, pressure P0 = n0 T0 and field B0,
with reference plasma beta beta0 = 8 pi P0 / B0^2 (physical units), relativistic
gyrofrequency Omega0 = e B0 c / (3 T0), synchrotron cooling time
tau0 = m_e^2 c^3 / (2 r_e^2 B0^2 T0) and chi = tau0 Omega0.

Returns
-------
The two radiative pressure sinks, stacked.

Returns
-------
The two radiative pressure sinks, stacked.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radiative_pressure_sinks(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    alpha_perp: float,
    alpha_par: float,
    beta0: float,
    theta0: float,
    chi: float,
) -> np.ndarray:
    """Return the synchrotron radiation-reaction sinks of the two pressures.

    Parameters
    ----------
    n : float or array_like
        Total (electron plus positron) number density in units of n0.
        Strictly positive.
    B : float or array_like
        Magnetic field strength in units of sqrt(4 pi n0 m_e c^2). Strictly
        positive.
    p_perp : float or array_like
        Total perpendicular pressure P_perp in units of n0 m_e c^2. Strictly
        positive.
    p_par : float or array_like
        Total parallel pressure P_par in units of n0 m_e c^2. Strictly
        positive.
    alpha_perp : float
        Closure constant
        alpha_perp = n int p_perp^4 gamma^-2 f d^3p / (6 m_e^2 P_perp^2),
        where inside the integral p_perp is the particle momentum component
        perpendicular to B, gamma the particle Lorentz factor and f the
        momentum distribution; it equals 16/15 for an isotropic
        ultra-relativistic thermal plasma. Non-negative.
    alpha_par : float
        Closure constant
        alpha_par = n int p_perp^2 p_par^2 gamma^-2 f d^3p / (3 m_e^2 P_perp P_par),
        with p_par the particle momentum component along B; it equals 8/15 for
        an isotropic ultra-relativistic thermal plasma. Non-negative.
    beta0 : float
        Reference plasma beta 8 pi P0 / B0^2. Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2). Strictly positive.
    chi : float
        Reference product tau0 Omega0 of the synchrotron cooling time and the
        relativistic gyrofrequency. Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2,) + S``, with ``S`` the broadcast shape of ``n``,
        ``B``, ``p_perp`` and ``p_par``. Row 0 is dP_perp/dt and row 1 is
        dP_par/dt caused by radiation reaction alone, in units of
        n0 m_e c^2 Omega0.

    Raises
    ------
    ValueError
        If ``n``, ``B``, ``p_perp`` or ``p_par`` has a non-finite or
        non-positive entry, if these four do not broadcast together, if
        ``alpha_perp`` or ``alpha_par`` is negative, non-finite or not a
        scalar, or if ``beta0``, ``theta0`` or ``chi`` is not a finite,
        strictly positive scalar.
    """
    return sinks  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike


def _rs_positive(name, x):
    """Float array of a strictly positive, finite input."""
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return a


def _rs_scalar(name, x, allow_zero=False):
    """Finite scalar that is strictly positive (or non-negative)."""
    if np.ndim(x) != 0:
        raise ValueError(f"{name} must be a scalar")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not np.isfinite(v) or v < 0.0 or (v == 0.0 and not allow_zero):
        kind = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be finite and {kind}")
    return v


def _rs_coupling(beta0, theta0, chi):
    """Radiative coupling constant of the normalised equations."""
    return beta0 / (2.0 * theta0 * theta0 * chi)


def _rs_sinks_kernel(kap, p, q, ap, aq):
    """Sinks for a precomputed per-cell rate factor kap (no validation)."""
    return -ap * kap * p * p, -aq * kap * p * q


def _oracle_radiative_pressure_sinks(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    alpha_perp: float,
    alpha_par: float,
    beta0: float,
    theta0: float,
    chi: float,
) -> np.ndarray:
    n = _rs_positive("n", n)
    B = _rs_positive("B", B)
    p = _rs_positive("p_perp", p_perp)
    q = _rs_positive("p_par", p_par)
    ap = _rs_scalar("alpha_perp", alpha_perp, allow_zero=True)
    aq = _rs_scalar("alpha_par", alpha_par, allow_zero=True)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    try:
        n, B, p, q = np.broadcast_arrays(n, B, p, q)
    except ValueError:
        raise ValueError("n, B, p_perp and p_par must broadcast together") from None
    kap = _rs_coupling(beta0, theta0, chi) * B * B / n
    rp, rq = _rs_sinks_kernel(kap, p, q, ap, aq)
    return np.stack((rp, rq))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cases with independent candidate and oracle argument graphs."""
    return [{'setup': 'import numpy as np\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(radiative_pressure_sinks, np.array([1.0, 0.72, 1.33]), '
              'np.array([7.0710678118654755, 5.091168824543142, 9.40452]), np.array([1000.0, 845.0, '
              '1120.0]), np.array([1000.0, 910.0, 1164.0]), 0.9, 0.48, 40.0, 1000.0, 2000.0)',
      'gold_call': '_independent_call(_oracle_radiative_pressure_sinks, np.array([1.0, 0.72, 1.33]), '
                   'np.array([7.0710678118654755, 5.091168824543142, 9.40452]), np.array([1000.0, '
                   '845.0, 1120.0]), np.array([1000.0, 910.0, 1164.0]), 0.9, 0.48, 40.0, 1000.0, '
                   '2000.0)'},
     {'setup': 'import numpy as np\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(radiative_pressure_sinks, 1.0, 0.223606797749979, 1.0, 1.0, 16.0 / '
              '15.0, 0.0, 40.0, 1.0, 2000.0)',
      'gold_call': '_independent_call(_oracle_radiative_pressure_sinks, 1.0, 0.223606797749979, 1.0, '
                   '1.0, 16.0 / 15.0, 0.0, 40.0, 1.0, 2000.0)'},
     {'setup': 'import numpy as np\n'
               'n_col = np.array([[0.05], [1.0], [3.0]])\n'
               'p_row = np.array([[2.0e-3, 0.4, 5.0, 60.0]])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(radiative_pressure_sinks, n_col, 2.5, p_row, 3.0 * p_row, 8.0 / 15.0, '
              '16.0 / 15.0, 4.0, 7.0, 150.0)',
      'gold_call': '_independent_call(_oracle_radiative_pressure_sinks, n_col, 2.5, p_row, 3.0 * '
                   'p_row, 8.0 / 15.0, 16.0 / 15.0, 4.0, 7.0, 150.0)'},
     {'setup': 'import numpy as np\n'
               'def _probe_call():\n'
               '    try:\n'
               '        radiative_pressure_sinks(np.array([1.0, 0.0]), 1.0, 1.0, 1.0, 0.9, 0.48, 40.0, '
               '1.0, 2000.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def _probe_gold():\n'
               '    try:\n'
               '        _oracle_radiative_pressure_sinks(np.array([1.0, 0.0]), 1.0, 1.0, 1.0, 0.9, '
               '0.48, 40.0, 1.0, 2000.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_call()',
      'gold_call': '_probe_gold()'}]
