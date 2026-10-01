"""
Step 06: steady-state profile of the transverse displacement channel.

Contract
--------
Only an outward-travelling population of this channel is present, injected at
the base with energy density $W_base$, so its profile follows from a single
integration from the base to the top of the grid.

The background is static and there is no bulk flow, so in steady state the
height derivative of this channel's energy flux balances minus the volumetric
rate at which the channel deposits energy. The flux is the channel's energy
density times the speed at which it is carried. The deposition rate is the one
returned in row 0 by $deposition_rates$; call that function, do not
re-derive it here.

Conventions
-----------
Heights in Mm on a UNIFORM grid, propagation speed in Mm s^-1, mass densities
in 1e-12 kg m^-3, correlation length in Mm, wave energy density in mJ m^-3, and
the returned deposition rate in uW m^-3. In these units the height derivative
of the flux equals minus 1e-3 times the deposition rate; that factor is pure
unit bookkeeping.

Integrate the FLUX variable with classical fourth-order Runge-Kutta, one step
per grid interval. The background arrays are needed at the half-steps: obtain
them with the fourth-order midpoint interpolant
(-y[i-1] + 9 y[i] + 9 y[i+1] - y[i+2]) / 16 on the interior, and in the first
and last intervals with the four-point Lagrange interpolant through the four
nearest nodes evaluated at the half-step.

Return a float array of shape (2, N): row 0 the channel energy density in
mJ m^-3, row 1 its volumetric deposition rate in uW m^-3.

Validation
----------
Raise ValueError if $z$ is not a one-dimensional, strictly increasing,
uniformly spaced grid of at least four finite points; if $v_kink$,
$rho_avg$, $rho_e$ or $L_kink$ is not a finite, strictly positive array
of the same length as $z$; or if $W_base$ is not finite or is negative.

Returns
-------
A float array of shape (2, N) holding the transverse channel's energy density in mJ m^-3 and its volumetric deposition rate in uW m^-3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transverse_channel_profile(z: "ArrayLike", v_kink: "ArrayLike",
                               rho_avg: "ArrayLike", rho_e: "ArrayLike",
                               L_kink: "ArrayLike",
                               W_base: float) -> "np.ndarray":
    '''Steady-state energy density and deposition rate of the transverse channel.

    Parameters
    ----------
    z : array_like
        One-dimensional uniform height grid in Mm, strictly increasing, at
        least four points.
    v_kink : array_like
        Speed at which the transverse channel is carried, in Mm s^-1, one
        value per grid point. Finite and strictly positive.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3, one value per grid
        point. Finite and strictly positive.
    rho_e : array_like
        Ambient mass density in 1e-12 kg m^-3, one value per grid point.
        Finite and strictly positive.
    L_kink : array_like
        Perpendicular correlation length of this channel in Mm, one value per
        grid point. Finite and strictly positive.
    W_base : float
        Energy density injected at the base of the domain, in mJ m^-3. Finite
        and non-negative.

    Returns
    -------
    np.ndarray
        Shape (2, N) float array. Row 0 is the channel energy density in
        mJ m^-3, row 1 its volumetric deposition rate in uW m^-3.

    Raises
    ------
    ValueError
        On a malformed grid, a malformed background array or a negative
        injection.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _uniform_grid(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 4:
        raise ValueError("z must be a one-dimensional grid of at least four points")
    if not np.all(np.isfinite(z)):
        raise ValueError("z must be finite")
    step = np.diff(z)
    if np.any(step <= 0.0):
        raise ValueError("z must be strictly increasing")
    if not np.allclose(step, step[0], rtol=1e-12, atol=0.0):
        raise ValueError("z must be uniformly spaced")
    return z, float(step[0])


def _positive_profile(name, arr, n):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1 or a.size != n:
        raise ValueError("%s must have one value per grid point" % name)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if np.any(a <= 0.0):
        raise ValueError("%s must be strictly positive" % name)
    return a


def _midpoints(y):
    """Fourth-order values of y at the interval midpoints of a uniform grid."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.empty(n - 1, dtype=float)
    out[1:n - 2] = (-y[0:n - 3] + 9.0 * y[1:n - 2]
                    + 9.0 * y[2:n - 1] - y[3:n]) / 16.0
    out[0] = (5.0 * y[0] + 15.0 * y[1] - 5.0 * y[2] + y[3]) / 16.0
    out[n - 2] = (y[n - 4] - 5.0 * y[n - 3]
                  + 15.0 * y[n - 2] + 5.0 * y[n - 1]) / 16.0
    return out


def _oracle_transverse_channel_profile(z: "ArrayLike", v_kink: "ArrayLike",
                                       rho_avg: "ArrayLike", rho_e: "ArrayLike",
                                       L_kink: "ArrayLike",
                                       W_base: float) -> "np.ndarray":
    z, step = _uniform_grid(z)
    n = z.size
    v = _positive_profile("v_kink", v_kink, n)
    ra = _positive_profile("rho_avg", rho_avg, n)
    re = _positive_profile("rho_e", rho_e, n)
    lk = _positive_profile("L_kink", L_kink, n)
    w0 = float(W_base)
    if not np.isfinite(w0) or w0 < 0.0:
        raise ValueError("W_base must be finite and non-negative")

    hv, hra, hre, hlk = (_midpoints(v), _midpoints(ra),
                         _midpoints(re), _midpoints(lk))

    # With heights in Mm, speeds in Mm s^-1 and energy densities in mJ m^-3,
    # the height derivative of a flux is 1e-3 times a rate in uW m^-3.
    flux_unit_factor = 1.0e-3

    def _slope(flux, speed, rho_x, length):
        energy = max(flux, 0.0) / speed
        return -flux_unit_factor * _kink_rate(energy, rho_x, length)

    flux = np.empty(n, dtype=float)
    flux[0] = w0 * v[0]
    for i in range(n - 1):
        k1 = _slope(flux[i], v[i], re[i], lk[i])
        k2 = _slope(flux[i] + 0.5 * step * k1, hv[i], hre[i], hlk[i])
        k3 = _slope(flux[i] + 0.5 * step * k2, hv[i], hre[i], hlk[i])
        k4 = _slope(flux[i] + step * k3, v[i + 1], re[i + 1], lk[i + 1])
        flux[i + 1] = flux[i] + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    energy = np.clip(flux, 0.0, None) / v
    zeros = np.zeros(n, dtype=float)
    rate = _oracle_deposition_rates(energy, zeros, zeros, ra, re, lk,
                                    np.ones(n, dtype=float))[0]
    return np.asarray([energy, rate], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Separate candidate and oracle fixture dependencies and mutable inputs."""
    return [{'setup': 'import numpy as np\n'
               '\n'
               'def _args(channel_speeds, cross_section_structure, perpendicular_correlation_lengths, '
               'stratified_background):\n'
               '    z = np.linspace(0.0, 120.0, 121)\n'
               '    bg = stratified_background(z.copy(), 1.5, 42.0, 12.5, 0.8, 3.6, 400.0, 0.62, 0.83, '
               '45.0, 0.1)\n'
               '    sp = cross_section_structure(bg[1].copy(), bg[5].copy(), 0.16)\n'
               '    vv = channel_speeds(bg[3].copy(), bg[1].copy(), sp[1].copy(), sp[0].copy())\n'
               '    ll = perpendicular_correlation_lengths(bg[4].copy(), bg[5].copy(), 0.16, '
               'bg[2].copy(), bg[3].copy())\n'
               '    return (z, vv[1].copy(), bg[1].copy(), sp[0].copy(), ll[0].copy(), 0.9)\n',
      'call': 'transverse_channel_profile(*_args(channel_speeds, cross_section_structure, '
              'perpendicular_correlation_lengths, stratified_background))',
      'gold_call': '_oracle_transverse_channel_profile(*_args(_oracle_channel_speeds, '
                   '_oracle_cross_section_structure, _oracle_perpendicular_correlation_lengths, '
                   '_oracle_stratified_background))'},
     {'setup': 'import numpy as np\n'
               'def _args():\n'
               '    return (np.linspace(0.0, 3.0, 4), np.ones(4), np.ones(4),\n'
               '            np.ones(4), np.ones(4), 0.0)\n',
      'call': 'transverse_channel_profile(*_args())',
      'gold_call': '_oracle_transverse_channel_profile(*_args())'},
     {'setup': 'import numpy as np\n'
               'def _code(fn):\n'
               '    z = np.array([0.0, 1.0, 2.0, 4.0])\n'
               '    one = np.ones(4)\n'
               '    try:\n'
               '        fn(z, one, one, one, one, 0.5)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_code(transverse_channel_profile)',
      'gold_call': '_code(_oracle_transverse_channel_profile)'}]
