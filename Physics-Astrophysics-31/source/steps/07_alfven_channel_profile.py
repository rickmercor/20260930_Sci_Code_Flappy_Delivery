"""
Step 07: steady-state profile of the two-population Alfven channel.

Contract
--------
This channel carries TWO populations. An outward-travelling one is injected at
the base of the domain with energy density $W_out_base$; an inward-travelling
one enters at the top with energy density $W_in_top$ and propagates towards
the base. Both are carried at the Alfven speed.

As in step 06 the background is static, so in steady state the height
derivative of a population's energy flux balances minus the volumetric rate at
which that population deposits energy, and the flux is that population's energy
density times the Alfven speed. Those two rates are the ones returned in rows 1
and 2 by $deposition_rates$; call that function, do not re-derive them here.
Because the two boundary conditions sit at opposite ends of the domain, this is
a two-point boundary-value problem and must be relaxed rather than integrated
once.

Conventions
-----------
Units as in step 06, and the same 1e-3 factor between the height derivative of
a flux and a deposition rate in uW m^-3. Both populations are advanced in their
FLUX variable with classical fourth-order Runge-Kutta, one step per grid
interval, using the same fourth-order half-step interpolation as step 06 for
every array evaluated between nodes. The outward population is integrated from
the base upwards. The inward population is integrated from the top downwards,
and because it decays along its own downward path its flux INCREASES with
height.

Relax by Picard iteration in this order: initialise the inward profile
uniformly at $W_in_top$; recompute the outward profile from the current
inward profile; recompute the inward profile from the outward profile just
obtained; take the relative change as the largest absolute change in the
inward profile divided by its largest absolute value; stop as soon as that is
below `tol`, or after `itmax` sweeps.

Return a float array of shape (3, N): row 0 the outward energy density, row 1
the inward energy density, both in mJ m^-3, and row 2 the total volumetric
deposition rate of the channel in uW m^-3, summed over both populations.

Validation
----------
Raise ValueError if $z$ is not a one-dimensional, strictly increasing,
uniformly spaced grid of at least four finite points; if $v_alfven$,
$rho_avg$ or $L_alfven$ is not a finite, strictly positive array of the
same length as $z$; if $W_out_base$ or $W_in_top$ is not finite or is
negative; if `tol` is not finite or not strictly positive; or if `itmax`
is not an integer of at least 1.

Returns
-------
A float array of shape (3, N) holding the outward and inward Alfven energy densities in mJ m^-3 and the channel's total volumetric deposition rate in uW m^-3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def alfven_channel_profile(z: "ArrayLike", v_alfven: "ArrayLike",
                           rho_avg: "ArrayLike", L_alfven: "ArrayLike",
                           W_out_base: float, W_in_top: float,
                           tol: float = 1e-15,
                           itmax: int = 500) -> "np.ndarray":
    '''Steady-state energy densities and deposition rate of the Alfven channel.

    Parameters
    ----------
    z : array_like
        One-dimensional uniform height grid in Mm, strictly increasing, at
        least four points.
    v_alfven : array_like
        Alfven speed in Mm s^-1, one value per grid point. Finite and > 0.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3, one value per grid
        point. Finite and strictly positive.
    L_alfven : array_like
        Perpendicular correlation length of this channel in Mm, one value per
        grid point. Finite and strictly positive.
    W_out_base : float
        Outward energy density injected at the base, in mJ m^-3. Finite and
        non-negative.
    W_in_top : float
        Inward energy density entering at the top, in mJ m^-3. Finite and
        non-negative.
    tol : float, optional
        Relative convergence tolerance of the Picard relaxation. Finite, > 0.
    itmax : int, optional
        Maximum number of Picard sweeps. Integer, at least 1.

    Returns
    -------
    np.ndarray
        Shape (3, N) float array. Row 0 outward energy density, row 1 inward
        energy density, both in mJ m^-3; row 2 the total volumetric deposition
        rate of the channel in uW m^-3.

    Raises
    ------
    ValueError
        On a malformed grid, a malformed background array, a negative
        injection or a malformed relaxation control.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_alfven_channel_profile(z: "ArrayLike", v_alfven: "ArrayLike",
                                   rho_avg: "ArrayLike", L_alfven: "ArrayLike",
                                   W_out_base: float, W_in_top: float,
                                   tol: float = 1e-15,
                                   itmax: int = 500) -> "np.ndarray":
    z, step = _uniform_grid(z)
    n = z.size
    v = _positive_profile("v_alfven", v_alfven, n)
    ra = _positive_profile("rho_avg", rho_avg, n)
    la = _positive_profile("L_alfven", L_alfven, n)
    w_out0 = float(W_out_base)
    w_in_top = float(W_in_top)
    for name, value in (("W_out_base", w_out0), ("W_in_top", w_in_top)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("%s must be finite and non-negative" % name)
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be finite and strictly positive")
    if not isinstance(itmax, (int, np.integer)) or int(itmax) < 1:
        raise ValueError("itmax must be an integer of at least 1")

    ones = np.ones(n, dtype=float)
    hv, hra, hla = _midpoints(v), _midpoints(ra), _midpoints(la)
    # Same unit bookkeeping as step 06: with heights in Mm, speeds in Mm s^-1
    # and energy densities in mJ m^-3, the height derivative of a flux is 1e-3
    # times a deposition rate expressed in uW m^-3.
    flux_unit_factor = 1.0e-3

    def _out_rate(flux, speed, rho_a, length, partner):
        energy = max(flux, 0.0) / speed
        return _alfven_rate(energy, partner, rho_a, length)

    def _in_rate(flux, speed, rho_a, length, partner):
        energy = max(flux, 0.0) / speed
        return _alfven_rate(energy, partner, rho_a, length)

    w_in = np.full(n, w_in_top, dtype=float)
    w_out = np.full(n, w_out0, dtype=float)
    for _ in range(int(itmax)):
        h_in = _midpoints(w_in)
        flux = np.empty(n, dtype=float)
        flux[0] = w_out0 * v[0]
        for i in range(n - 1):
            k1 = -flux_unit_factor * _out_rate(flux[i], v[i], ra[i], la[i], w_in[i])
            k2 = -flux_unit_factor * _out_rate(flux[i] + 0.5 * step * k1,
                                               hv[i], hra[i], hla[i], h_in[i])
            k3 = -flux_unit_factor * _out_rate(flux[i] + 0.5 * step * k2,
                                               hv[i], hra[i], hla[i], h_in[i])
            k4 = -flux_unit_factor * _out_rate(flux[i] + step * k3, v[i + 1],
                                               ra[i + 1], la[i + 1], w_in[i + 1])
            flux[i + 1] = flux[i] + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        w_out_new = np.clip(flux, 0.0, None) / v

        h_out = _midpoints(w_out_new)
        gflux = np.empty(n, dtype=float)
        gflux[n - 1] = w_in_top * v[n - 1]
        for i in range(n - 1, 0, -1):
            k1 = flux_unit_factor * _in_rate(gflux[i], v[i], ra[i], la[i],
                                             w_out_new[i])
            k2 = flux_unit_factor * _in_rate(gflux[i] - 0.5 * step * k1, hv[i - 1],
                                             hra[i - 1], hla[i - 1], h_out[i - 1])
            k3 = flux_unit_factor * _in_rate(gflux[i] - 0.5 * step * k2, hv[i - 1],
                                             hra[i - 1], hla[i - 1], h_out[i - 1])
            k4 = flux_unit_factor * _in_rate(gflux[i] - step * k3, v[i - 1],
                                             ra[i - 1], la[i - 1], w_out_new[i - 1])
            gflux[i - 1] = gflux[i] - step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        w_in_new = np.clip(gflux, 0.0, None) / v

        scale = np.max(np.abs(w_in_new))
        change = np.max(np.abs(w_in_new - w_in)) / (scale if scale > 0.0 else 1.0)
        w_out, w_in = w_out_new, w_in_new
        if change < tolerance:
            break

    zeros = np.zeros(n, dtype=float)
    rates = _oracle_deposition_rates(zeros, w_out, w_in, ra, ones, ones, la)
    return np.asarray([w_out, w_in, rates[1] + rates[2]], dtype=float)

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
               '    return (z, vv[0].copy(), bg[1].copy(), ll[1].copy(), 0.5, 0.025)\n'
               "kwargs = {'tol': 1e-15, 'itmax': 500}\n",
      'call': 'alfven_channel_profile(*_args(channel_speeds, cross_section_structure, '
              'perpendicular_correlation_lengths, stratified_background), **kwargs)',
      'gold_call': '_oracle_alfven_channel_profile(*_args(_oracle_channel_speeds, '
                   '_oracle_cross_section_structure, _oracle_perpendicular_correlation_lengths, '
                   '_oracle_stratified_background), **kwargs)'},
     {'setup': 'import numpy as np\n'
               'def _args():\n'
               '    return (np.linspace(0.0, 6.0, 7), np.ones(7), np.ones(7),\n'
               '            np.ones(7), 0.4, 0.0)\n'
               "kwargs = {'tol': 1.0e-14, 'itmax': 50}\n",
      'call': 'alfven_channel_profile(*_args(), **kwargs)',
      'gold_call': '_oracle_alfven_channel_profile(*_args(), **kwargs)'},
     {'setup': 'import numpy as np\n'
               'def _code(fn):\n'
               '    z = np.linspace(0.0, 4.0, 5)\n'
               '    one = np.ones(5)\n'
               '    try:\n'
               '        fn(z, one, one, one, 0.5, 0.02, 1.0e-14, 0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_code(alfven_channel_profile)',
      'gold_call': '_code(_oracle_alfven_channel_profile)'}]
