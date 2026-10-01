"""
Step 05: the volumetric rate at which each wave population deposits its energy.



Contract

--------

Evaluate the three local dissipation closures specified in the Scientific

background. Given the wave energy densities, local mass densities and

correlation lengths, return the deposition rates of the transverse

displacement channel, the outward Alfven population and the inward Alfven

population, in that row order. Each rate depends only on the supplied

local arguments.

## Formulas

--------

In SI units, the closures are



    Q_kink = W_kink^(3/2) / (L_kink * sqrt(rho_e))

    Q_out  = 2 * W_out * sqrt(W_in) / (L_alfven * sqrt(rho_avg))

    Q_in   = 2 * W_in * sqrt(W_out) / (L_alfven * sqrt(rho_avg))



The kink rate uses the ambient density rho_e. The Alfven rates use the

area-weighted mean density rho_avg and the oppositely directed wave

population in the square root.



For this function, wave energy densities are supplied in mJ m^-3, mass

densities in units of 1e-12 kg m^-3, and lengths in Mm. Return rates in

uW m^-3. With C = 10^(3/2), the formulas in these supplied numerical units

are therefore



    Q_kink = C * max(W_kink, 0)^(3/2) / (L_kink * sqrt(rho_e))

    Q_out  = 2*C * max(W_out, 0) * sqrt(max(W_in, 0))

             / (L_alfven * sqrt(rho_avg))

    Q_in   = 2*C * max(W_in, 0) * sqrt(max(W_out, 0))

             / (L_alfven * sqrt(rho_avg))



Here max is elementwise. Treat a negative energy density as zero

wherever it enters a rate.



Conventions

-----------

All seven arguments broadcast to a common shape. Return a float array

with a leading axis of length 3 followed by that common broadcast shape,

with rows Q_kink, Q_out, Q_in. Thus one-dimensional inputs of length N

give shape (3, N), and scalar inputs give shape (3,).



Validation

----------

Raise ValueError if the seven arguments do not broadcast to a common

shape, if any input value is not finite, or if any of rho_avg, rho_e,

L_kink or L_alfven is not strictly positive.

Returns
-------
A float array of shape (3, N) holding the volumetric deposition rate of the transverse channel and those of the outward and inward Alfven populations, all in uW m^-3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def deposition_rates(W_kink: "ArrayLike", W_out: "ArrayLike",
                     W_in: "ArrayLike", rho_avg: "ArrayLike",
                     rho_e: "ArrayLike", L_kink: "ArrayLike",
                     L_alfven: "ArrayLike") -> "np.ndarray":
    '''Volumetric deposition rate of each wave population, in uW m^-3.

    Parameters
    ----------
    W_kink : array_like
        Energy density of the transverse displacement channel, in mJ m^-3.
        Finite.
    W_out : array_like
        Energy density of the outward-travelling Alfven population, in
        mJ m^-3. Finite.
    W_in : array_like
        Energy density of the inward-travelling Alfven population, in mJ m^-3.
        Finite.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_e : array_like
        Ambient mass density in 1e-12 kg m^-3. Finite and > 0.
    L_kink : array_like
        Perpendicular correlation length of the transverse channel, in Mm.
        Finite and > 0.
    L_alfven : array_like
        Perpendicular correlation length of the Alfven channel, in Mm. Finite
        and > 0.

    Returns
    -------
    np.ndarray
        Shape (3, N) float array in uW m^-3. Row 0 is the deposition rate of
        the transverse displacement channel, row 1 that of the outward Alfven
        population, row 2 that of the inward Alfven population.

    Raises
    ------
    ValueError
        On non-broadcastable, non-finite or non-positive input.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _rate_factor():
    """mJ m^-3 to the three halves, divided by Mm and by sqrt(1e-12 kg m^-3),
    expressed in uW m^-3:  (1e-3)**1.5 / (1e6 * sqrt(1e-12)) * 1e6 = 10**1.5"""
    return 10.0 ** 1.5


def _kink_rate(energy, rho_e, L_kink):
    """Scalar form of the transverse-channel closure, in uW m^-3."""
    w = energy if energy > 0.0 else 0.0
    return _rate_factor() * w ** 1.5 / (L_kink * rho_e ** 0.5)


def _alfven_rate(energy, partner, rho_avg, L_alfven):
    """Scalar form of one Alfven population's closure, in uW m^-3."""
    w = energy if energy > 0.0 else 0.0
    p = partner if partner > 0.0 else 0.0
    return 2.0 * _rate_factor() * p ** 0.5 * w / (L_alfven * rho_avg ** 0.5)


def _broadcast_seven(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    try:
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    return [np.broadcast_to(a, shape).astype(float) for a in arrs]


def _oracle_deposition_rates(W_kink: "ArrayLike", W_out: "ArrayLike",
                             W_in: "ArrayLike", rho_avg: "ArrayLike",
                             rho_e: "ArrayLike", L_kink: "ArrayLike",
                             L_alfven: "ArrayLike") -> "np.ndarray":
    wk, wo, wi, ra, re, lk, la = _broadcast_seven(
        W_kink, W_out, W_in, rho_avg, rho_e, L_kink, L_alfven)
    names = ("W_kink", "W_out", "W_in", "rho_avg", "rho_e", "L_kink", "L_alfven")
    for name, arr in zip(names, (wk, wo, wi, ra, re, lk, la)):
        if not np.all(np.isfinite(arr)):
            raise ValueError("%s must be finite" % name)
    for name, arr in (("rho_avg", ra), ("rho_e", re),
                      ("L_kink", lk), ("L_alfven", la)):
        if np.any(arr <= 0.0):
            raise ValueError("%s must be strictly positive" % name)

    wk = np.clip(wk, 0.0, None)
    wo = np.clip(wo, 0.0, None)
    wi = np.clip(wi, 0.0, None)

    rate_factor = _rate_factor()
    q_kink = rate_factor * wk ** 1.5 / (lk * np.sqrt(re))
    gamma = 2.0 * rate_factor / (la * np.sqrt(ra))
    q_out = gamma * np.sqrt(wi) * wo
    q_in = gamma * np.sqrt(wo) * wi
    return np.asarray([q_kink, q_out, q_in], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Separate candidate and oracle fixture dependencies and mutable inputs."""
    return [{'setup': 'import numpy as np\n'
               '\n'
               'def _args(cross_section_structure, perpendicular_correlation_lengths, '
               'stratified_background):\n'
               '    z = np.linspace(0.0, 120.0, 41)\n'
               '    bg = stratified_background(z.copy(), 1.5, 42.0, 12.5, 0.8, 3.6, 400.0, 0.62, 0.83, '
               '45.0, 0.1)\n'
               '    sp = cross_section_structure(bg[1].copy(), bg[5].copy(), 0.16)\n'
               '    ll = perpendicular_correlation_lengths(bg[4].copy(), bg[5].copy(), 0.16, '
               'bg[2].copy(), bg[3].copy())\n'
               '    wk = 0.9 * np.exp(-z / 90.0)\n'
               '    wo = 0.5 * np.exp(-z / 150.0)\n'
               '    wi = 0.025 * np.exp((z - 120.0) / 90.0)\n'
               '    return (wk, wo, wi, bg[1].copy(), sp[0].copy(), ll[0].copy(), ll[1].copy())\n',
      'call': 'deposition_rates(*_args(cross_section_structure, perpendicular_correlation_lengths, '
              'stratified_background))',
      'gold_call': '_oracle_deposition_rates(*_args(_oracle_cross_section_structure, '
                   '_oracle_perpendicular_correlation_lengths, _oracle_stratified_background))'},
     {'setup': 'import numpy as np\n'
               'def _args():\n'
               '    return (np.array([0.0, -1.0e-9]), np.array([0.5, 0.0]),\n'
               '            np.array([0.0, 0.3]), np.array([1.0, 1.0]),\n'
               '            np.array([1.0, 1.0]), np.array([1.0, 1.0]),\n'
               '            np.array([1.0, 1.0]))\n',
      'call': 'deposition_rates(*_args())',
      'gold_call': '_oracle_deposition_rates(*_args())'},
     {'setup': 'import numpy as np\n'
               'def _code(fn):\n'
               '    try:\n'
               '        fn(0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_code(deposition_rates)',
      'gold_call': '_code(_oracle_deposition_rates)'}]
