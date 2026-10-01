"""
Step 08: optically thin radiative loss rate of the structured bundle.

Contract
--------
Radiation is a local, optically thin, two-body process: at every point inside
the bundle the plasma loses energy at a volumetric rate equal to the radiative
loss function evaluated at the local temperature, multiplied by the square of
the LOCAL hydrogen number density at that point.

The bundle is not homogeneous across its cross-section. A fraction $f$ of the
area is strand interior at mass density $rho_i$, the rest is ambient material
at $rho_e$, and the one-dimensional model carries only the area-weighted mean
$rho_avg$ and the corresponding area-weighted mean hydrogen number density
$n_H$. The composition is the same everywhere, so at any point the hydrogen
number density is in the same ratio to $n_H$ as the local mass density is to
$rho_avg$. The temperature is uniform across the cross-section.

Return the loss rate per unit volume of the bundle as a whole. How that rate
follows from the arguments is NOT written out here; supply it.

The radiative loss function is prescribed, not modelled. It is the single
power law

$Lambda(T) = lambda0 (T / 1 MK)^lambda_exponent$

with `lambda0` given in units of 1e-35 W m^3 and $lambda_exponent$ the
power-law index.

Conventions
-----------
Hydrogen number density in 1e15 m^-3, temperature in MK, the three densities in
1e-12 kg m^-3, $f$ dimensionless, `lambda0` in 1e-35 W m^3, and the
returned loss rate in uW m^-3. The five array arguments broadcast to a common
shape. Return a float array of that shape; for float inputs return a float
array of shape ().

Validation
----------
Raise ValueError if $n_H$, $T$, $rho_avg$, $rho_e$ and $rho_i$ do not
broadcast to a common shape, if any value is not finite, if any $n_H$ is
negative, if any $T$ is not strictly positive, if any of the three densities
is not strictly positive, if $f$ does not satisfy 0 < f < 1, or if
`lambda0` is not finite or not strictly positive.

Returns
-------
A float array of the broadcast shape holding the volumetric radiative loss rate of the bundle in uW m^-3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radiative_loss_profile(n_H: "ArrayLike", T: "ArrayLike",
                           rho_avg: "ArrayLike", rho_e: "ArrayLike",
                           rho_i: "ArrayLike", f: float, lambda0: float,
                           lambda_exponent: float) -> "np.ndarray":
    '''Volumetric optically thin radiative loss rate of the bundle, in uW m^-3.

    Parameters
    ----------
    n_H : array_like
        Area-weighted mean hydrogen number density over the cross-section, in
        1e15 m^-3. Finite and non-negative.
    T : array_like
        Temperature in MK, broadcastable against ``n_H``. Finite and > 0.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_e : array_like
        Ambient mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_i : array_like
        Strand-interior mass density in 1e-12 kg m^-3. Finite and > 0.
    f : float
        Fraction of the cross-sectional area occupied by strand interior.
        Finite, with 0 < f < 1.
    lambda0 : float
        Radiative loss function at 1 MK, in units of 1e-35 W m^3. Finite and
        strictly positive.
    lambda_exponent : float
        Power-law index of the radiative loss function. Finite.

    Returns
    -------
    np.ndarray
        Volumetric radiative loss rate in uW m^-3, of the broadcast shape.

    Raises
    ------
    ValueError
        On non-broadcastable, non-finite or out-of-range input.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_radiative_loss_profile(n_H: "ArrayLike", T: "ArrayLike",
                                   rho_avg: "ArrayLike", rho_e: "ArrayLike",
                                   rho_i: "ArrayLike", f: float, lambda0: float,
                                   lambda_exponent: float) -> "np.ndarray":
    arrs = [np.asarray(a, dtype=float) for a in (n_H, T, rho_avg, rho_e, rho_i)]
    try:
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    nh, temp, ra, re, ri = [np.broadcast_to(a, shape).astype(float) for a in arrs]

    names = ("n_H", "T", "rho_avg", "rho_e", "rho_i")
    for name, arr in zip(names, (nh, temp, ra, re, ri)):
        if not np.all(np.isfinite(arr)):
            raise ValueError("%s must be finite" % name)
    if np.any(nh < 0.0):
        raise ValueError("n_H must be non-negative")
    if np.any(temp <= 0.0):
        raise ValueError("T must be strictly positive")
    for name, arr in (("rho_avg", ra), ("rho_e", re), ("rho_i", ri)):
        if np.any(arr <= 0.0):
            raise ValueError("%s must be strictly positive" % name)
    ff = float(f)
    if not np.isfinite(ff) or not (0.0 < ff < 1.0):
        raise ValueError("f must be finite and satisfy 0 < f < 1")
    lam0 = float(lambda0)
    if not np.isfinite(lam0) or lam0 <= 0.0:
        raise ValueError("lambda0 must be finite and strictly positive")
    exponent = float(lambda_exponent)
    if not np.isfinite(exponent):
        raise ValueError("lambda_exponent must be finite")

    n_ambient = nh * re / ra
    n_interior = nh * ri / ra
    mean_square = (1.0 - ff) * n_ambient ** 2 + ff * n_interior ** 2
    # (1e15 m^-3)^2 * 1e-35 W m^3 = 1e-5 W m^-3 = 10 uW m^-3
    loss_factor = 10.0
    return loss_factor * mean_square * lam0 * temp ** exponent

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Separate candidate and oracle fixture dependencies and mutable inputs."""
    return [{'setup': 'import numpy as np\n'
               '\n'
               'def _args(cross_section_structure, stratified_background):\n'
               '    z = np.linspace(0.0, 120.0, 41)\n'
               '    bg = stratified_background(z.copy(), 1.5, 42.0, 12.5, 0.8, 3.6, 400.0, 0.62, 0.83, '
               '45.0, 0.1)\n'
               '    sp = cross_section_structure(bg[1].copy(), bg[5].copy(), 0.16)\n'
               '    return (bg[0].copy(), bg[2].copy(), bg[1].copy(), sp[0].copy(), sp[1].copy(), '
               '0.16, 1.15, -0.5)\n',
      'call': 'radiative_loss_profile(*_args(cross_section_structure, stratified_background))',
      'gold_call': '_oracle_radiative_loss_profile(*_args(_oracle_cross_section_structure, '
                   '_oracle_stratified_background))'},
     {'setup': 'import numpy as np\n'
               'def _args():\n'
               '    return (np.array([0.0, 1.0]), np.array([1.0, 1.0]),\n'
               '            np.array([1.0, 1.0]), np.array([1.0, 1.0]),\n'
               '            np.array([1.0, 1.0]), 0.25, 1.15, 0.0)\n',
      'call': 'radiative_loss_profile(*_args())',
      'gold_call': '_oracle_radiative_loss_profile(*_args())'},
     {'setup': 'import numpy as np\n'
               'def _code(fn):\n'
               '    try:\n'
               '        fn(1.0, 0.0, 1.0, 1.0, 1.0, 0.16, 1.15, -0.5)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_code(radiative_loss_profile)',
      'gold_call': '_code(_oracle_radiative_loss_profile)'}]
