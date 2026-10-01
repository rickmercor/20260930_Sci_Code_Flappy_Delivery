"""
Step 03: the speed at which each wave channel is carried along the bundle.

Contract
--------
Two families of waves travel along the bundle and they do not share a
propagation speed.

- Row 0: the speed at which the Alfven channel is carried. It is built on the
field strength and on the single density the one-dimensional model carries,
the area-weighted mean.
- Row 1: the speed at which the transverse displacement channel is carried.
That channel is a collective oscillation in which a strand and the material
surrounding it move together, so its speed is set by the field strength and
by BOTH component densities, and it is not the Alfven speed of either
component alone. The same field threads the strand interior and the ambient
medium.

Neither speed is written out here; supply both.

Conventions
-----------
Field strength in G, all three densities in 1e-12 kg m^-3, returned speeds in
Mm s^-1. Use $mu_0 = 4 pi x 1e-7$ in SI. All four inputs broadcast to a common
shape. Return a float array of shape (2, N) with the rows in the order above;
for float inputs return shape (2,).

Validation
----------
Raise ValueError if the four inputs do not broadcast to a common shape, if any
value is not finite, if any field strength is not strictly positive, or if any
of the three densities is not strictly positive.

Returns
-------
A float array of shape (2, N) holding the speed at which the Alfven channel is carried and the speed at which the transverse channel is carried, both in Mm s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def channel_speeds(B: "ArrayLike", rho_avg: "ArrayLike", rho_i: "ArrayLike",
                   rho_e: "ArrayLike") -> "np.ndarray":
    '''Propagation speed of each wave channel, in Mm s^-1.

    Parameters
    ----------
    B : array_like
        Magnetic field strength in G. Finite and strictly positive.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_i : array_like
        Strand-interior mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_e : array_like
        Ambient mass density in 1e-12 kg m^-3. Finite and > 0.

    Returns
    -------
    np.ndarray
        Shape (2, N) float array in Mm s^-1. Row 0 is the speed of the Alfven
        channel; row 1 is the speed of the transverse displacement channel.

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


def _broadcast_all(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    try:
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    return [np.broadcast_to(a, shape).astype(float) for a in arrs]


def _oracle_channel_speeds(B: "ArrayLike", rho_avg: "ArrayLike",
                           rho_i: "ArrayLike", rho_e: "ArrayLike") -> "np.ndarray":
    b, ra, ri, re = _broadcast_all(B, rho_avg, rho_i, rho_e)
    for name, arr in (("B", b), ("rho_avg", ra), ("rho_i", ri), ("rho_e", re)):
        if not np.all(np.isfinite(arr)):
            raise ValueError("%s must be finite" % name)
        if np.any(arr <= 0.0):
            raise ValueError("%s must be strictly positive" % name)

    # Gauss and 1e-12 kg m^-3 in, Mm s^-1 out:
    #   v[m/s] = (1e-4 B) / sqrt(mu_0 * 1e-12 * rho), then divide by 1e6
    mu_0 = 4.0e-7 * np.pi
    speed_factor = 1.0e-6 * 1.0e-4 / np.sqrt(1.0e-12 * mu_0)
    v_alfven = speed_factor * b / np.sqrt(ra)
    v_kink = speed_factor * b * np.sqrt(2.0 / (ri + re))
    return np.asarray([v_alfven, v_kink], dtype=float)

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
               '    return (bg[3].copy(), bg[1].copy(), sp[1].copy(), sp[0].copy())\n',
      'call': 'channel_speeds(*_args(cross_section_structure, stratified_background))',
      'gold_call': '_oracle_channel_speeds(*_args(_oracle_cross_section_structure, '
                   '_oracle_stratified_background))'},
     {'setup': 'import numpy as np\n'
               'def _args():\n'
               '    return (np.array([1.0]), np.array([1.0]), np.array([1.0]),\n'
               '            np.array([1.0]))\n',
      'call': 'channel_speeds(*_args())',
      'gold_call': '_oracle_channel_speeds(*_args())'},
     {'setup': 'import numpy as np\n'
               'def _code(fn):\n'
               '    try:\n'
               '        fn(np.array([1.0, 1.0]), np.array([1.0, 0.0]),\n'
               '           np.array([1.0, 1.0]), np.array([1.0, 1.0]))\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_code(channel_speeds)',
      'gold_call': '_code(_oracle_channel_speeds)'}]
