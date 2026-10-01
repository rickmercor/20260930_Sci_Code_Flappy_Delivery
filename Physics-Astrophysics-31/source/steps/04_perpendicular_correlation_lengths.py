"""
Step 04: the perpendicular correlation length of each wave channel.



Contract

--------

Evaluate the two correlation lengths specified in the Scientific background.

Return the transverse displacement (kink) channel length in row 0 and the

Alfven channel length in row 1. The first depends on strand radius, density

contrast and filling factor; the second depends on local temperature and

field strength.

## Formulas

--------

Use the following two closures:



    L_kink = sqrt(2) * R * sqrt(5*pi*f) * (zeta + 1 - f)^(3/2)

             / ((zeta - 1) * (1 - f^(5/2)))



    L_alfven = 100 * sqrt(T / B)



R, L_kink and L_alfven are in Mm, T is in MK and B is in gauss.

The second expression is equivalently 1e8 * sqrt(T[MK] / B[G]) metres.

Its numerical normalization and unit convention are part of this contract.

The first length scales linearly with R and diverges as zeta approaches

unity from above.



Conventions

-----------

R, zeta, T and B broadcast to a common shape; f is a scalar.

Contrast and filling factor are dimensionless. Return a float array with

a leading axis of length 2 followed by the common broadcast shape, with

rows ordered as L_kink, L_alfven. Thus one-dimensional inputs of length N

give shape (2, N), and scalar inputs give shape (2,).



Validation

----------

Raise ValueError if R, zeta, T and B do not broadcast to a common shape,

if any input value is not finite, if any of R, T or B is not strictly

positive, if any zeta is not strictly greater than 1, or if f does not

satisfy 0 < f < 1.

Returns
-------
A float array of shape (2, N) holding the perpendicular correlation length of the transverse channel and that of the Alfven channel, both in Mm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def perpendicular_correlation_lengths(R: "ArrayLike", zeta: "ArrayLike",
                                      f: float, T: "ArrayLike",
                                      B: "ArrayLike") -> "np.ndarray":
    '''Perpendicular correlation length of each channel, in Mm.

    Parameters
    ----------
    R : array_like
        Strand radius in Mm. Finite and strictly positive.
    zeta : array_like
        Density contrast, broadcastable against ``R``. Finite and > 1.
    f : float
        Fraction of the cross-sectional area occupied by strand interior.
        Finite, with 0 < f < 1.
    T : array_like
        Temperature in MK, broadcastable against ``R``. Finite and > 0.
    B : array_like
        Field strength in G, broadcastable against ``R``. Finite and > 0.

    Returns
    -------
    np.ndarray
        Shape (2, N) float array in Mm. Row 0 is the perpendicular correlation
        length of the transverse displacement channel; row 1 is that of the
        Alfven channel.

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


def _broadcast_four(R, zeta, T, B):
    arrs = [np.asarray(a, dtype=float) for a in (R, zeta, T, B)]
    try:
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    return [np.broadcast_to(a, shape).astype(float) for a in arrs]


def _oracle_perpendicular_correlation_lengths(R: "ArrayLike", zeta: "ArrayLike",
                                              f: float, T: "ArrayLike",
                                              B: "ArrayLike") -> "np.ndarray":
    r, zt, t, b = _broadcast_four(R, zeta, T, B)
    for name, arr in (("R", r), ("zeta", zt), ("T", t), ("B", b)):
        if not np.all(np.isfinite(arr)):
            raise ValueError("%s must be finite" % name)
    for name, arr in (("R", r), ("T", t), ("B", b)):
        if np.any(arr <= 0.0):
            raise ValueError("%s must be strictly positive" % name)
    if np.any(zt <= 1.0):
        raise ValueError("zeta must be strictly greater than 1")
    ff = float(f)
    if not np.isfinite(ff) or not (0.0 < ff < 1.0):
        raise ValueError("f must be finite and satisfy 0 < f < 1")

    inverse_kink = (np.sqrt(2.0) * (zt - 1.0)
                    / (2.0 * r * np.sqrt(5.0 * ff * np.pi))
                    * (1.0 - ff ** 2.5) / (zt + 1.0 - ff) ** 1.5)
    L_kink = 1.0 / inverse_kink
    # 1e8 * sqrt(T[MK] / B[G]) metres, expressed in Mm
    alfven_length_constant_mm = 1.0e8 / 1.0e6
    L_alfven = alfven_length_constant_mm * np.sqrt(t / b)
    return np.asarray([L_kink, L_alfven], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Separate candidate and oracle fixture dependencies and mutable inputs."""
    return [{'setup': 'import numpy as np\n'
               '\n'
               'def _args(stratified_background):\n'
               '    z = np.linspace(0.0, 120.0, 41)\n'
               '    bg = stratified_background(z.copy(), 1.5, 42.0, 12.5, 0.8, 3.6, 400.0, 0.62, 0.83, '
               '45.0, 0.1)\n'
               '    return (bg[4].copy(), bg[5].copy(), 0.16, bg[2].copy(), bg[3].copy())\n',
      'call': 'perpendicular_correlation_lengths(*_args(stratified_background))',
      'gold_call': '_oracle_perpendicular_correlation_lengths(*_args(_oracle_stratified_background))'},
     {'setup': 'import numpy as np\n'
               'def _args():\n'
               '    return (np.array([0.8]), np.array([1.0 + 1.0e-9]), 0.99,\n'
               '            np.array([0.62]), np.array([12.5]))\n',
      'call': 'perpendicular_correlation_lengths(*_args())',
      'gold_call': '_oracle_perpendicular_correlation_lengths(*_args())'},
     {'setup': 'import numpy as np\n'
               'def _code(fn):\n'
               '    try:\n'
               '        fn(np.array([0.8, 0.8]), np.array([1.0, 2.0]), 0.16,\n'
               '           np.array([1.0, 1.0]), np.array([10.0, 10.0]))\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_code(perpendicular_correlation_lengths)',
      'gold_call': '_code(_oracle_perpendicular_correlation_lengths)'}]
