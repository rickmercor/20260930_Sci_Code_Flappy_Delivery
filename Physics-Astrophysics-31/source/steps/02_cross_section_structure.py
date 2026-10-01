"""
Step 02: resolve the cross-sectional density structure of the strand bundle.

Contract
--------
Seen end-on, the bundle is a two-component medium: a fraction $f$ of the
cross-sectional area is strand interior at mass density $rho_i$, the
remaining $1 - f$ is ambient material at $rho_e$, and the two are related
by the density contrast $zeta = rho_i / rho_e$. The single density carried by
the one-dimensional model is the area-weighted mean over the cross-section,
$rho_avg$, which is what this step is given.

Given $rho_avg$, `zeta` and $f$, return the two component densities that
reproduce that area-weighted mean.

Conventions
-----------
Densities are in units of 1e-12 kg m^-3. $rho_avg$ and `zeta` broadcast to
a common shape (a single station may be passed as a length-1 array or as a
float); $f$ is a scalar. Return a float array of shape (2, N) whose rows are,
in this order, the ambient density and the strand-interior density. For float
inputs return shape (2,).

Validation
----------
Raise ValueError if $rho_avg$ and `zeta` do not broadcast to a common
shape, if any $rho_avg$ is not finite or not strictly positive, if any
`zeta` is not finite or is below 1, or if $f$ is not finite or does not
satisfy 0 < f < 1.

Returns
-------
A float array of shape (2, N) holding the ambient and the strand-interior mass density, both in 1e-12 kg m^-3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cross_section_structure(rho_avg: "ArrayLike", zeta: "ArrayLike",
                            f: float) -> "np.ndarray":
    '''Split the area-weighted density into its two cross-sectional components.

    Parameters
    ----------
    rho_avg : array_like
        Area-weighted mean mass density over the bundle cross-section, in
        units of 1e-12 kg m^-3. Finite and strictly positive.
    zeta : array_like
        Density contrast rho_i / rho_e, broadcastable against ``rho_avg``.
        Finite and not less than 1.
    f : float
        Fraction of the cross-sectional area occupied by strand interior.
        Finite, with 0 < f < 1.

    Returns
    -------
    np.ndarray
        Shape (2, N) float array in units of 1e-12 kg m^-3. Row 0 is the
        ambient density rho_e, row 1 the strand-interior density rho_i.

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


def _broadcast_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        shape = np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    return np.broadcast_to(a, shape).astype(float), np.broadcast_to(b, shape).astype(float)


def _oracle_cross_section_structure(rho_avg: "ArrayLike", zeta: "ArrayLike",
                                    f: float) -> "np.ndarray":
    rho, zt = _broadcast_pair(rho_avg, zeta)
    if not np.all(np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError("rho_avg must be finite and strictly positive")
    if not np.all(np.isfinite(zt)) or np.any(zt < 1.0):
        raise ValueError("zeta must be finite and not less than 1")
    ff = float(f)
    if not np.isfinite(ff) or not (0.0 < ff < 1.0):
        raise ValueError("f must be finite and satisfy 0 < f < 1")

    weight = 1.0 - ff + ff * zt
    rho_e = rho / weight
    rho_i = zt * rho_e
    return np.asarray([rho_e, rho_i], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Differential test cases for this step.'''
    # Each call builds its own input arrays, so an implementation that works
    # in place on its arguments cannot change what the other call receives.
    benchmark = (
        "import numpy as np\n"
        "def _args():\n"
        "    rho = np.array([3.5125060, 2.5, 1.2, 0.42])\n"
        "    zeta = np.array([3.6, 3.3, 3.0, 2.93])\n"
        "    return (rho, zeta, 0.16)\n"
    )
    boundary = (
        "import numpy as np\n"
        "def _args():\n"
        "    return (np.array([1.0]), np.array([1.0]), 0.5)\n"
    )
    edge = (
        "import numpy as np\n"
        "def _code(fn):\n"
        "    try:\n"
        "        fn(np.array([1.0, 2.0]), np.array([2.0, 2.0, 2.0]), 0.2)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        {"setup": benchmark,
         "call": "cross_section_structure(*_args())",
         "gold_call": "_oracle_cross_section_structure(*_args())"},
        {"setup": boundary,
         "call": "cross_section_structure(*_args())",
         "gold_call": "_oracle_cross_section_structure(*_args())"},
        {"setup": edge,
         "call": "_code(cross_section_structure)",
         "gold_call": "_code(_oracle_cross_section_structure)"},
    ]
