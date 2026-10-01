"""
Step 09: locate the lowest height at which heating balances losses.

Contract
--------
Given a heating rate and a loss rate sampled on a common uniform height grid,
find the LOWEST height at which their difference changes sign, that is the
first crossing of heating and losses as one moves up from the base.

## Conventions

-----------

Heights in Mm on a UNIFORM grid; the two rate arrays share that grid and must

be in the same units as each other. Form the residual as heating minus losses.

Scan upwards for the first pair of adjacent nodes whose residual values have

strictly opposite signs, and let `$j$` be the index of the lower node of that

pair. The interpolation stencil is the four consecutive nodes

``i0 .. i0 + 3`` with `$i0 = j - 1$`, so that the bracketing pair is the middle

two of the four; where that stencil would leave the grid, clamp ``i0`` into

the range ``0 .. N - 4`` (a crossing in the first interval uses

`$i0 = 0$`, one in the last interval uses `$i0 = N - 4$`). Build the cubic

Lagrange interpolant of the residual through those four nodes in the local

coordinate t, with t = 0 at node ``i0`` and t increasing by one per grid

interval, so the bracketing pair sits at t = j - i0 and t = j - i0 + 1.

Bisect that interpolant over this bracketing unit interval for 200 halvings,

keeping the half whose endpoints straddle a sign change (a value of exactly

zero counts as non-negative), and return the midpoint of the final bracket,

converted back to a height in Mm.



Return a Python float.



Validation

----------

Raise ValueError if `$z$` is not a one-dimensional, strictly increasing,

uniformly spaced grid of at least four finite points; if ``heating`` or

``losses`$is not a finite one-dimensional array of the same length as$$z$`;

or if the residual has no sign change anywhere inside the grid.

Returns
-------
A Python float: the lowest height in Mm at which the heating rate equals the loss rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def first_balance_height(z: "ArrayLike", heating: "ArrayLike",
                         losses: "ArrayLike") -> float:
    '''Lowest height at which the heating rate equals the loss rate.

    Parameters
    ----------
    z : array_like
        One-dimensional uniform height grid in Mm, strictly increasing, at
        least four points.
    heating : array_like
        Volumetric heating rate on that grid, finite, one value per point.
    losses : array_like
        Volumetric loss rate on that grid in the same units as ``heating``,
        finite, one value per point.

    Returns
    -------
    float
        The lowest height in Mm at which heating and losses are equal.

    Raises
    ------
    ValueError
        On a malformed grid, a malformed rate array, or a residual that never
        changes sign inside the grid.
    '''
    return z_star  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_profile(name, arr, n):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1 or a.size != n:
        raise ValueError("%s must have one value per grid point" % name)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    return a


def _cubic_through(y, t):
    return (-y[0] * (t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0
            + y[1] * t * (t - 2.0) * (t - 3.0) / 2.0
            - y[2] * t * (t - 1.0) * (t - 3.0) / 2.0
            + y[3] * t * (t - 1.0) * (t - 2.0) / 6.0)


def _oracle_first_balance_height(z: "ArrayLike", heating: "ArrayLike",
                                 losses: "ArrayLike") -> float:
    z, step = _uniform_grid(z)
    n = z.size
    heat = _finite_profile("heating", heating, n)
    loss = _finite_profile("losses", losses, n)

    residual = heat - loss
    crossings = np.nonzero(np.sign(residual[:-1]) * np.sign(residual[1:]) < 0.0)[0]
    if crossings.size == 0:
        raise ValueError("the residual never changes sign inside the grid")
    lower = int(crossings[0])
    i0 = int(np.clip(lower - 1, 0, n - 4))
    node_values = residual[i0:i0 + 4]
    t_lo = float(lower - i0)

    a, b = t_lo, t_lo + 1.0
    fa = _cubic_through(node_values, a)
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = _cubic_through(node_values, mid)
        if (fa < 0.0) != (fm < 0.0):
            b = mid
        else:
            a, fa = mid, fm
    t_star = 0.5 * (a + b)
    return float(z[i0] + t_star * step)

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
        "    z = np.linspace(0.0, 120.0, 241)\n"
        "    return (z, 3.0 * np.exp(-z / 300.0), 47.0 * np.exp(-z / 21.0))\n"
    )
    boundary = (
        "import numpy as np\n"
        "def _args():\n"
        "    return (np.linspace(0.0, 3.0, 4), np.array([0.0, 0.9, 2.0, 3.0]),\n"
        "            np.array([1.0, 1.0, 1.0, 1.0]))\n"
    )
    # Regression for the stencil: a strongly curved residual crossing in an
    # interior interval. Starting the four-node stencil at the lower bracketing
    # node instead of one node below it moves the answer by about 5e-3.
    stencil = (
        "import numpy as np\n"
        "def _args():\n"
        "    z = np.linspace(0.0, 4.5, 10)\n"
        "    return (z, np.exp(z), 20.0 - 2.0 * z)\n"
    )
    edge = (
        "import numpy as np\n"
        "def _code(fn):\n"
        "    z = np.linspace(0.0, 3.0, 4)\n"
        "    try:\n"
        "        fn(z, np.ones(4), np.zeros(4))\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        {"setup": benchmark,
         "call": "first_balance_height(*_args())",
         "gold_call": "_oracle_first_balance_height(*_args())"},
        {"setup": boundary,
         "call": "first_balance_height(*_args())",
         "gold_call": "_oracle_first_balance_height(*_args())"},
        {"setup": edge,
         "call": "_code(first_balance_height)",
         "gold_call": "_code(_oracle_first_balance_height)"},
        {"setup": stencil,
         "call": "first_balance_height(*_args())",
         "gold_call": "_oracle_first_balance_height(*_args())"},
    ]
