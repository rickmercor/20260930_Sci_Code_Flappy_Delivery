"""
Compose the earlier steps to return the real average phase of connected diagrams at the selected order, expressed as a percentage.

The ratio of signed to absolute diagram weight measures cancellation at a selected perturbative order.

$\mathcal S_p=100\operatorname{Re}C_p/C_{{\rm abs},p}$, with $C_{{\rm abs},p}>10^{-12}$. For real weights this is the average sign.

Returns
-------
float: real average phase in percent
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def connected_average_sign(
    couplings: np.ndarray,
    hopping: np.ndarray,
    beta: float,
    tau: float,
    sites: np.ndarray,
    component: int,
    order: int,
    quadrature: int,
) -> float:
    """Return the real average phase of connected order-p diagrams.

    Parameters
    ----------
    couplings, hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal symmetric couplings and skew hopping,
        identical shape, even L >= 2. Use map_shifted_vertices without
        regrouping its monomial rows or changing the reference.
    beta : float
        Finite positive inverse temperature.
    tau : float
        Finite external time in [0,beta].
    sites : ndarray, shape (2,)
        Integer-valued sites in [0,L) for the two external spins.
    component : int
        Spin color in {0,1,2}.
    order : int
        Selected insertion order p in {0,1,2,3}; not a truncated sum.
    quadrature : int
        2 to 24 nodes per variable of the folded ordered_time_rule.

    Returns
    -------
    percentage : float
        100*Re(C_p)/C_abs,p from connected_coefficient_series with the
        supplied quadrature. C_abs,p sums magnitudes of individual
        connected pairings before summation and integration. For real
        diagram weights this is the average sign in percent; otherwise
        it is the real average phase in percent. Inputs are unchanged.

    Raises
    ------
    ValueError
        For invalid inputs under connected_coefficient_series, a
        nonfinite coefficient/result, or C_abs,p <= 1e-12.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_connected_average_sign(
    couplings: np.ndarray,
    hopping: np.ndarray,
    beta: float,
    tau: float,
    sites: np.ndarray,
    component: int,
    order: int,
    quadrature: int,
) -> float:
    """Return the real average phase of connected order-p diagrams.

    Parameters
    ----------
    couplings, hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal symmetric couplings and skew hopping,
        identical shape, even L >= 2. Use map_shifted_vertices without
        regrouping its monomial rows or changing the reference.
    beta : float
        Finite positive inverse temperature.
    tau : float
        Finite external time in [0,beta].
    sites : ndarray, shape (2,)
        Integer-valued sites in [0,L) for the two external spins.
    component : int
        Spin color in {0,1,2}.
    order : int
        Selected insertion order p in {0,1,2,3}; not a truncated sum.
    quadrature : int
        2 to 24 nodes per variable of the folded ordered_time_rule.

    Returns
    -------
    percentage : float
        100*Re(C_p)/C_abs,p from connected_coefficient_series with the
        supplied quadrature. C_abs,p sums magnitudes of individual
        connected pairings before summation and integration. For real
        diagram weights this is the average sign in percent; otherwise
        it is the real average phase in percent. Inputs are unchanged.

    Raises
    ------
    ValueError
        For invalid inputs under connected_coefficient_series, a
        nonfinite coefficient/result, or C_abs,p <= 1e-12.
    """
    series = _oracle_connected_coefficient_series(
        couplings, hopping, beta, tau, sites, component, order, quadrature
    )
    denominator = series[5, -1].real
    if not np.all(np.isfinite(series)) or denominator <= 1e-12:
        raise ValueError("connected absolute weight must exceed 1e-12")
    result = 100 * series[2, -1].real / denominator
    if not np.isfinite(result):
        raise ValueError("average phase must be finite")
    return float(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])
""",
            "call": (
                "connected_average_sign(j, h, 2.4, 0.83, sites, 2, 3, " "2)"
            ),
            "gold_call": (
                "_oracle_connected_average_sign(j, h, 2.4, 0.83, "
                "sites, 2, 3, 2)"
            ),
            "tol": 1e-07,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 2, 2))
h = np.zeros_like(j)
for color, value in enumerate([0.7, 1.2, -0.3]):
    j[color, 0, 1] = j[color, 1, 0] = value
for color, value in enumerate([0.19, -0.27, 0.13]):
    h[color, 0, 1] = value
    h[color, 1, 0] = -value
sites = np.array([0, 1])


def with_preservation(fn, *args):
    before = [
        arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args
    ]
    result = fn(*args)
    preserved = all(np.array_equal(arg, old) for arg, old in zip(args, before))
    return np.r_[np.asarray(result).ravel(), float(preserved)]
""",
            "call": (
                "with_preservation(connected_average_sign, j, h, 2.1, "
                "0.68, sites, 0, 2, 3)"
            ),
            "gold_call": (
                "with_preservation(\n    "
                "_oracle_connected_average_sign, j, h, 2.1, 0.68, "
                "sites, 0, 2, 3\n)"
            ),
            "tol": 1e-07,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])
h[:] = 0
""",
            "call": (
                "connected_average_sign(j, h, 2.4, 0.83, sites, 2, 0, " "2)"
            ),
            "gold_call": (
                "_oracle_connected_average_sign(j, h, 2.4, 0.83, "
                "sites, 2, 0, 2)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "expect_value_error(connected_average_sign, j, h, "
                "2.4, 0.83, sites, 2, 4, 2)"
            ),
            "gold_call": (
                "expect_value_error(\n    "
                "_oracle_connected_average_sign, j, h, 2.4, 0.83, "
                "sites, 2, 4, 2\n)"
            ),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

j = np.zeros((3, 4, 4))
h = np.zeros_like(j)
for color, left, right, value in (
    (0, 0, 1, 0.9),
    (1, 1, 2, 1.1),
    (2, 2, 3, 0.7),
):
    j[color, left, right] = j[color, right, left] = value
for color, left, right, value in (
    (0, 0, 1, 0.23),
    (1, 0, 2, -0.31),
    (2, 0, 3, 0.19),
):
    h[color, left, right] = value
    h[color, right, left] = -value
sites = np.array([0, 0])
j[:] = 0
h[:] = 0


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "expect_value_error(connected_average_sign, j, h, "
                "2.4, 0.83, sites, 2, 1, 2)"
            ),
            "gold_call": (
                "expect_value_error(\n    "
                "_oracle_connected_average_sign, j, h, 2.4, 0.83, "
                "sites, 2, 1, 2\n)"
            ),
            "tol": 1e-08,
        },
    ]
