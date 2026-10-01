"""
Integrate the shifted insertion weights at every order through p, then remove vacuum components from both the signed and positive diagram-weight series by formal power-series division.

Color-preserving linked-cluster combinatorics separates connected spin-response diagrams from vacuum components.

$C_n=X_n-\sum_{k=1}^n Z_k C_{n-k}$ and $C_{{\rm abs},n}=X_{{\rm abs},n}-\sum_{k=1}^n Z_{{\rm abs},k}C_{{\rm abs},n-k}$, with $Z_0=Z_{{\rm abs},0}=1$. Signed moments include $(-1)^n$; positive moments integrate each separate pairing magnitude.

Returns
-------
complex ndarray (6,p+1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def connected_coefficient_series(
    couplings: np.ndarray,
    hopping: np.ndarray,
    beta: float,
    tau: float,
    sites: np.ndarray,
    component: int,
    order: int,
    quadrature: int,
) -> np.ndarray:
    """Integrate signed and pairing-absolute connected coefficients.

    Parameters
    ----------
    couplings, hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal symmetric couplings and skew hopping,
        identical shape, even L >= 2, as in map_shifted_vertices.
    beta : float
        Finite positive inverse temperature.
    tau : float
        Finite external time in [0, beta].
    sites : ndarray, shape (2,)
        Integer-valued sites j,l in [0,L).
    component : int
        Color a in {0,1,2} of both external spins.
    order : int
        Highest insertion order p in {0,1,2,3}.
    quadrature : int
        2 to 24 nodes per variable of the folded ordered_time_rule.

    Returns
    -------
    series : complex ndarray, shape (6, p+1)
        Rows [X, Z, C, X_abs, Z_abs, C_abs]. X_n and Z_n are
        normalized-H0 numerator/vacuum Dyson coefficients of xi**n
        for H(xi)=H0+xi*(H_Maj-H0). C=X/Z as a formal power series.
        X_abs,n and Z_abs,n integrate individual pairing magnitudes
        before any vertex/pairing sums; use the exact row decomposition
        from map_shifted_vertices and no Dyson minus sign. C_abs is
        the formal quotient X_abs/Z_abs. This removes vacuum components
        while preserving magnitudes of separate connected diagrams.
        A component must include both external spin blocks because
        each interaction vertex has even population of every color.
        Z_0=Z_abs,0=1. Apply the supplied quadrature to every order,
        including lower orders entering the quotient. No projection or
        reference refitting is applied. Inputs are unchanged.

    Raises
    ------
    ValueError
        For invalid tensor, temperature, observable, order or quadrature
        under the contracts above and of the producing functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_connected_coefficient_series(
    couplings: np.ndarray,
    hopping: np.ndarray,
    beta: float,
    tau: float,
    sites: np.ndarray,
    component: int,
    order: int,
    quadrature: int,
) -> np.ndarray:
    """Integrate signed and pairing-absolute connected coefficients.

    Parameters
    ----------
    couplings, hopping : real ndarray, shape (3, L, L)
        Finite zero-diagonal symmetric couplings and skew hopping,
        identical shape, even L >= 2, as in map_shifted_vertices.
    beta : float
        Finite positive inverse temperature.
    tau : float
        Finite external time in [0, beta].
    sites : ndarray, shape (2,)
        Integer-valued sites j,l in [0,L).
    component : int
        Color a in {0,1,2} of both external spins.
    order : int
        Highest insertion order p in {0,1,2,3}.
    quadrature : int
        2 to 24 nodes per variable of the folded ordered_time_rule.

    Returns
    -------
    series : complex ndarray, shape (6, p+1)
        Rows [X, Z, C, X_abs, Z_abs, C_abs]. X_n and Z_n are
        normalized-H0 numerator/vacuum Dyson coefficients of xi**n
        for H(xi)=H0+xi*(H_Maj-H0). C=X/Z as a formal power series.
        X_abs,n and Z_abs,n integrate individual pairing magnitudes
        before any vertex/pairing sums; use the exact row decomposition
        from map_shifted_vertices and no Dyson minus sign. C_abs is
        the formal quotient X_abs/Z_abs. This removes vacuum components
        while preserving magnitudes of separate connected diagrams.
        A component must include both external spin blocks because
        each interaction vertex has even population of every color.
        Z_0=Z_abs,0=1. Apply the supplied quadrature to every order,
        including lower orders entering the quotient. No projection or
        reference refitting is applied. Inputs are unchanged.

    Raises
    ------
    ValueError
        For invalid tensor, temperature, observable, order or quadrature
        under the contracts above and of the producing functions.
    """
    j = _tensor(couplings, "couplings", False)
    a = _tensor(hopping, "hopping", True)
    if j.shape != a.shape:
        raise ValueError("couplings and hopping must have identical shapes")
    beta = _temperature(beta)
    pair, color, tau = _observable(sites, component, tau, beta, j.shape[1])
    order = _integer(order, "order", 0, 3)
    _integer(quadrature, "quadrature", 2, 24)
    vertices = _oracle_map_shifted_vertices(j, a)
    result = np.zeros((6, order + 1), complex)
    for power in range(order + 1):
        rule = _oracle_ordered_time_rule(beta, tau, power, quadrature)
        moments = rule[:, -1] @ _oracle_shifted_wick_weights(
            a, beta, vertices, rule[:, :-1], pair, color, tau
        )
        result[:2, power] = (-1) ** power * moments[:2]
        result[3:5, power] = moments[2:]
        for row in (0, 3):
            result[row + 2, power] = result[row, power] - sum(
                result[row + 1, k] * result[row + 2, power - k]
                for k in range(1, power + 1)
            )
    return result

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
                "connected_coefficient_series(j, h, 2.4, 0.83, sites, "
                "2, 3, 2)"
            ),
            "gold_call": (
                "_oracle_connected_coefficient_series(j, h, 2.4, "
                "0.83, sites, 2, 3, 2)"
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
                "with_preservation(\n    connected_coefficient_series, "
                "j, h, 2.1, 0.68, sites, 0, 2, 3\n)"
            ),
            "gold_call": (
                "with_preservation(\n    "
                "_oracle_connected_coefficient_series, j, h, 2.1, "
                "0.68, sites, 0, 2, 3\n)"
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
                "connected_coefficient_series(j, h, 2.4, 0.83, sites, "
                "2, 0, 2)"
            ),
            "gold_call": (
                "_oracle_connected_coefficient_series(j, h, 2.4, "
                "0.83, sites, 2, 0, 2)"
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
                "expect_value_error(\n    "
                "connected_coefficient_series, j, h, 2.4, 0.83, "
                "sites, 2, 4, 2\n)"
            ),
            "gold_call": (
                "expect_value_error(\n    "
                "_oracle_connected_coefficient_series, j, h, 2.4, "
                "0.83, sites, 2, 4, 2\n)"
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
""",
            "call": (
                "connected_coefficient_series(j, h, 2.4, 0.83, sites, "
                "2, 3, 2)"
            ),
            "gold_call": (
                "_oracle_connected_coefficient_series(j, h, 2.4, "
                "0.83, sites, 2, 3, 2)"
            ),
            "tol": 1e-08,
        },
    ]
