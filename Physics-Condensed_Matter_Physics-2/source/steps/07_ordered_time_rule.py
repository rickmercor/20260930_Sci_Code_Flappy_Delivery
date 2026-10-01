"""
Build the prescribed ordered-simplex Gauss rule by folding times into half a thermal period, resolving both external-time boundaries and pairwise half-period crossings. Return ordered nodes and weights.

Absolute propagator weights can require additional smooth integration panels inside the imaginary-time simplex.

$h=\beta/2$, $f=\tau\bmod h$, and $\mathbf t=\operatorname{sort}_{\downarrow}(\mathbf u+h\mathbf b)$ for $\mathbf b\in\{0,1\}^n$ and descending folded times $h\geq u_1\geq\cdots\geq u_n\geq0$, split at $f$. The total quadrature weight is $\beta^n/n!$.

Returns
-------
real ndarray (R,n+1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def ordered_time_rule(
    beta: float, tau: float, order: int, quadrature: int
) -> np.ndarray:
    """Construct a folded-half-period ordered-simplex Gauss rule.

    Parameters
    ----------
    beta : float
        Finite positive interval length.
    tau : float
        Finite external time in [0, beta].
    order : int
        Number n of insertion times, 0 <= n <= 3.
    quadrature : int
        Gauss-Legendre nodes per variable, 2 <= q <= 24.

    Returns
    -------
    rule : real ndarray, shape (R, n+1)
        Rows [t1, ..., tn, weight], with descending physical times in
        [0, beta]. For n=0 return [[1.0]]. Otherwise set h=beta/2 and
        f=tau modulo h. Enumerate bit tuples in {0,1}**n lexicographically.
        For each tuple enumerate k=0,...,n folded variables above f,
        skipping zero-volume regions. Enumerate the ascending unit-
        interval Gauss nodes in lexicographic tensor-product order.
        Generate descending folded variables u on [f,h] for the first
        k positions and [0,f] for the rest. In each block start at its
        upper bound, map x to lower+(upper-lower)*x, multiply the weight
        by (upper-lower) times its unit Gauss weight, and replace upper
        with the new point. Set physical t to the descending sort of
        u+h*bits. No extra factorial. Total weight is beta**n/n!.
        These panels resolve pairwise half-period crossings and the
        external time. They do not guarantee exact integration for
        arbitrary hopping; convergence remains a numerical question.

    Raises
    ------
    ValueError
        For nonfinite beta/tau, nonpositive beta, tau outside [0,beta],
        or noninteger/out-of-range order or quadrature.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_ordered_time_rule(
    beta: float, tau: float, order: int, quadrature: int
) -> np.ndarray:
    """Construct a folded-half-period ordered-simplex Gauss rule.

    Parameters
    ----------
    beta : float
        Finite positive interval length.
    tau : float
        Finite external time in [0, beta].
    order : int
        Number n of insertion times, 0 <= n <= 3.
    quadrature : int
        Gauss-Legendre nodes per variable, 2 <= q <= 24.

    Returns
    -------
    rule : real ndarray, shape (R, n+1)
        Rows [t1, ..., tn, weight], with descending physical times in
        [0, beta]. For n=0 return [[1.0]]. Otherwise set h=beta/2 and
        f=tau modulo h. Enumerate bit tuples in {0,1}**n lexicographically.
        For each tuple enumerate k=0,...,n folded variables above f,
        skipping zero-volume regions. Enumerate the ascending unit-
        interval Gauss nodes in lexicographic tensor-product order.
        Generate descending folded variables u on [f,h] for the first
        k positions and [0,f] for the rest. In each block start at its
        upper bound, map x to lower+(upper-lower)*x, multiply the weight
        by (upper-lower) times its unit Gauss weight, and replace upper
        with the new point. Set physical t to the descending sort of
        u+h*bits. No extra factorial. Total weight is beta**n/n!.
        These panels resolve pairwise half-period crossings and the
        external time. They do not guarantee exact integration for
        arbitrary hopping; convergence remains a numerical question.

    Raises
    ------
    ValueError
        For nonfinite beta/tau, nonpositive beta, tau outside [0,beta],
        or noninteger/out-of-range order or quadrature.
    """
    beta = _temperature(beta)
    time = _numeric(tau, "tau", True)
    if time.ndim != 0 or not 0 <= time <= beta:
        raise ValueError("tau must lie in [0,beta]")
    order = _integer(order, "order", 0, 3)
    q = _integer(quadrature, "quadrature", 2, 24)
    if order == 0:
        return np.ones((1, 1))
    half = beta / 2
    folded = float(time) % half
    nodes, weights = leggauss(q)
    nodes, weights = (nodes + 1) / 2, weights / 2
    result = []
    for bits in product((0, 1), repeat=order):
        for above in range(order + 1):
            bounds = [(folded, half)] * above + [(0.0, folded)] * (
                order - above
            )
            if any(lo == hi for lo, hi in bounds):
                continue
            for picks in product(range(q), repeat=order):
                times = []
                weight = 1.0
                upper = half
                for k, pick in enumerate(picks):
                    lower, initial_upper = bounds[k]
                    if k == 0 or k == above:
                        upper = initial_upper
                    width = upper - lower
                    weight *= width * weights[pick]
                    upper = lower + width * nodes[pick]
                    times.append(upper + half * bits[k])
                result.append(sorted(times, reverse=True) + [weight])
    return np.asarray(result, float)

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

import numpy as np
""",
            "call": ("ordered_time_rule(2.4, 0.83, 3, 2)"),
            "gold_call": ("_oracle_ordered_time_rule(2.4, 0.83, 3, 2)"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np
""",
            "call": ("ordered_time_rule(1.7, 0.0, 1, 4)"),
            "gold_call": ("_oracle_ordered_time_rule(1.7, 0.0, 1, 4)"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np
""",
            "call": ("ordered_time_rule(1.9, 1.9, 2, 3)"),
            "gold_call": ("_oracle_ordered_time_rule(1.9, 1.9, 2, 3)"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np
""",
            "call": ("ordered_time_rule(2.3, 0.4, 0, 2)"),
            "gold_call": ("_oracle_ordered_time_rule(2.3, 0.4, 0, 2)"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np
""",
            "call": ("ordered_time_rule(2.4, 1.2, 3, 3)"),
            "gold_call": ("_oracle_ordered_time_rule(2.4, 1.2, 3, 3)"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "expect_value_error(ordered_time_rule, 2.4, 0.83, 3, " "2.5)"
            ),
            "gold_call": (
                "expect_value_error(_oracle_ordered_time_rule, 2.4, "
                "0.83, 3, 2.5)"
            ),
            "tol": 1e-08,
        },
    ]
