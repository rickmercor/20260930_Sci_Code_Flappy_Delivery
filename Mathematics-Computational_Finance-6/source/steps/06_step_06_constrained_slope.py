"""
Selects the slope actually used for the inverse-Gaussian draw, returning the unconstrained projection slope when it satisfies the positivity requirement and the constrained fallback otherwise.

Constrained slope selection.

The projection step produces an unconstrained slope that matches the conditional first two moments of the variance increment, but that slope is admissible only when it keeps the updated variance non-negative for every realization of the draw. Three conditions encode this requirement: the slope must be strictly positive, it must not exceed the upper bound implied by the positivity constraint, and the constraint evaluated at zero slope must be non-negative. When all three hold, the unconstrained slope is used unchanged; otherwise the slope is replaced by the constrained fallback that saturates the positivity constraint, which is the value the inverse-Gaussian draw is actually parametrized with. The selection is inclusive at both boundaries, so a slope exactly equal to its upper bound and a constraint value exactly zero both count as feasible, and no numerical tolerance is applied.

Returns
-------
beta_tilde : float -- the slope used for the inverse-Gaussian draw: the unconstrained slope when it is feasible, otherwise the constrained fallback (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def constrained_slope(nu: float, omega: np.ndarray, alpha: float, c: float, beta: float, betaL: float, C0: float) -> float:
    """Select the slope actually used for the inverse-Gaussian draw.

    Parameters
    ----------
    nu : float
        Volatility-of-variance coefficient (must be > 0).
    omega : (N,) float array
        Lift weights of the N factors.
    alpha : float
        Conditional mean of the variance increment (must be > 0).
    c : float
        Intercept of the positivity constraint (must be non-zero).
    beta : float
        Unconstrained projection slope.
    betaL : float
        Upper admissible slope implied by the positivity requirement.
    C0 : float
        Value of the positivity constraint evaluated at zero slope.

    Returns
    -------
    beta_tilde : float
        The slope used for the draw: the unconstrained slope when it is
        feasible, otherwise the constrained fallback.

    Raises
    ------
    ValueError
        If alpha is not > 0, if c is zero, if nu is not > 0, or if omega is
        not a one-dimensional array with at least one entry.
    """
    return beta_tilde

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math


def _real_scalar_05(value) -> bool:
    """True when value is a real scalar (not a bool, not an array)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _oracle_constrained_slope(nu: float, omega: np.ndarray, alpha: float, c: float, beta: float, betaL: float, C0: float) -> float:
    """Reference implementation of constrained_slope."""
    if not _real_scalar_05(alpha) or not (float(alpha) > 0.0):
        raise ValueError("alpha must be a real number > 0")
    if not _real_scalar_05(c) or float(c) == 0.0:
        raise ValueError("c must be a real number != 0")
    if not _real_scalar_05(nu) or not (float(nu) > 0.0):
        raise ValueError("nu must be a real number > 0")
    om = np.asarray(omega, dtype=float)
    if om.ndim != 1 or om.size < 1:
        raise ValueError("omega must be a one-dimensional array with at least one entry")
    for name, value in (("beta", beta), ("betaL", betaL), ("C0", C0)):
        if not _real_scalar_05(value):
            raise ValueError(f"{name} must be a real number")
    alpha = float(alpha)
    c = float(c)
    nu = float(nu)
    beta = float(beta)
    betaL = float(betaL)
    C0 = float(C0)
    feasible = (beta > 0.0) and (beta <= betaL) and (C0 >= 0.0)
    if feasible:
        return beta
    return nu * alpha * float(np.sum(om)) / c

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 4 cases: a feasible unconstrained slope, the pinned constrained
    fallback, the inclusive feasibility boundary, a slope above its upper
    bound, and an invalid-input edge.
    """
    return [
        {
            # feasible: beta inside (0, betaL] with C0 >= 0 keeps the unconstrained slope
            "setup": "import numpy as np\nomega = np.array([0.13979052887176874, 0.17801924180267542, 0.2267024147327589, 0.2886990435709822, 0.36764997786658365])",
            "call": "float(constrained_slope(0.1, omega, 0.02, 0.125, 0.01, 0.04, 0.01))",
            "gold_call": "float(_oracle_constrained_slope(0.1, omega, 0.02, 0.125, 0.01, 0.04, 0.01))",
        },
        {
            # pinned benchmark step: the unconstrained slope fails the positivity constraint
            "setup": "import numpy as np\nomega = np.array([0.13979052887176874, 0.17801924180267542, 0.2267024147327589, 0.2886990435709822, 0.36764997786658365])",
            "call": "float(constrained_slope(0.1, omega, 0.021119426636698066, 0.12534218765222555, 0.016401825054830583, 0.037685562928788524, -0.029283898615811155))",
            "gold_call": "float(_oracle_constrained_slope(0.1, omega, 0.021119426636698066, 0.12534218765222555, 0.016401825054830583, 0.037685562928788524, -0.029283898615811155))",
        },
        {
            # boundary: beta exactly at its upper bound and C0 exactly zero are both feasible
            "setup": "import numpy as np\nomega = np.array([0.13979052887176874, 0.17801924180267542, 0.2267024147327589, 0.2886990435709822, 0.36764997786658365])",
            "call": "float(constrained_slope(0.1, omega, 0.02, 0.125, 0.04, 0.04, 0.0))",
            "gold_call": "float(_oracle_constrained_slope(0.1, omega, 0.02, 0.125, 0.04, 0.04, 0.0))",
        },
        {
            # boundary: a slope above its upper bound falls back to the constrained value
            "setup": "import numpy as np\nomega = np.array([0.13979052887176874, 0.17801924180267542, 0.2267024147327589, 0.2886990435709822, 0.36764997786658365])",
            "call": "float(constrained_slope(0.1, omega, 0.02, 0.125, 0.05, 0.04, 0.01))",
            "gold_call": "float(_oracle_constrained_slope(0.1, omega, 0.02, 0.125, 0.05, 0.04, 0.01))",
        },
        {
            # edge: a zero constraint intercept is invalid
            "setup": """import numpy as np
omega = np.array([0.13979052887176874, 0.17801924180267542, 0.2267024147327589, 0.2886990435709822, 0.36764997786658365])
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: constrained_slope(0.1, omega, 0.02, 0.0, 0.01, 0.04, 0.01))",
            "gold_call": "_guard(lambda: _oracle_constrained_slope(0.1, omega, 0.02, 0.0, 0.01, 0.04, 0.01))",
        },
    ]
