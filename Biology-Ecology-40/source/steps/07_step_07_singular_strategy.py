"""
Given the supply sequence, the burn-in and a bracket (lower, upper) in phi at which the selection gradient of the preceding step is respectively positive and negative, locate the singular strategy phi* by the Illinois method in ln phi, stopping when the bracket or the step in ln phi falls below the tolerance, and return phi*, the gradients at the two ends of the initial bracket, the gradient and the curvature at phi*, the slope of the gradient with respect to ln phi at phi*, taken as the central difference of the gradient between phi* exp(-0.02) and phi* exp(0.02) divided by 0.04, and the number of gradient evaluations made inside the bracket.

Under fluctuating supply the selection gradient of the secondary allocation changes sign once. At small phi the long lag a strain pays whenever its primary resource runs out first costs more than the slightly lower growth rate and the short lag it pays when its secondary resource runs out, so larger phi is favoured; at large phi the balance reverses. The trait therefore evolves towards the singular strategy phi* at which the gradient vanishes, and because the gradient falls through zero there, phi* is convergence stable: a resident on either side is invaded by mutants closer to it. Whether it is also uninvadable is decided by the sign of the curvature of the invasion growth rate at phi*.

With a fixed supply sequence the gradient is a deterministic, continuous function of phi, and a root in ln phi can be bracketed and refined. The Illinois variant of regula falsi is well suited: it keeps a bracket at every iteration, which the noise-free but slightly irregular gradient requires, and it avoids the stagnation of plain regula falsi by halving the retained end's function value whenever the same end is kept twice. Each evaluation of the gradient costs a full pass over the supply sequence, so an iteration that converges superlinearly matters.

Returns
-------
dict holding the float phi_star, the float gradient_lower, the float gradient_upper, the float gradient_star, the float curvature, the float convergence_slope and the int evaluations.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def singular_strategy(supply: np.ndarray, burn_in: int, lower: float = 0.01, upper: float = 0.9,
                      tolerance: float = 1e-9) -> dict:
    """Singular strategy of the secondary allocation by bracketed root finding in ln phi.

    Parameters
    ----------
    supply : np.ndarray
        Per-cycle supplies of the two resources, shape (n_cycles, 2).
    burn_in : int
        Cycles discarded before averaging.
    lower : float
        Lower bracket end, where the gradient is positive.
    upper : float
        Upper bracket end, where the gradient is negative.
    tolerance : float
        Convergence tolerance in ln phi.

    Returns
    -------
    dict
        Under the keys phi_star, gradient_lower, gradient_upper, gradient_star, curvature,
        convergence_slope and evaluations.

    Raises
    ------
    ValueError
        When the bracket is invalid, the gradient does not change sign across it, or the
        tolerance is not positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_singular_strategy(supply: np.ndarray, burn_in: int, lower: float = 0.01, upper: float = 0.9,
                              tolerance: float = 1e-9) -> dict:
    """Reference implementation."""
    lower = float(lower)
    upper = float(upper)
    tolerance = float(tolerance)
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower <= 0.0 or upper <= lower \
            or upper > math.exp(-0.07):
        raise ValueError("the bracket must satisfy 0 < lower < upper <= exp(-0.07)")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    def _gradient(u):
        return _oracle_selection_gradient(math.exp(u), supply, burn_in)["gradient"]  # noqa: F821

    a, b = math.log(lower), math.log(upper)
    ga, gb = _gradient(a), _gradient(b)
    g_lower, g_upper = ga, gb
    if not (ga > 0.0 > gb):
        raise ValueError("the selection gradient must be positive at lower and negative at upper")
    side = 0
    x = a
    evaluations = 0
    for _ in range(100):
        x_new = b - gb * (b - a) / (gb - ga)
        gx = _gradient(x_new)
        evaluations += 1
        moved = abs(x_new - x)
        x = x_new
        if gx == 0.0:
            a = b = x
            break
        if gx > 0.0:
            a, ga = x, gx
            if side == 1:
                gb *= 0.5
            side = 1
        else:
            b, gb = x, gx
            if side == -1:
                ga *= 0.5
            side = -1
        if b - a < tolerance or moved < tolerance:
            break
    at_star = _oracle_selection_gradient(math.exp(x), supply, burn_in)  # noqa: F821
    left = _oracle_selection_gradient(math.exp(x - 0.02), supply, burn_in)["gradient"]  # noqa: F821
    right = _oracle_selection_gradient(math.exp(x + 0.02), supply, burn_in)["gradient"]  # noqa: F821
    return {"phi_star": math.exp(x),
            "gradient_lower": float(g_lower),
            "gradient_upper": float(g_upper),
            "gradient_star": float(at_star["gradient"]),
            "curvature": float(at_star["curvature"]),
            "convergence_slope": float((right - left) / 0.04),
            "evaluations": int(evaluations)}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the declared test cases for this step."""
    SETUP = """import math
import numpy as np

def isolated(fn, *args, **kwargs):
    return fn(*(x.copy() if isinstance(x, np.ndarray) else x for x in args),
              **{k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in kwargs.items()})

def num(v, k=6):
    return float(round(float(v), k))

def supply_sequence(n, alpha, seed):
    rng = np.random.default_rng(seed)
    g = rng.gamma(alpha, 1.0, size=(n, 2))
    g = np.maximum(g, 1e-300)
    return g / g.sum(axis=1, keepdims=True)

def pack(d):
    return (num(d["phi_star"]), num(d["gradient_lower"]), num(d["gradient_upper"]),
            num(d["gradient_star"], 7) + 0.0, num(d["curvature"]), num(d["convergence_slope"]))
"""

    return [
        {
            "setup": SETUP + """
def digest(fn):
    s = supply_sequence(800, 0.03, 31)
    return pack(isolated(fn, s, 100))
""",
            "call": "digest(singular_strategy)",
            "gold_call": "digest(_oracle_singular_strategy)",
        },
        {
            "setup": SETUP + """
def ordering(fn):
    s3 = supply_sequence(800, 0.03, 31)
    s1 = supply_sequence(800, 0.1, 31)
    strong = isolated(fn, s3, 100)["phi_star"]
    weak = isolated(fn, s1, 100)["phi_star"]
    tight = isolated(fn, s3, 100, 0.01, 0.9, 1e-11)["phi_star"]
    return (num(weak), bool(weak < strong), bool(abs(math.log(tight / strong)) < 2e-4))
""",
            "call": "ordering(singular_strategy)",
            "gold_call": "ordering(_oracle_singular_strategy)",
        },
        {
            "setup": SETUP + """
def narrow(fn):
    s = supply_sequence(800, 0.03, 31)
    wide = isolated(fn, s, 100)["phi_star"]
    tight = isolated(fn, s, 100, wide * 0.9, wide * 1.1)["phi_star"]
    return (num(tight), bool(abs(math.log(tight / wide)) < 2e-4))
""",
            "call": "narrow(singular_strategy)",
            "gold_call": "narrow(_oracle_singular_strategy)",
        },
        {
            "setup": SETUP + """
def rejects(fn):
    s = supply_sequence(200, 0.03, 31)
    bad = [
        (s, 50, 0.5, 0.2, 1e-4),
        (s, 50, 0.0, 0.5, 1e-4),
        (s, 50, 0.1, 0.99, 1e-4),
        (s, 50, 0.1, 0.5, 0.0),
        (s, 50, 0.6, 0.9, 1e-4),
        (np.full((20, 2), 0.5), 5, 0.01, 0.9, 1e-4),
    ]
    count = 0
    for args in bad:
        try:
            isolated(fn, *args)
        except ValueError:
            count += 1
    return count
""",
            "call": "rejects(singular_strategy)",
            "gold_call": "rejects(_oracle_singular_strategy)",
        },
    ]
