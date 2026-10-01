"""
Evaluate the regularized incomplete beta function I_x(a, b) together with its complement 1 - I_x(a, b).

For a spatially aggregated population whose sector counts follow a negative binomial distribution with aggregation parameter k, the transformed distance w = pi * lambda * r^2 / (pi * lambda * r^2 + q * k) is beta distributed. The cumulative distribution of the l-th nearest-neighbour distance, the probability that a sector is censored, and the truncated distance moments are therefore regularized incomplete beta functions with possibly non-integer parameters.

Returns
-------
return values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regularized_beta(a: float, b: float, x: float) -> list:
    """Return [I_x(a, b), 1 - I_x(a, b)] for the regularized incomplete beta function.
 
    I_x(a, b) = B(x; a, b) / B(a, b) with B(x; a, b) = integral_0^x t**(a - 1) (1 - t)**(b - 1) dt.
 
    Parameters
    ----------
    a : float
        First shape parameter, finite and strictly positive.
    b : float
        Second shape parameter, finite and strictly positive.
    x : float
        Upper integration limit, finite with 0 <= x <= 1.
 
    Returns
    -------
    values : list of float
        [I, 1 - I] as native Python floats. The complement must be computed without
        cancellation, so that it stays accurate to at least 12 significant digits when I is
        close to one (and vice versa). I_0 = 0 and I_1 = 1.
 
    Raises
    ------
    ValueError
        If a, b or x is not a finite real number, if a <= 0 or b <= 0, if x is outside
        [0, 1], or if the continued-fraction evaluation fails to converge.
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_regularized_beta(a: float, b: float, x: float) -> list:
    import math

    def _real(value, name):
        if isinstance(value, complex):
            if value.imag != 0.0:
                raise ValueError(f"{name} must be a real number")
            value = value.real
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    a = _real(a, "a")
    b = _real(b, "b")
    x = _real(x, "x")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x < 0.0 or x > 1.0:
        raise ValueError("x must lie in [0, 1]")
    if x == 0.0:
        return [0.0, 1.0]
    if x == 1.0:
        return [1.0, 0.0]

    tiny = 1e-300

    def _continued_fraction(p, r, z):
        qab = p + r
        qap = p + 1.0
        qam = p - 1.0
        c = 1.0
        d = 1.0 - qab * z / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 100000):
            m2 = 2.0 * m
            aa = m * (r - m) * z / ((qam + m2) * (p + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            aa = -(p + m) * (qab + m) * z / ((p + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) <= 1e-15:
                return h
        raise ValueError("incomplete beta continued fraction did not converge")

    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        lower = min(math.exp(log_front + math.log(_continued_fraction(a, b, x))) / a, 1.0)
        return [float(lower), float(1.0 - lower)]
    upper = min(math.exp(log_front + math.log(_continued_fraction(b, a, 1.0 - x))) / b, 1.0)
    return [float(1.0 - upper), float(upper)]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
 
    def _expect_value_error(model_expr, gold_expr):
        setup = (
            "def run_model():\n"
            "    try:\n"
            f"        {model_expr}\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            f"        {gold_expr}\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
        )
        return {"setup": setup, "call": "run_model()", "gold_call": "run_gold()"}
 
    return [
        # normal: integer first parameter and non-integer aggregation parameter
        {"setup": "", "call": "regularized_beta(2.0, 2.3, 0.63)", "gold_call": "_oracle_regularized_beta(2.0, 2.3, 0.63)"},
        # normal: half-integer shifted parameters used by a truncated first moment
        {"setup": "", "call": "regularized_beta(2.5, 1.8, 0.41)", "gold_call": "_oracle_regularized_beta(2.5, 1.8, 0.41)"},
        # boundary: x = 1 gives I = 1 and a zero complement
        {"setup": "", "call": "regularized_beta(3.0, 0.7, 1.0)", "gold_call": "_oracle_regularized_beta(3.0, 0.7, 1.0)"},
        # boundary: x = 0 gives I = 0 and a unit complement
        {"setup": "", "call": "regularized_beta(3.0, 0.7, 0.0)", "gold_call": "_oracle_regularized_beta(3.0, 0.7, 0.0)"},
        # edge: complement far in the tail must survive (scaled to order one)
        {"setup": "", "call": "regularized_beta(2.0, 40.0, 0.9)[1] * 1e36", "gold_call": "_oracle_regularized_beta(2.0, 40.0, 0.9)[1] * 1e36"},
        # edge: strongly aggregated population with a small second parameter
        {"setup": "", "call": "regularized_beta(1.0, 0.05, 0.2)", "gold_call": "_oracle_regularized_beta(1.0, 0.05, 0.2)"},
        # invalid: non-positive second parameter
        _expect_value_error("regularized_beta(2.0, 0.0, 0.5)", "_oracle_regularized_beta(2.0, 0.0, 0.5)"),
        # invalid: x above one
        _expect_value_error("regularized_beta(2.0, 1.0, 1.2)", "_oracle_regularized_beta(2.0, 1.0, 1.2)"),
    ]
