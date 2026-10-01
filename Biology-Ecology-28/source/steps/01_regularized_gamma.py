"""
Evaluate the regularized lower and upper incomplete gamma functions P(a, x) and Q(a, x) = 1 - P(a, x).

Under complete spatial randomness the number of individuals inside one sector of radius r is Poisson distributed, so the distance to the l-th nearest individual has a gamma-type distribution in the variable pi * lambda * r^2 / q. Every censoring correction in point-centred quarter sampling (the censored proportion, the moment inflation factors and the conditional tail moments) is written with incomplete gamma functions, and both tails must stay accurate when one of them is close to one.

Returns
-------
return values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regularized_gamma(a: float, x: float) -> list:
    """Return the regularized incomplete gamma functions [P(a, x), Q(a, x)].
 
    P(a, x) = gamma(a, x) / Gamma(a) with gamma(a, x) = integral_0^x t**(a - 1) exp(-t) dt,
    and Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x).
 
    Parameters
    ----------
    a : float
        Shape parameter, finite and strictly positive (need not be an integer).
    x : float
        Argument, finite and non-negative.
 
    Returns
    -------
    values : list of float
        [P, Q] as native Python floats, each accurate to at least 12 significant digits
        relative to the smaller of the two tails' natural scale (P(a, 0) = 0, Q(a, 0) = 1).
 
    Raises
    ------
    ValueError
        If a or x is not a finite real number, if a <= 0, if x < 0, or if the series or
        continued-fraction evaluation fails to converge.
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_regularized_gamma(a: float, x: float) -> list:
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
    x = _real(x, "x")
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return [0.0, 1.0]

    tiny = 1e-300
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        shape = a
        for _ in range(100000):
            shape += 1.0
            term *= x / shape
            total += term
            if abs(term) <= abs(total) * 1e-17:
                break
        else:
            raise ValueError("incomplete gamma series did not converge")
        lower = min(math.exp(log_prefactor + math.log(total)), 1.0)
        return [float(lower), float(1.0 - lower)]

    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 1e-15:
            break
    else:
        raise ValueError("incomplete gamma continued fraction did not converge")
    upper = min(math.exp(log_prefactor + math.log(h)), 1.0)
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
        # normal: integer shape used for the censored proportion with l = 2
        {"setup": "", "call": "regularized_gamma(2.0, 1.3)", "gold_call": "_oracle_regularized_gamma(2.0, 1.3)"},
        # normal: half-integer shape used by the first adjusted moment, argument above a + 1
        {"setup": "", "call": "regularized_gamma(2.5, 6.75)", "gold_call": "_oracle_regularized_gamma(2.5, 6.75)"},
        # boundary: x = 0 gives P = 0 and Q = 1 exactly
        {"setup": "", "call": "regularized_gamma(3.0, 0.0)", "gold_call": "_oracle_regularized_gamma(3.0, 0.0)"},
        # edge: far upper tail, Q must not be lost to cancellation
        {"setup": "", "call": "regularized_gamma(1.5, 60.0)[1] * 1e24", "gold_call": "_oracle_regularized_gamma(1.5, 60.0)[1] * 1e24"},
        # edge: small shape with argument close to a + 1
        {"setup": "", "call": "regularized_gamma(0.05, 1.02)", "gold_call": "_oracle_regularized_gamma(0.05, 1.02)"},
        # edge: large shape near its mean
        {"setup": "", "call": "regularized_gamma(400.0, 390.0)", "gold_call": "_oracle_regularized_gamma(400.0, 390.0)"},
        # invalid: non-positive shape
        _expect_value_error("regularized_gamma(0.0, 1.0)", "_oracle_regularized_gamma(0.0, 1.0)"),
        # invalid: negative argument
        _expect_value_error("regularized_gamma(2.0, -0.5)", "_oracle_regularized_gamma(2.0, -0.5)"),
    ]
