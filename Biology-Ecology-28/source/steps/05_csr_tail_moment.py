"""
Compute the conditional moment E[R^u | R > C] of the l-th nearest-neighbour sector distance under complete spatial randomness with a given density.

The negative-binomial moment estimators for censored surveys keep every uncensored distance and replace each censored sector by the expected value of R^u beyond the search radius. That expectation is evaluated in the Poisson (complete spatial randomness) limit at an initial density estimate, which gives a ratio of upper incomplete gamma functions; for large expected counts inside the radius both functions underflow, while their ratio tends smoothly to C^u.

Returns
-------
return moment
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def csr_tail_moment(u: float, ell: int, q: int, lam: float, C: float) -> float:
    """Return E[R^u | R > C] for the ell-th nearest-neighbour distance in one of q sectors under CSR.
 
    Parameters
    ----------
    u : float
        Moment order, a finite real with ell + u / 2 > 0 (negative orders allowed).
    ell : int
        Nearest-neighbour order, an integer >= 1.
    q : int
        Number of equal-angle sectors around a sampling point, an integer >= 1.
    lam : float
        Population density used for the Poisson model, finite and positive.
    C : float
        Maximum search radius, finite and positive.
 
    Returns
    -------
    moment : float
        Native Python float. It must stay finite and accurate when the expected number of
        individuals inside the radius is very large (hundreds or more), where it approaches C**u.
 
    Raises
    ------
    ValueError
        If u, lam or C is not a finite real, if ell or q is not an integer >= 1, if lam <= 0 or
        C <= 0, if ell + u / 2 <= 0, if the incomplete gamma evaluation fails to converge, or if
        the moment is not representable as a finite float.
    """
    return moment

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_csr_tail_moment(u: float, ell: int, q: int, lam: float, C: float) -> float:
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

    def _order(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer >= 1")
        number = _real(value, name)
        if number != math.floor(number) or number < 1.0:
            raise ValueError(f"{name} must be an integer >= 1")
        return int(number)

    u = _real(u, "u")
    ell = _order(ell, "ell")
    q = _order(q, "q")
    lam = _real(lam, "lam")
    C = _real(C, "C")
    if lam <= 0.0 or C <= 0.0:
        raise ValueError("lam and C must be positive")
    shape = ell + 0.5 * u
    if shape <= 0.0:
        raise ValueError("ell + u / 2 must be positive")
    scale = math.pi * lam / q
    T = scale * C * C
    if not math.isfinite(T):
        raise ValueError("pi * lam * C**2 / q must be finite")

    tiny = 1e-300

    def _log_upper_gamma(s, x):
        if x == 0.0:
            return math.lgamma(s)
        if x < s + 1.0:
            lower = _oracle_regularized_gamma(s, x)[0]
            if lower >= 1.0:
                raise ValueError("upper incomplete gamma function underflows")
            return math.lgamma(s) + math.log1p(-lower)
        b = x + 1.0 - s
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 100000):
            an = -i * (i - s)
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
        return -x + s * math.log(x) + math.log(h)

    log_moment = -0.5 * u * math.log(scale) + _log_upper_gamma(shape, T) - _log_upper_gamma(float(ell), T)
    try:
        moment = math.exp(log_moment)
    except OverflowError:
        raise ValueError("tail moment is not representable as a finite float") from None
    if not math.isfinite(moment) or moment == 0.0:
        raise ValueError("tail moment is not representable as a finite float")
    return float(moment)

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
        # normal: second moment beyond a 10 m radius for second-nearest distances
        {"setup": "", "call": "csr_tail_moment(2.0, 2, 4, 0.0375, 10.0)", "gold_call": "_oracle_csr_tail_moment(2.0, 2, 4, 0.0375, 10.0)"},
        # normal: reciprocal moment for the nearest neighbour (half-integer shape)
        {"setup": "", "call": "csr_tail_moment(-1.0, 1, 4, 0.02, 10.0)", "gold_call": "_oracle_csr_tail_moment(-1.0, 1, 4, 0.02, 10.0)"},
        # boundary: very sparse population, the tail moment approaches the complete moment
        {"setup": "", "call": "csr_tail_moment(-2.0, 3, 2, 1e-7, 1.0) * 1e7", "gold_call": "_oracle_csr_tail_moment(-2.0, 3, 2, 1e-7, 1.0) * 1e7"},
        # edge: dense population, both incomplete gamma functions underflow but the ratio is close to C**u
        {"setup": "", "call": "csr_tail_moment(1.0, 2, 4, 2.5, 20.0)", "gold_call": "_oracle_csr_tail_moment(1.0, 2, 4, 2.5, 20.0)"},
        # edge: fractional moment order with a single sector
        {"setup": "", "call": "csr_tail_moment(0.7, 1, 1, 0.3, 1.5)", "gold_call": "_oracle_csr_tail_moment(0.7, 1, 1, 0.3, 1.5)"},
        # invalid: ell + u / 2 = 0
        _expect_value_error("csr_tail_moment(-2.0, 1, 4, 0.05, 10.0)", "_oracle_csr_tail_moment(-2.0, 1, 4, 0.05, 10.0)"),
        # invalid: zero sectors
        _expect_value_error("csr_tail_moment(1.0, 2, 0, 0.05, 10.0)", "_oracle_csr_tail_moment(1.0, 2, 0, 0.05, 10.0)"),
        # invalid: non-positive density
        _expect_value_error("csr_tail_moment(1.0, 2, 4, -0.05, 10.0)", "_oracle_csr_tail_moment(1.0, 2, 4, -0.05, 10.0)"),
    ]
