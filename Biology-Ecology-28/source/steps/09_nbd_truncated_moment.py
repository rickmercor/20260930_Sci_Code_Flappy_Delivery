"""
Compute the partial moment E[R^u; R <= C] of the l-th nearest-neighbour sector distance under the negative-binomial model, together with the censoring probability P(R > C).

Under negative-binomial aggregation the l-th nearest-neighbour distance moments exist only for -2l < u < 2k, and restricting the integral to the searched disc turns the complete beta-function moment into an incomplete one with shifted parameters. These partial moments are the large-sample limits of the uncensored parts of the empirical moment sums in a censored survey, and the censoring probability is the limiting censored fraction.

Returns
-------
return values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nbd_truncated_moment(u: float, lam: float, k: float, q: int, ell: int, C: float) -> list:
    """Return [E[R^u; R <= C], P(R > C)] for the ell-th nearest-neighbour distance under the NBD model.
 
    E[R^u; R <= C] denotes the integral of r^u g(r; lam, k) over 0 < r <= C, where g is the
    negative-binomial density of the ell-th nearest-neighbour distance in one of q sectors.
 
    Parameters
    ----------
    u : float
        Moment order, a finite real with ell + u / 2 > 0 and k - u / 2 > 0.
    lam : float
        Population density, finite and positive.
    k : float
        Aggregation parameter, finite and positive.
    q : int
        Number of sectors, an integer >= 1.
    ell : int
        Nearest-neighbour order, an integer >= 1.
    C : float
        Maximum search radius, finite and positive.
 
    Returns
    -------
    values : list of float
        [partial_moment, censoring_probability] as native Python floats.
 
    Raises
    ------
    ValueError
        If u, lam, k or C is not a finite real, if lam, k or C is not positive, if q or ell is not
        an integer >= 1, if ell + u / 2 <= 0 or k - u / 2 <= 0, if an incomplete beta evaluation
        fails, or if the moment is not representable as a finite float.
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nbd_truncated_moment(u: float, lam: float, k: float, q: int, ell: int, C: float) -> list:
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
    lam = _real(lam, "lam")
    k = _real(k, "k")
    C = _real(C, "C")
    q = _order(q, "q")
    ell = _order(ell, "ell")
    if lam <= 0.0 or k <= 0.0 or C <= 0.0:
        raise ValueError("lam, k and C must be positive")
    shape_a = ell + 0.5 * u
    shape_b = k - 0.5 * u
    if shape_a <= 0.0 or shape_b <= 0.0:
        raise ValueError("the moment requires ell + u/2 > 0 and k - u/2 > 0")

    mass = math.pi * lam * C * C / q
    if not math.isfinite(mass):
        raise ValueError("pi * lam * C**2 / q must be finite")
    w = mass / (mass + k)
    log_complete = (0.5 * u * math.log(k * q / (math.pi * lam)) + math.lgamma(shape_a) + math.lgamma(shape_b)
                    - math.lgamma(ell) - math.lgamma(k))
    try:
        complete = math.exp(log_complete)
    except OverflowError:
        raise ValueError("moment is not representable as a finite float") from None
    partial = complete * _oracle_regularized_beta(shape_a, shape_b, w)[0]
    censoring = _oracle_regularized_beta(float(ell), k, w)[1]
    if not math.isfinite(partial):
        raise ValueError("moment is not representable as a finite float")
    return [float(partial), float(censoring)]

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
        # normal: second moment of second-nearest distances inside a 10 m radius
        {"setup": "", "call": "nbd_truncated_moment(2.0, 0.05, 2.3, 4, 2, 10.0)", "gold_call": "_oracle_nbd_truncated_moment(2.0, 0.05, 2.3, 4, 2, 10.0)"},
        # normal: reciprocal moment of nearest-neighbour distances with k below one
        {"setup": "", "call": "nbd_truncated_moment(-1.0, 0.02, 0.8, 4, 1, 10.0)", "gold_call": "_oracle_nbd_truncated_moment(-1.0, 0.02, 0.8, 4, 1, 10.0)"},
        # boundary: a huge radius recovers the complete moment and a vanishing censoring probability
        {"setup": "", "call": "nbd_truncated_moment(1.0, 0.05, 3.0, 4, 2, 1e4)", "gold_call": "_oracle_nbd_truncated_moment(1.0, 0.05, 3.0, 4, 2, 1e4)"},
        # edge: strong aggregation, fractional order and a short radius
        {"setup": "", "call": "nbd_truncated_moment(0.5, 0.2, 0.3, 2, 3, 2.0)", "gold_call": "_oracle_nbd_truncated_moment(0.5, 0.2, 0.3, 2, 3, 2.0)"},
        # invalid: second moment needs k > 1
        _expect_value_error("nbd_truncated_moment(2.0, 0.05, 1.0, 4, 2, 10.0)", "_oracle_nbd_truncated_moment(2.0, 0.05, 1.0, 4, 2, 10.0)"),
        # invalid: order -2 with the nearest neighbour
        _expect_value_error("nbd_truncated_moment(-2.0, 0.05, 2.0, 4, 1, 10.0)", "_oracle_nbd_truncated_moment(-2.0, 0.05, 2.0, 4, 1, 10.0)"),
    ]
