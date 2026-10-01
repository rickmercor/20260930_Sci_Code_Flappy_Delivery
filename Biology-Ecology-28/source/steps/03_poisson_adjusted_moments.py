"""
From a right-censored point-centred quarter survey, compute the estimated censoring quantile m_hat_C and the censoring-adjusted first and second distance moments of the Poisson framework.

When the search radius is capped at C, sectors in which fewer than l individuals lie within C are right-censored, and plain averages of the recorded distances are biased downward. Under complete spatial randomness the observed proportion of uncensored sectors fixes the expected sector count within C through the gamma distribution function of order l, and each empirical moment can then be inflated by the ratio of the complete to the truncated gamma integral of matching order.

Returns
-------
return values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def poisson_adjusted_moments(distances: list, C: float, ell: int) -> list:
    """Return [m_hat_C, M_1, M_2] for a right-censored point-centred quarter survey.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q equal-angle sectors (columns). Each entry is the distance
        from the sampling point to the ell-th nearest individual in that sector, a finite
        positive real, or None when the sector is censored. A recorded distance larger than C
        is also treated as censored. Every row must have the same length q >= 1.
    C : float
        Maximum search radius, finite and positive, in the same length unit as the distances.
    ell : int
        Nearest-neighbour order recorded in every sector, an integer >= 1.
 
    Returns
    -------
    values : list of float
        [m_hat_C, M_1, M_2] as native Python floats. m_hat_C solves
        gamma(ell, m) / Gamma(ell) = 1 - n0 / (n q), where n0 is the number of censored
        sectors; M_u (u = 1, 2) is the Poisson censoring-adjusted estimate of E[R^u], built from
        the sum of r^u over uncensored sectors divided by n q. With no censored sector,
        m_hat_C = inf and M_u is the plain sample moment.
 
    Raises
    ------
    ValueError
        If C is not a finite positive real, if ell is not an integer >= 1, if distances is not a
        non-empty rectangular sequence of rows with at least one sector, if an entry is neither
        None nor a finite positive real, or if every sector is censored.
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_poisson_adjusted_moments(distances: list, C: float, ell: int) -> list:
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

    def _survey(table, radius):
        if isinstance(table, (str, bytes)):
            raise ValueError("distances must be a sequence of rows")
        try:
            rows = list(table)
        except TypeError:
            raise ValueError("distances must be a sequence of rows") from None
        if not rows:
            raise ValueError("distances must contain at least one sampling point")
        observed = []
        censored = 0
        width = None
        for row in rows:
            if isinstance(row, (str, bytes)):
                raise ValueError("each sampling point must be a sequence of sector distances")
            try:
                entries = list(row)
            except TypeError:
                raise ValueError("each sampling point must be a sequence of sector distances") from None
            if not entries:
                raise ValueError("each sampling point needs at least one sector")
            if width is None:
                width = len(entries)
            elif len(entries) != width:
                raise ValueError("all sampling points must have the same number of sectors")
            for entry in entries:
                if entry is None:
                    censored += 1
                    continue
                r = _real(entry, "distance")
                if r <= 0.0:
                    raise ValueError("distances must be positive")
                if r > radius:
                    censored += 1
                else:
                    observed.append(r)
        if not observed:
            raise ValueError("every sector is censored")
        return observed, censored, len(rows), width

    C = _real(C, "C")
    if C <= 0.0:
        raise ValueError("C must be positive")
    ell = _order(ell, "ell")
    observed, censored, n_points, q = _survey(distances, C)
    total = n_points * q
    first = math.fsum(observed) / total
    second = math.fsum(r * r for r in observed) / total
    if censored == 0:
        return [math.inf, float(first), float(second)]

    p_obs = (total - censored) / total

    def _lower(shape, m):
        return _oracle_regularized_gamma(shape, m)[0]

    hi = max(1.0, float(ell))
    while _lower(ell, hi) < p_obs:
        hi *= 2.0
        if hi > 1e300:
            raise ValueError("censoring quantile is not representable")
    lo = 0.0
    for _ in range(4000):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _lower(ell, mid) < p_obs:
            lo = mid
        else:
            hi = mid
    m_hat = lo if abs(_lower(ell, lo) - p_obs) < abs(_lower(ell, hi) - p_obs) else hi

    moment1 = first / _lower(ell + 0.5, m_hat)
    moment2 = second / _lower(ell + 1.0, m_hat)
    return [float(m_hat), float(moment1), float(moment2)]

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
 
    survey = "S = [[1.58, None, 9.52, 5.8], [8.54, 8.05, 3.38, None], [4.52, 5.25, 5.84, None], [5.94, 5.91, 4.41, 3.55]]"
    return [
        # normal: second-nearest distances with three censored sectors
        {"setup": survey, "call": "poisson_adjusted_moments(S, 10.0, 2)", "gold_call": "_oracle_poisson_adjusted_moments(S, 10.0, 2)"},
        # normal: nearest neighbour, censoring flagged by distances beyond the radius
        {"setup": "S = [[3.1, 12.5, 4.2, 9999.0], [2.2, 7.7, None, 6.1]]",
         "call": "poisson_adjusted_moments(S, 10.0, 1)", "gold_call": "_oracle_poisson_adjusted_moments(S, 10.0, 1)"},
        # boundary: no censored sector, adjusted moments equal the sample moments
        {"setup": "S = [[2.0, 3.0], [4.0, 5.0]]",
         "call": "poisson_adjusted_moments(S, 10.0, 1)[1:]", "gold_call": "_oracle_poisson_adjusted_moments(S, 10.0, 1)[1:]"},
        # edge: seven of eight sectors censored with third-nearest distances
        {"setup": "S = [[None, None, None, 9.9], [None, None, None, None]]",
         "call": "poisson_adjusted_moments(S, 10.0, 3)", "gold_call": "_oracle_poisson_adjusted_moments(S, 10.0, 3)"},
        # edge: a distance exactly at the radius is uncensored
        {"setup": "S = [[10.0, None, 2.5]]",
         "call": "poisson_adjusted_moments(S, 10.0, 1)", "gold_call": "_oracle_poisson_adjusted_moments(S, 10.0, 1)"},
        # invalid: ragged survey
        _expect_value_error("poisson_adjusted_moments([[1.0, 2.0], [3.0]], 10.0, 1)", "_oracle_poisson_adjusted_moments([[1.0, 2.0], [3.0]], 10.0, 1)"),
        # invalid: every sector censored
        _expect_value_error("poisson_adjusted_moments([[None, 11.0]], 10.0, 1)", "_oracle_poisson_adjusted_moments([[None, 11.0]], 10.0, 1)"),
        # invalid: non-integer neighbour order
        _expect_value_error("poisson_adjusted_moments([[1.0, 2.0]], 10.0, 1.5)", "_oracle_poisson_adjusted_moments([[1.0, 2.0]], 10.0, 1.5)"),
    ]
