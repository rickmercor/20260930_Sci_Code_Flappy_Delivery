"""
Evaluate the right-censored negative-binomial log-likelihood of a point-centred quarter survey at a given density and aggregation parameter.

If sector counts follow a negative binomial distribution with mean pi * lambda * r^2 / q and aggregation parameter k, the l-th nearest-neighbour distance has a closed-form density, and a sector censored at radius C contributes the probability that fewer than l individuals lie within C. The resulting likelihood uses every recorded distance and every censored sector, and the Poisson model is recovered as k grows without bound.

Returns
-------
return log_likelihood
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nbd_censored_log_likelihood(distances: list, C: float, ell: int, lam: float, k: float) -> float:
    """Return the censored NBD log-likelihood of the survey at (lam, k).
 
    The log-likelihood is the sum over uncensored sectors of log g(r; lam, k) plus n0 times
    log P(R > C; lam, k), where g is the negative-binomial density of the ell-th nearest-neighbour
    distance in one of q sectors and n0 is the number of censored sectors.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector. Rows must share one length q >= 1.
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.
    lam : float
        Population density (individuals per squared length unit), finite and positive.
    k : float
        Negative-binomial aggregation parameter, finite and positive.
 
    Returns
    -------
    log_likelihood : float
        Natural-log likelihood as a native Python float (including all normalizing constants).
 
    Raises
    ------
    ValueError
        If C, lam or k is not a finite positive real, if ell is not an integer >= 1, if distances is
        not a non-empty rectangular sequence of rows whose entries are None or finite positive reals,
        if every sector is censored, or if the censoring probability underflows to zero while some
        sector is censored.
    """
    return log_likelihood

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nbd_censored_log_likelihood(distances: list, C: float, ell: int, lam: float, k: float) -> float:
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
    lam = _real(lam, "lam")
    k = _real(k, "k")
    if C <= 0.0 or lam <= 0.0 or k <= 0.0:
        raise ValueError("C, lam and k must be positive")
    ell = _order(ell, "ell")
    observed, censored, _, q = _survey(distances, C)

    rate = math.pi * lam / q
    constant = (math.log(2.0) + ell * math.log(rate) + math.lgamma(ell + k) - math.lgamma(k)
                - math.lgamma(ell) - ell * math.log(k))
    terms = [constant + (2 * ell - 1) * math.log(r) - (ell + k) * math.log1p(rate * r * r / k) for r in observed]
    if censored:
        mass = rate * C * C
        w = mass / (mass + k)
        survival = _oracle_regularized_beta(float(ell), k, w)[1]
        if survival <= 0.0:
            raise ValueError("censoring probability underflows to zero")
        terms.append(censored * math.log(survival))
    value = math.fsum(terms)
    if not math.isfinite(value):
        raise ValueError("log-likelihood is not finite")
    return float(value)

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
        # normal: second-nearest distances, moderate aggregation
        {"setup": survey, "call": "nbd_censored_log_likelihood(S, 10.0, 2, 0.045, 2.1)",
         "gold_call": "_oracle_nbd_censored_log_likelihood(S, 10.0, 2, 0.045, 2.1)"},
        # normal: nearest neighbour, strong aggregation (k below one)
        {"setup": "S = [[0.6, None, 7.9, 0.8], [None, None, 1.1, 9.4], [0.5, 6.6, None, 0.9]]",
         "call": "nbd_censored_log_likelihood(S, 10.0, 1, 0.08, 0.6)",
         "gold_call": "_oracle_nbd_censored_log_likelihood(S, 10.0, 1, 0.08, 0.6)"},
        # boundary: no censored sector, only the density terms contribute
        {"setup": "S = [[2.0, 3.5, 1.2, 4.4], [2.5, 0.9, 3.3, 5.0]]",
         "call": "nbd_censored_log_likelihood(S, 10.0, 1, 0.1, 3.0)",
         "gold_call": "_oracle_nbd_censored_log_likelihood(S, 10.0, 1, 0.1, 3.0)"},
        # edge: nearly Poisson population (very large k) with third-nearest distances
        {"setup": "S = [[None, 5.9, None, 4.8], [None, None, 3.9, None], [5.5, None, None, 2.7]]",
         "call": "nbd_censored_log_likelihood(S, 6.0, 3, 0.12, 1e6)",
         "gold_call": "_oracle_nbd_censored_log_likelihood(S, 6.0, 3, 0.12, 1e6)"},
        # invalid: non-positive aggregation parameter
        _expect_value_error("nbd_censored_log_likelihood([[1.0, None]], 10.0, 1, 0.1, 0.0)",
                           "_oracle_nbd_censored_log_likelihood([[1.0, None]], 10.0, 1, 0.1, 0.0)"),
        # invalid: NaN distance
        _expect_value_error("nbd_censored_log_likelihood([[1.0, float('nan')]], 10.0, 1, 0.1, 2.0)",
                           "_oracle_nbd_censored_log_likelihood([[1.0, float('nan')]], 10.0, 1, 0.1, 2.0)"),
    ]
