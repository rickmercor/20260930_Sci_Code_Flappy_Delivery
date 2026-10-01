"""
For a negative-binomial population with known density and aggregation, compute the large-sample limits of the censored Pollard-type initial density and of the censored negative-binomial moment density estimate, and the resulting asymptotic relative bias.

As the number of sampling points grows, the censored fraction converges to the model censoring probability and each empirical partial moment converges to its model value, while censored sectors are still imputed with complete-spatial-randomness tail moments at the limiting initial density. Because that imputation ignores aggregation beyond the search radius, the censored moment estimator keeps a systematic bias that depends only on the design (q, l, C) and on the population.

Returns
-------
return values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def asymptotic_shen_bias(lam: float, k: float, q: int, ell: int, C: float) -> list:
    """Return [lambda_init_inf, lambda_n_inf, relative_bias] for the censored NBD moment estimator.
 
    In the limit n -> infinity with the censored fraction equal to p0 = P(R > C; lam, k):
      * m_inf solves gamma(ell, m) / Gamma(ell) = 1 - p0;
      * M_2_inf = E[R^2; R <= C] * Gamma(ell + 1) / gamma(ell + 1, m_inf) is the limit of the
        Poisson censoring-adjusted second moment, and lambda_init_inf = ell * q / (pi * M_2_inf)
        is the limit of the censored Pollard-type density;
      * for u in (-1, 1, 2), E_u_inf = E[R^u; R <= C] + p0 * E_CSR[R^u | R > C] with the
        conditional CSR moment evaluated at lambda_init_inf;
      * lambda_n_inf = q (2 ell - 1) E_{-1,inf} / (pi E_{1,inf}) - q ell / (pi E_{2,inf}), and
        relative_bias = lambda_n_inf / lam - 1.
    If p0 underflows to zero, all censoring corrections vanish (m_inf = inf).
 
    Parameters
    ----------
    lam : float
        True population density, finite and positive.
    k : float
        True aggregation parameter, finite with k > 1 (so that E[R^2] exists).
    q : int
        Number of sectors, an integer >= 1.
    ell : int
        Nearest-neighbour order, an integer >= 1.
    C : float
        Maximum search radius, finite and positive.
 
    Returns
    -------
    values : list of float
        [lambda_init_inf, lambda_n_inf, relative_bias] as native Python floats.
 
    Raises
    ------
    ValueError
        If lam or C is not a finite positive real, if k is not a finite real > 1, if q or ell is not
        an integer >= 1, if the limiting uncensored fraction 1 - p0 underflows to zero, or if a
        special-function evaluation fails (see nbd_truncated_moment and csr_tail_moment).
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_asymptotic_shen_bias(lam: float, k: float, q: int, ell: int, C: float) -> list:
    import math

    partial = {}
    p0 = None
    for u in (-1.0, 1.0, 2.0):
        partial[u], p0 = _oracle_nbd_truncated_moment(u, lam, k, q, ell, C)
    lam = float(lam)
    k = float(k)
    q = int(float(q))
    ell = int(float(ell))
    C = float(C)
    mass = math.pi * lam * C * C / q
    w = mass / (mass + k)
    p_obs = _oracle_regularized_beta(float(ell), k, w)[0]

    if p0 == 0.0:
        second = partial[2.0]
        lam_init = ell * q / (math.pi * second)
        adjusted = dict(partial)
    else:
        if p_obs <= 0.0:
            raise ValueError("the limiting uncensored fraction underflows to zero")

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
        m_inf = lo if abs(_lower(ell, lo) - p_obs) < abs(_lower(ell, hi) - p_obs) else hi
        second = partial[2.0] / _lower(ell + 1.0, m_inf)
        lam_init = ell * q / (math.pi * second)
        adjusted = {u: partial[u] + p0 * _oracle_csr_tail_moment(u, ell, q, lam_init, C) for u in (-1.0, 1.0, 2.0)}

    lam_n = (q * (2 * ell - 1) * adjusted[-1.0] / (math.pi * adjusted[1.0])
             - q * ell / (math.pi * adjusted[2.0]))
    return [float(lam_init), float(lam_n), float(lam_n / lam - 1.0)]

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
        # normal: second-nearest distances, moderate aggregation, about 30 percent censoring
        {"setup": "", "call": "asymptotic_shen_bias(0.0437, 2.126, 4, 2, 10.0)", "gold_call": "_oracle_asymptotic_shen_bias(0.0437, 2.126, 4, 2, 10.0)"},
        # normal: nearest neighbour with weaker censoring
        {"setup": "", "call": "asymptotic_shen_bias(0.03, 1.9, 4, 1, 10.0)", "gold_call": "_oracle_asymptotic_shen_bias(0.03, 1.9, 4, 1, 10.0)"},
        # boundary: a very large radius leaves almost no censoring and almost no bias
        {"setup": "", "call": "asymptotic_shen_bias(0.05, 3.0, 4, 2, 200.0)", "gold_call": "_oracle_asymptotic_shen_bias(0.05, 3.0, 4, 2, 200.0)"},
        # edge: sparse population, third-nearest distances, heavy censoring
        {"setup": "", "call": "asymptotic_shen_bias(0.005, 1.5, 4, 3, 10.0)", "gold_call": "_oracle_asymptotic_shen_bias(0.005, 1.5, 4, 3, 10.0)"},
        # edge: nearly Poisson population
        {"setup": "", "call": "asymptotic_shen_bias(0.02, 500.0, 2, 2, 8.0)", "gold_call": "_oracle_asymptotic_shen_bias(0.02, 500.0, 2, 2, 8.0)"},
        # invalid: k = 1 has no finite second moment
        _expect_value_error("asymptotic_shen_bias(0.05, 1.0, 4, 2, 10.0)", "_oracle_asymptotic_shen_bias(0.05, 1.0, 4, 2, 10.0)"),
        # invalid: zero density
        _expect_value_error("asymptotic_shen_bias(0.0, 2.0, 4, 2, 10.0)", "_oracle_asymptotic_shen_bias(0.0, 2.0, 4, 2, 10.0)"),
    ]
