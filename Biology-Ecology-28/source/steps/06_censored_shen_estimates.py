"""
Compute the censored negative-binomial moment estimates of population density and aggregation parameter from a right-censored point-centred quarter survey.

For aggregated populations the negative binomial distance model links the reciprocal, first and second distance moments to both the density and the aggregation parameter k. With a capped search radius, each moment is estimated by keeping the uncensored distances and imputing every censored sector with its complete-spatial-randomness conditional moment beyond the radius, evaluated at the censored Pollard-type density; the density and k then follow from the negative binomial moment relations.

Returns
-------
return estimates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def censored_shen_estimates(distances: list, C: float, ell: int) -> list:
    """Return [lambda_n, k_n], the censored NBD moment estimates of density and aggregation.
 
    For u in (-1, 1, 2) the adjusted moment is
        E_u = (sum of r^u over uncensored sectors + n0 * E_CSR[R^u | R > C]) / (n q),
    where n0 is the number of censored sectors and the conditional moment uses the censored
    Pollard-type density lambda_P as the initial density. Then
        lambda_n = q (2 ell - 1) E_{-1} / (pi E_1) - q ell / (pi E_2),
    and k_n is the aggregation parameter k at which the exact negative-binomial moment ratio
    E[R^-1] E[R^2] / E[R] equals the ratio E_{-1} E_2 / E_1 of the adjusted moments.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector.
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.
 
    Returns
    -------
    estimates : list of float
        [lambda_n, k_n] as native Python floats. With no censored sector the adjusted moments are
        the plain sample moments.
 
    Raises
    ------
    ValueError
        Under the conditions of poisson_adjusted_moments, if the censored Pollard-type density is
        not positive, if a conditional tail moment is invalid (see csr_tail_moment), or if the
        moment ratio makes k_n undefined (division by zero).
    """
    return estimates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_censored_shen_estimates(distances: list, C: float, ell: int) -> list:
    import math

    _, lam_init = _oracle_censored_poisson_densities(distances, C, ell)
    if not lam_init > 0.0:
        raise ValueError("the censored Pollard-type density must be positive")
    radius = float(C)
    order = int(float(ell))
    rows = [list(row) for row in distances]
    n_points = len(rows)
    q = len(rows[0])
    total = n_points * q
    observed = []
    censored = 0
    for row in rows:
        for entry in row:
            if entry is None or float(entry) > radius:
                censored += 1
            else:
                observed.append(float(entry))

    adjusted = {}
    for u in (-1.0, 1.0, 2.0):
        tail = _oracle_csr_tail_moment(u, order, q, lam_init, radius) if censored else 0.0
        adjusted[u] = (math.fsum(r ** u for r in observed) + censored * tail) / total

    lam_n = (q * (2 * order - 1) * adjusted[-1.0] / (math.pi * adjusted[1.0])
             - q * order / (math.pi * adjusted[2.0]))
    ratio = adjusted[-1.0] * adjusted[2.0] / adjusted[1.0]
    denominator = (2 * order - 1) * ratio - 2 * order
    if denominator == 0.0:
        raise ValueError("aggregation estimate is undefined for this moment ratio")
    k_n = ((2 * order - 1) * ratio - order) / denominator
    return [float(lam_n), float(k_n)]

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
 
    survey = ("S = [[1.58, None, 9.52, 5.8], [8.54, 8.05, 3.38, None], [4.52, 5.25, 5.84, None], "
              "[5.94, 5.91, 4.41, 3.55], [6.8, 6.42, 9.8, None], [9.9, None, None, 5.99]]")
    return [
        # normal: second-nearest distances with seven censored sectors
        {"setup": survey, "call": "censored_shen_estimates(S, 10.0, 2)", "gold_call": "_oracle_censored_shen_estimates(S, 10.0, 2)"},
        # normal: aggregated nearest-neighbour survey with long gaps and short clustered distances
        {"setup": "S = [[0.6, None, 7.9, 0.8], [None, None, 1.1, 9.4], [0.5, 6.6, None, 0.9], [2.2, None, 0.7, 5.1]]",
         "call": "censored_shen_estimates(S, 10.0, 1)", "gold_call": "_oracle_censored_shen_estimates(S, 10.0, 1)"},
        # boundary: no censored sector, plain sample moments
        {"setup": "S = [[2.0, 3.5, 1.2, 4.4], [2.5, 0.9, 3.3, 5.0]]",
         "call": "censored_shen_estimates(S, 10.0, 1)", "gold_call": "_oracle_censored_shen_estimates(S, 10.0, 1)"},
        # edge: third-nearest distances with a short radius and heavy censoring
        {"setup": "S = [[None, 5.9, None, 4.8], [None, None, 3.9, None], [5.5, None, None, 2.7]]",
         "call": "censored_shen_estimates(S, 6.0, 3)", "gold_call": "_oracle_censored_shen_estimates(S, 6.0, 3)"},
        # invalid: a single sector makes the Pollard-type density zero
        _expect_value_error("censored_shen_estimates([[3.0]], 10.0, 1)", "_oracle_censored_shen_estimates([[3.0]], 10.0, 1)"),
        # invalid: text instead of a survey
        _expect_value_error("censored_shen_estimates('1.0 2.0', 10.0, 1)", "_oracle_censored_shen_estimates('1.0 2.0', 10.0, 1)"),
    ]
