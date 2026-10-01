"""
Compute the censored Cottam-type and censored Pollard-type density estimates from a right-censored point-centred quarter survey.

The classical Cottam-type and Pollard-type estimators of population density use the first and second moments of point-to-individual distances under complete spatial randomness. Replacing those sample moments with their censoring-adjusted counterparts gives density estimates that remain usable when the search radius is capped; for the nearest neighbour the Cottam-type version reproduces the Warde-Petranka correction.

Returns
-------
return densities
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def censored_poisson_densities(distances: list, C: float, ell: int) -> list:
    """Return [lambda_C, lambda_P], the censored Cottam-type and Pollard-type density estimates.

    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector (same convention as
        poisson_adjusted_moments).
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.

    Returns
    -------
    densities : list of float
        [lambda_C, lambda_P] in individuals per squared length unit, as native Python floats:
        lambda_C = q * ell / (4 * M_1**2) and lambda_P = (n * q * ell - 1) / (pi * n * M_2),
        with M_1 and M_2 the Poisson censoring-adjusted moments.

    Raises
    ------
    ValueError
        Under the same conditions as poisson_adjusted_moments (invalid C, ell or survey, or
        every sector censored).
    """
    return densities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_censored_poisson_densities(
    distances: list, C: float, ell: int
) -> list:
    """Oracle implementation of censored_poisson_densities."""
    import math

    moments = _oracle_poisson_adjusted_moments(distances, C, ell)
    M_1 = moments[1]
    M_2 = moments[2]
    n = len(distances)
    q = len(distances[0])

    lambda_C = q * ell / (4.0 * M_1**2)
    lambda_P = (n * q * ell - 1.0) / (math.pi * n * M_2)

    return [float(lambda_C), float(lambda_P)]

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
        {"setup": survey, "call": "censored_poisson_densities(S, 10.0, 2)", "gold_call": "_oracle_censored_poisson_densities(S, 10.0, 2)"},
        # normal: nearest neighbour with half of the sectors censored
        {"setup": "S = [[3.1, None, 4.2, None], [2.2, 7.7, None, None], [5.5, None, 1.9, 8.8]]",
         "call": "censored_poisson_densities(S, 9.0, 1)", "gold_call": "_oracle_censored_poisson_densities(S, 9.0, 1)"},
        # boundary: a single uncensored sector, the Pollard numerator vanishes
        {"setup": "S = [[3.0]]", "call": "censored_poisson_densities(S, 10.0, 1)", "gold_call": "_oracle_censored_poisson_densities(S, 10.0, 1)"},
        # edge: two sectors per point and heavy censoring of third-nearest distances
        {"setup": "S = [[None, 11.6], [None, None], [4.4, None]]",
         "call": "censored_poisson_densities(S, 12.0, 3)", "gold_call": "_oracle_censored_poisson_densities(S, 12.0, 3)"},
        # invalid: negative distance
        _expect_value_error("censored_poisson_densities([[1.0, -2.0]], 10.0, 1)", "_oracle_censored_poisson_densities([[1.0, -2.0]], 10.0, 1)"),
        # invalid: non-positive radius
        _expect_value_error("censored_poisson_densities([[1.0, 2.0]], 0.0, 1)", "_oracle_censored_poisson_densities([[1.0, 2.0]], 0.0, 1)"),
    ]
