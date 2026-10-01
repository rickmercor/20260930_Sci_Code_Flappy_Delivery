"""
Fit the censored negative-binomial model to a point-centred quarter survey and report the asymptotic relative bias of the censored negative-binomial moment density estimator for that fitted population and the same design.

This is the final orchestrator. The joint maximum likelihood fit gives the density and aggregation of the surveyed population, and the large-sample analysis of the moment-based censoring correction then shows how far the simpler closed-form estimator would drift from the true density if the same sectors, neighbour order and search radius were used with many more sampling points.

Returns
-------
return relative_bias
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def censored_design_bias(distances: list, C: float, ell: int) -> float:
    """Return the asymptotic relative bias of the censored NBD moment estimator at the fitted population.
 
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
    relative_bias : float
        lambda_n_inf / lam_hat - 1 as a native Python float, where (lam_hat, k_hat) is the censored
        NBD maximum likelihood fit and lambda_n_inf is the large-sample limit of the censored NBD
        moment density estimate for q sectors, order ell and radius C at that population.
 
    Raises
    ------
    ValueError
        Under the conditions of nbd_censored_mle, if k_hat <= 1, or under the conditions of
        asymptotic_shen_bias.
    """
    return relative_bias

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_censored_design_bias(distances: list, C: float, ell: int) -> float:
    lam_hat, k_hat, _ = _oracle_nbd_censored_mle(distances, C, ell)
    if not k_hat > 1.0:
        raise ValueError("the fitted aggregation parameter must exceed 1")
    q = len(list(list(distances)[0]))
    return float(_oracle_asymptotic_shen_bias(lam_hat, k_hat, q, ell, C)[2])

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
        # normal: second-nearest distances, 10 m radius
        {"setup": "D = [[9.48, None, None, 9.33], [None, None, 6.39, 9.67], [7.19, 2.46, None, 3.84], [3.67, 7.9, 6.59, 6.66], [5.34, None, 9.26, None], [None, 3.23, 2.28, 3.16], [7.56, 6.13, 6.31, 4.69], [3.5, 6.96, 2.75, 4.17], [5.35, 4.85, 6.22, 6.77], [8.09, 6.74, 5.75, 4.74], [None, 3.72, 5.38, None], [7.15, None, None, None], [None, 4.61, None, 8.5], [7.47, 8.3, None, 4.04], [3.48, 3.11, 3.33, 3.56], [8.04, 5.74, 5.87, 9.85], [6.99, 7.68, None, 7.82], [6.73, 5.58, 6.32, 4.4], [7.8, 8.96, None, None], [None, 4.84, 5.75, None]]",
         "call": "censored_design_bias(D, 10.0, 2)",
         "gold_call": "_oracle_censored_design_bias(D, 10.0, 2)"},
        # normal: nearest-neighbour distances
        {"setup": "D = [[9.07, 1.72, 1.47, 4.42], [7.42, 5.36, 3.48, 6.48], [7.38, 5.53, 4.28, 5.0], [4.28, 3.49, 8.01, 5.5], [5.77, 5.72, 3.88, 3.87], [3.47, 7.87, 3.37, 0.99], [4.64, 4.64, 7.73, None], [1.29, 3.82, 2.84, 2.72], [None, None, 3.96, 8.88], [4.65, 5.07, 5.04, 1.96], [None, 2.62, 7.11, None], [None, 5.82, None, 4.06], [None, 9.13, 9.82, 9.21], [None, 2.93, 4.16, 2.51], [2.1, 5.99, 1.59, 0.62], [9.94, None, 1.49, 2.17], [7.59, 1.27, 4.49, None], [None, 6.95, 5.12, 7.45], [6.41, 3.38, 6.09, None], [None, 5.02, 2.46, 8.98]]",
         "call": "censored_design_bias(D, 10.0, 1)",
         "gold_call": "_oracle_censored_design_bias(D, 10.0, 1)"},
        # edge: third-nearest distances with a 12 m radius
        {"setup": "D = [[11.15, 8.57, 4.46, 4.9], [6.1, 6.53, 7.84, None], [7.16, 7.28, 6.24, 4.78], [None, 5.55, 4.21, 6.08], [5.48, 11.53, None, 8.5], [11.89, 9.65, None, 6.93], [9.9, 10.24, 4.61, 5.58], [10.27, 5.48, 4.87, 10.02], [11.91, 7.5, 9.46, None], [10.28, 9.91, None, None], [4.76, 5.99, 5.58, 5.89], [None, 11.94, 7.17, 5.91], [6.96, 8.77, 5.11, 6.3], [None, None, None, 6.17], [5.96, 5.9, 3.06, 7.24]]",
         "call": "censored_design_bias(D, 12.0, 3)",
         "gold_call": "_oracle_censored_design_bias(D, 12.0, 3)"},
        # invalid: strongly aggregated sample whose fitted k is below 1
        _expect_value_error("censored_design_bias([[None, None, None, 8.6], [9.44, None, 5.74, None], [3.54, 2.84, 7.87, None], [7.02, None, 8.55, 4.43], [None, 9.93, 7.23, None], [8.13, 4.93, 5.62, 7.96], [3.84, 6.88, 2.21, 5.54], [4.67, 9.83, None, 1.88], [7.35, None, None, 7.27], [7.26, 5.42, 4.84, 2.75], [None, None, None, None], [None, None, 6.83, 5.85], [4.41, 3.91, None, 4.18], [6.91, 5.41, None, 7.03], [3.22, 6.58, None, 1.22], [3.75, None, 9.47, 3.78], [None, 3.63, 4.08, 8.34], [3.95, 3.55, 1.82, 2.09], [6.86, 2.38, 6.03, 6.2], [None, 9.51, 9.02, None]], 10.0, 2)",
                           "_oracle_censored_design_bias([[None, None, None, 8.6], [9.44, None, 5.74, None], [3.54, 2.84, 7.87, None], [7.02, None, 8.55, 4.43], [None, 9.93, 7.23, None], [8.13, 4.93, 5.62, 7.96], [3.84, 6.88, 2.21, 5.54], [4.67, 9.83, None, 1.88], [7.35, None, None, 7.27], [7.26, 5.42, 4.84, 2.75], [None, None, None, None], [None, None, 6.83, 5.85], [4.41, 3.91, None, 4.18], [6.91, 5.41, None, 7.03], [3.22, 6.58, None, 1.22], [3.75, None, 9.47, 3.78], [None, 3.63, 4.08, 8.34], [3.95, 3.55, 1.82, 2.09], [6.86, 2.38, 6.03, 6.2], [None, 9.51, 9.02, None]], 10.0, 2)"),
        # invalid: ragged survey
        _expect_value_error("censored_design_bias([[1.0, 2.0], [3.0]], 10.0, 1)",
                           "_oracle_censored_design_bias([[1.0, 2.0], [3.0]], 10.0, 1)"),
    ]
