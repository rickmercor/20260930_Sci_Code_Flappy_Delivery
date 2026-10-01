"""
Orchestrate the sensitivity calculation and report the residual of the corrected Asimov estimate against the exact median significance.

For a planned on/off counting experiment specified by the expected signal $s$, the expected background $b$ and the standard deviation $\sigma_b$ of the background estimate from the control measurement, the median discovery significance can be approximated in closed form or computed exactly. The closed forms evaluate a test statistic at the Asimov data set, $n = s + b$ and $m = \tau b$ with $\tau = b/\sigma_b^2$, either the first-order statistic $q_0$ or the higher-order corrected statistic $q_0^*$ built from the modified root $r^*$ of Barndorff-Nielsen. The exact value is the median of the profile-construction significance over the joint Poisson distribution of the two counts. The Asimov estimate involves two independent levels of approximation, the replacement of the median by the value at the expected data and the choice of statistic at that point, and the higher-order correction addresses only the second.

The pipeline evaluates the corrected Asimov significance $Z_A(q_0^*)$ and the exact median $\mathrm{med}[Z \mid s]$ and reports their difference as the residual, positive when the Asimov estimate overstates the sensitivity. The first-order value $Z_A(q_0)$ is returned alongside for comparison.

Returns
-------
tuple of four floats, (corrected Asimov residual, corrected Asimov significance, exact median discovery significance, first-order Asimov significance), in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orchestrate_asimov_residual(s: float, b: float, sigma_b: float) -> tuple:
    r"""Return the residual of the corrected Asimov significance together with its ingredients.

    Parameters
    ----------
    s : float
        Nominal expected number of signal events, finite and at least zero.
    b : float
        Expected number of background events in the signal region, finite
        and above zero.
    sigma_b : float
        Standard deviation of the background estimate inferred from the
        control measurement, finite and above zero. The scale factor of the
        control region is b / sigma_b^2.

    Returns
    -------
    result : tuple
        A tuple of four native Python floats. The first entry is the residual,
        the corrected Asimov significance minus the exact median discovery
        significance. The second is the corrected Asimov significance, the
        third the exact median and the fourth the first-order Asimov
        significance.

    Raises
    ------
    ValueError
        If s is negative or not finite, or if b or sigma_b is not finite or
        not above zero.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_orchestrate_asimov_residual(s: float, b: float, sigma_b: float) -> tuple:
    # The first-order step validates the configuration.
    z_first_order = _oracle_first_order_asimov_significance(s, b, sigma_b)
    z_corrected = _oracle_corrected_asimov_significance(s, b, sigma_b)
    tau = float(b) / (float(sigma_b) * float(sigma_b))
    z_median = _oracle_median_discovery_significance(s, b, tau)
    return (float(z_corrected - z_median), float(z_corrected), float(z_median), float(z_first_order))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "orchestrate_asimov_residual"),
                           ("run_gold", "_oracle_orchestrate_asimov_residual")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    return [
        # the benchmark configuration
        {
            "setup": "s, b, sigma_b = 5.0, 0.8, 0.8\n",
            "call": "orchestrate_asimov_residual(s, b, sigma_b)",
            "gold_call": "_oracle_orchestrate_asimov_residual(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # a configuration with a fractional Asimov control count
        {
            "setup": "s, b, sigma_b = 4.5, 1.25, 1.0\n",
            "call": "orchestrate_asimov_residual(s, b, sigma_b)",
            "gold_call": "_oracle_orchestrate_asimov_residual(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # a configuration in which the corrected estimate lies above the first-order one
        {
            "setup": "s, b, sigma_b = 2.0, 1.0, 0.5\n",
            "call": "orchestrate_asimov_residual(s, b, sigma_b)",
            "gold_call": "_oracle_orchestrate_asimov_residual(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # boundary: no signal, where the corrected root takes its limit at the hypothesis point while the median and the first-order value are zero
        {
            "setup": "s, b, sigma_b = 0.0, 0.8, 0.8\n",
            "call": "orchestrate_asimov_residual(s, b, sigma_b)",
            "gold_call": "_oracle_orchestrate_asimov_residual(s, b, sigma_b)",
            "tol": 1e-6,
        },
        # edge: larger expected counts, where the residual shrinks
        {
            "setup": "s, b, sigma_b = 14.0, 8.0, 2.0\n",
            "call": "orchestrate_asimov_residual(s, b, sigma_b)",
            "gold_call": "_oracle_orchestrate_asimov_residual(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # edge: a control region smaller than the signal region with a well-populated median grid
        {
            "setup": "s, b, sigma_b = 10.0, 6.0, 3.0\n",
            "call": "orchestrate_asimov_residual(s, b, sigma_b)",
            "gold_call": "_oracle_orchestrate_asimov_residual(s, b, sigma_b)",
            "tol": 1e-8,
        },
        {
            "setup": "# invalid: a background uncertainty of zero, outside the uncertain-background model\n"
                     "args = (5.0, 0.8, 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative signal strength\n"
                     "args = (-0.3, 0.8, 0.8)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a background of zero\n"
                     "args = (5.0, 0.0, 0.8)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ] + [{'setup': 's, b, sigma_b = 1e-6, 1.0, 1.0\n', 'call': 'orchestrate_asimov_residual(s, b, sigma_b)', 'gold_call': '_oracle_orchestrate_asimov_residual(s, b, sigma_b)', 'tol': 1e-08}]
