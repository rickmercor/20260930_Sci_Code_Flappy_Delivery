"""
Compute the higher-order corrected Asimov median discovery significance of the on/off experiment.

The Asimov estimate of the median discovery significance evaluates the test statistic at the expected data, $n \to s + b$ and $m \to \tau b$, instead of computing the true median of the test-statistic distribution. The choice of statistic at that fixed data set is a separate level of approximation. The first-order statistic $q_0$ gives $Z_A(q_0) = \max\{0, r_A(0)\}$, with $r_A(0)$ the signed likelihood-ratio root at the Asimov point, which for the on/off model is the first-order Asimov significance expressed through the background uncertainty. The corrected statistic $q_0^*$ instead applies the higher-order correction $r^*(0) = r(0) + \frac{1}{r(0)} \ln \frac{u(0)}{r(0)}$ at the same point, with the on/off auxiliary statistic $u(0)$ of the preceding step evaluated on the Asimov counts, and reports $Z_A(q_0^*) = \max\{0, r^*(0)\}$.

At the hypothesis point itself, $s = 0$, the Asimov root and the auxiliary statistic vanish together and the logarithmic adjustment is an indeterminate form. This is an interior point of the sample space, not one of its boundaries, and the modified root has a well-defined continuous limit there as $s \to 0^+$ along the Asimov path $n = s + b$, $m = \tau b$. Following the reference implementation of the method, the corrected root at $s = 0$ is assigned that limit. It depends only on $b$ and $\tau$, is finite, vanishes only for $\tau = 1$, and is positive for $\tau > 1$ and negative for $\tau < 1$, so the corrected significance at zero signal is the larger of zero and this limit. The limit cannot be obtained reliably by evaluating the adjustment at a small positive $s$, because the quotient of two vanishing quantities loses its precision there; it has to be derived from the expansion of the root and the auxiliary statistic in $s$.

The configuration is specified through the expected background $b$ and the standard deviation $\sigma_b$ of its estimate from the control measurement. Because the variance of the estimate $\hat{b} = m/\tau$ is $b/\tau$, the scale factor of the control region is $\tau = b/\sigma_b^2$ and the Asimov control count is $m_A = \tau b = b^2/\sigma_b^2$. The Asimov counts are real-valued and are used as they are. No continuity correction is applied, because the observation of the uncertain-background model is a pair of discrete counts for which no unique analogue of the half-bin shift exists.

Returns
-------
float, the corrected Asimov significance max(0, r*_A), accurate to 1e-8 and using the continuous Asimov-path limit at s = 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def corrected_asimov_significance(s: float, b: float, sigma_b: float) -> float:
    r"""Return the Asimov median significance from the higher-order corrected discovery statistic.

    Parameters
    ----------
    s : float
        Nominal expected number of signal events, finite and at least zero.
    b : float
        Expected number of background events in the signal region, finite
        and above zero.
    sigma_b : float
        Standard deviation of the background estimate inferred from the
        control measurement, finite and above zero.

    Returns
    -------
    z_asimov_corrected : float
        The corrected Asimov significance, the larger of zero and the
        refined root r*(0) evaluated at the Asimov counts n = s + b and
        m = b^2 / sigma_b^2 with scale factor b / sigma_b^2, as a native
        Python float, accurate to 1e-8. For s equal to zero the corrected
        root is its continuous limit as s tends to zero from above along the
        Asimov path, a function of b and the scale factor alone that is
        positive for a scale factor above one, negative below one and zero
        at one, and the returned value is the larger of zero and that limit.

    Raises
    ------
    ValueError
        If s is negative or not finite, or if b or sigma_b is not finite or
        not above zero.
    """
    return z_asimov_corrected

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math


def _oracle_corrected_asimov_significance(s: float, b: float, sigma_b: float) -> float:
    # The first-order step validates the configuration and gives the signed root at
    # the Asimov point, which is non-negative for s >= 0.
    r_asimov = _oracle_first_order_asimov_significance(s, b, sigma_b)
    s = float(s)
    b = float(b)
    sigma_b = float(sigma_b)
    tau = b / (sigma_b * sigma_b)
    if 0.0 < s / b <= 0.01:
        return float(max(0.0, _asimov_small_signal(s, b, sigma_b)[1]))
    if s == 0.0:
        # Both the root and the auxiliary statistic vanish. Expanding r_A and u_A to
        # third order in s gives ln(u_A / r_A) / r_A -> (tau - 1) / (6 sqrt(tau b (1 + tau))),
        # the limit used by the reference implementation at the hypothesis point.
        limit = (tau - 1.0) / (6.0 * math.sqrt(tau * b * (1.0 + tau)))
        return float(max(0.0, limit))
    u_asimov = _oracle_auxiliary_statistic(s + b, tau * b, tau, 0.0)
    return _oracle_corrected_discovery_significance(r_asimov, u_asimov)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "corrected_asimov_significance"),
                           ("run_gold", "_oracle_corrected_asimov_significance")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    return [
        # the benchmark configuration, a control count of one expected event
        {
            "setup": "s, b, sigma_b = 5.0, 0.8, 0.8\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
        },
        # a configuration with a fractional Asimov control count above one
        {
            "setup": "s, b, sigma_b = 4.5, 1.25, 1.0\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
        },
        # a configuration in which the correction raises the significance
        {
            "setup": "s, b, sigma_b = 2.0, 1.0, 0.5\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
        },
        # large expected counts, where the correction is small
        {
            "setup": "s, b, sigma_b = 30.0, 40.0, 2.0\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
        },
        # boundary: no signal with a scale factor above one, where the corrected root takes its positive limit
        {
            "setup": "s, b, sigma_b = 0.0, 0.8, 0.8\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # boundary: no signal with a control region three times the signal region
        {
            "setup": "s, b, sigma_b = 0.0, 3.0, 1.0\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # boundary: no signal with a scale factor below one, where the limit is negative and the significance zero
        {
            "setup": "s, b, sigma_b = 0.0, 2.0, 2.0\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # boundary: no signal with equal-size regions, where the limit vanishes
        {
            "setup": "s, b, sigma_b = 0.0, 1.0, 1.0\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # edge: a control region much smaller than the signal region
        {
            "setup": "s, b, sigma_b = 6.0, 2.0, 2.5\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
        },
        # edge: a background uncertainty one millionth of the background, where the control count is enormous
        {
            "setup": "s, b, sigma_b = 5.0, 0.8, 8.0e-7\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # edge: a background uncertainty one billionth of the background, the known-background limit of the corrected value
        {
            "setup": "s, b, sigma_b = 5.0, 0.8, 8.0e-10\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # edge: a large signal over a small background with a tiny background uncertainty
        {
            "setup": "s, b, sigma_b = 30.0, 0.4, 4.0e-7\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # edge: a small signal over a large background with a tiny background uncertainty
        {
            "setup": "s, b, sigma_b = 3.0, 5000.0, 5.0e-4\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
            "tol": 1e-8,
        },
        # edge: large expected counts, where the correction is tiny but the value must still be exact
        {
            "setup": "s, b, sigma_b = 300.0, 2000.0, 10.0\n",
            "call": "corrected_asimov_significance(s, b, sigma_b)",
            "gold_call": "_oracle_corrected_asimov_significance(s, b, sigma_b)",
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
        {
            "setup": "# invalid: a negative background uncertainty\n"
                     "args = (5.0, 0.8, -0.8)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ] + [{'setup': 's, b, sigma_b = 1e-6, 1.0, 1.0\n', 'call': 'corrected_asimov_significance(s, b, sigma_b)', 'gold_call': '_oracle_corrected_asimov_significance(s, b, sigma_b)', 'tol': 1e-08}, {'setup': 's, b, sigma_b = 1e-4, 1.0, 1.0\n', 'call': 'corrected_asimov_significance(s, b, sigma_b)', 'gold_call': '_oracle_corrected_asimov_significance(s, b, sigma_b)', 'tol': 1e-08}, {'setup': 's, b, sigma_b = 1e-7, 0.8, 0.8\n', 'call': 'corrected_asimov_significance(s, b, sigma_b)', 'gold_call': '_oracle_corrected_asimov_significance(s, b, sigma_b)', 'tol': 1e-08}]
