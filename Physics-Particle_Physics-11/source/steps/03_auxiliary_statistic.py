"""
Evaluate the auxiliary statistic of the higher-order likelihood-root correction for the on/off counting experiment.

The signed likelihood-ratio root $r(s)$ follows a standard Gaussian only in the large-sample limit. The higher-order correction of Barndorff-Nielsen, $r^*(s) = r(s) + \frac{1}{r(s)} \ln \frac{u(s)}{r(s)}$, brings its distribution closer to a standard Gaussian at small counts. The auxiliary statistic $u(s)$ is the model-dependent quantity of that correction for the two-Poisson on/off likelihood $L(s, b)$ of a signal-region count $n$ with mean $s + b$ and a control count $m$ with mean $\tau b$, treated as a full exponential family with the signal strength $s$ as the parameter of interest and the background $b$ as the nuisance parameter. It is built from the derivatives of the log-likelihood with respect to the maximum-likelihood estimate $\hat{s}$, through the change of that derivative between the constrained estimate $\hat{\hat{b}}(s)$ of the tested signal strength and the unconstrained estimate $\hat{b} = m/\tau$, standardised by the observed information of the two-parameter fit (Eqs. 35 and 36 of the source paper, and Refs. 17 to 19 therein for the general construction). It is a closed-form expression in $n$, $m$, $\tau$, $s$ and $\hat{\hat{b}}(s)$, and for the background-only hypothesis $s = 0$, where $\hat{\hat{b}}_0 = (n + m)/(1 + \tau)$, it collapses to a single logarithm times a count-dependent prefactor.

The statistic has the sign of $r(s)$ and approaches $r(s)$ in the large-sample limit. At the sample-space boundaries $u(s)$ is assigned the continuous limit of the expression. An empty signal region gives zero for every $s$, and for the background-only hypothesis an empty control region gives zero as well, through $\sqrt{x} \ln x \to 0$, which marks the boundary at which the logarithmic adjustment of $r^*$ is undefined. For a positive tested signal strength the empty-control limit depends on where $n$ sits relative to $(1 + \tau) s$, because the profiled background then vanishes linearly with $m$ when $n < (1 + \tau) s$ and stays finite when $n > (1 + \tau) s$. The limit is zero for $n > (1 + \tau) s$, $\sqrt{n} \ln(n/s)$ for $n < (1 + \tau) s$, and $\sqrt{n/2} \ln(n/s)$ at equality. The counts may be real-valued, because the same statistic is evaluated on the Asimov data set.

Returns
-------
float, the signed auxiliary statistic u(s) of the on/off likelihood, using the specified continuous limits at zero-count boundaries.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def auxiliary_statistic(n: float, m: float, tau: float, s: float) -> float:
    r"""Return the auxiliary statistic u(s) of the higher-order correction for the on/off model.

    Parameters
    ----------
    n : float
        Count in the signal region, finite and at least zero. Real values are
        allowed for Asimov data.
    m : float
        Count in the control region, finite and at least zero. Real values are
        allowed for Asimov data.
    tau : float
        Scale factor between the control and signal regions, finite and above
        zero.
    s : float
        Tested signal strength, finite and at least zero. The background is
        profiled at this value.

    Returns
    -------
    u : float
        The auxiliary statistic u(s) as a native Python float. It carries the
        sign of the signed likelihood-ratio root r(s). It is zero when the
        signal region is empty, when the control region is empty at s equal to
        zero or with n above (1 + tau) s, and when the signal region sits
        exactly at its expectation under the tested hypothesis. For an empty
        control region with positive s it is sqrt(n) ln(n / s) when n is
        below (1 + tau) s and sqrt(n / 2) ln(n / s) when n equals (1 + tau) s
        exactly.

    Raises
    ------
    ValueError
        If n or m is negative or not finite, if tau is not finite or not above
        zero, or if s is negative or not finite.
    """
    return u

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math


def _oracle_auxiliary_statistic(n: float, m: float, tau: float, s: float) -> float:
    # The profiling step validates every argument and raises for invalid data.
    b = _oracle_profile_background_estimate(n, m, tau, s)
    n = float(n)
    m = float(m)
    tau = float(tau)
    s = float(s)
    # Sample-space boundaries take the continuous limit of the expression. An empty
    # signal region gives zero. An empty control region gives zero when the profiled
    # background stays finite (s = 0 or n > (1 + tau) s); when it vanishes linearly
    # with m the limit is sqrt(n) ln(n / s), and at the crossover sqrt(n / 2) ln(n / s).
    if n == 0.0:
        return 0.0
    if m == 0.0:
        excess = n - (1.0 + tau) * s
        if s == 0.0 or excess > 0.0:
            return 0.0
        if excess < 0.0:
            return float(math.sqrt(n) * math.log(n / s))
        return float(math.sqrt(n / 2.0) * math.log(n / s))
    mean_on = s + b
    scale = math.sqrt(n * m) / math.sqrt(n / mean_on ** 2 + m / b ** 2)
    bracket = math.log(n / mean_on) / b - math.log(m / (tau * b)) / mean_on
    return float(scale * bracket)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "auxiliary_statistic"),
                           ("run_gold", "_oracle_auxiliary_statistic")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    return [
        # the benchmark Asimov data set at the background-only hypothesis
        {
            "setup": "n, m, tau, s = 5.8, 1.0, 1.25, 0.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # observed integer counts with an excess, tested at the background-only hypothesis
        {
            "setup": "n, m, tau, s = 9, 4, 1.5, 0.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # a positive tested signal strength, where the general form with the profiled background applies
        {
            "setup": "n, m, tau, s = 9, 4, 1.5, 2.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # a deficit in the signal region, where the statistic is negative
        {
            "setup": "n, m, tau, s = 2, 9, 1.0, 0.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # boundary: an empty signal region, where the continuous limit gives zero
        {
            "setup": "n, m, tau, s = 0, 5, 1.0, 0.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # boundary: an empty control region with a positive tested signal strength and n above (1 + tau) s
        {
            "setup": "n, m, tau, s = 6, 0, 2.0, 1.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # boundary: an empty control region with n below (1 + tau) s, where the limit is sqrt(n) ln(n / s)
        {
            "setup": "n, m, tau, s = 1, 0, 1.0, 2.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # boundary: an empty control region with a signal region well below the hypothesis
        {
            "setup": "n, m, tau, s = 3, 0, 2.0, 5.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # boundary: an empty control region with n exactly equal to (1 + tau) s
        {
            "setup": "n, m, tau, s = 4, 0, 1.0, 2.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # boundary: the same equality with real-valued inputs
        {
            "setup": "n, m, tau, s = 3.75, 0, 0.5, 2.5\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # edge: counts exactly at the background expectation, where the logarithm vanishes
        {
            "setup": "n, m, tau, s = 6, 12, 2.0, 0.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # edge: counts of order 1e8 with a small relative excess
        {
            "setup": "n, m, tau, s = 100010000, 100000000, 1.0, 0.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # edge: counts of order 1e9 with an excess of ten events
        {
            "setup": "n, m, tau, s = 1000000010, 1000000000, 1.0, 0.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # edge: the same regime with a scale factor above one and a deficit
        {
            "setup": "n, m, tau, s = 1000000000, 1250000010, 1.25, 0.0\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        # edge: a tested signal strength far above the observed counts
        {
            "setup": "n, m, tau, s = 50, 20, 0.5, 5.0e9\n",
            "call": "auxiliary_statistic(n, m, tau, s)",
            "gold_call": "_oracle_auxiliary_statistic(n, m, tau, s)",
        },
        {
            "setup": "# invalid: a scale factor of zero\n"
                     "args = (5, 3, 0.0, 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a control-region count that is not finite\n"
                     "args = (5, float('inf'), 1.0, 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative tested signal strength\n"
                     "args = (5, 3, 1.0, -1.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative scale factor\n"
                     "args = (5, 3, -1.0, 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
