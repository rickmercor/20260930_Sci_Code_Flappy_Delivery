"""
Compute the profiled background estimate of the on/off counting experiment for a fixed signal strength.

A search counts $n$ events in a signal region, modelled as Poisson with mean $s + b$, and constrains the unknown background $b$ with a control region that counts $m$ events, Poisson with mean $\tau b$ for a known scale factor $\tau$. The likelihood is the product of the two Poisson terms, $L(s, b) = \frac{(s+b)^n}{n!} e^{-(s+b)} \frac{(\tau b)^m}{m!} e^{-\tau b}$. The profile likelihood ratio needs the conditional maximum-likelihood estimate of $b$ for a fixed value of $s$, written $\hat{\hat{b}}(s)$, which is the non-negative root of the quadratic obtained by setting $\partial \ln L / \partial b$ to zero,

$$\hat{\hat{b}}(s) = \frac{A + \sqrt{A^2 + 4 (1 + \tau) s m}}{2 (1 + \tau)}, \qquad A = n + m - (1 + \tau) s .$$

For the background-only hypothesis $s = 0$ this reduces to $\hat{\hat{b}}_0 = (n + m)/(1 + \tau)$, the pooled estimate of both regions. The counts may be real-valued, because the same estimate is evaluated on the Asimov data set in which $n$ and $m$ are replaced by their expectation values. When the tested signal strength far exceeds the observed counts, $A$ is large and negative and the two terms of the numerator nearly cancel, so the root has to be evaluated in a form that keeps its full relative accuracy.

Returns
-------
float, the nonnegative conditional maximum-likelihood background estimate for the supplied counts, scale factor and fixed signal strength.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def profile_background_estimate(n: float, m: float, tau: float, s: float) -> float:
    r"""Return the conditional maximum-likelihood background for a fixed signal strength.

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
        zero, so that the control count has mean tau times the background.
    s : float
        Signal strength at which the background is profiled, finite and at
        least zero.

    Returns
    -------
    b_profiled : float
        The conditional estimate of the background for the given s, the
        non-negative root of the profiling condition, as a native Python
        float, accurate to a relative error of 1e-9 for every valid input,
        including signal strengths far above n + m. For s equal to zero it
        is the pooled estimate (n + m) / (1 + tau).

    Raises
    ------
    ValueError
        If n or m is negative or not finite, if tau is not finite or not above
        zero, or if s is negative or not finite.
    """
    return b_profiled

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math


def _oracle_profile_background_estimate(n: float, m: float, tau: float, s: float) -> float:
    n = float(n)
    m = float(m)
    tau = float(tau)
    s = float(s)
    if not (math.isfinite(n) and n >= 0.0):
        raise ValueError("n must be a finite count of at least zero")
    if not (math.isfinite(m) and m >= 0.0):
        raise ValueError("m must be a finite count of at least zero")
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be a finite scale factor above zero")
    if not (math.isfinite(s) and s >= 0.0):
        raise ValueError("s must be a finite signal strength of at least zero")

    linear = n + m - (1.0 + tau) * s
    # The discriminant is a square plus a non-negative term, so the root is real.
    root = math.sqrt(linear * linear + 4.0 * (1.0 + tau) * s * m)
    if linear >= 0.0:
        return float((linear + root) / (2.0 * (1.0 + tau)))
    # For a negative linear term the sum would cancel, so use the conjugate form,
    # which is the same root written as 2 s m / (root - linear).
    return float(2.0 * s * m / (root - linear))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "profile_background_estimate"),
                           ("run_gold", "_oracle_profile_background_estimate")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    return [
        # typical observed counts profiled at a positive signal strength
        {
            "setup": "n, m, tau, s = 9, 4, 1.5, 2.0\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # the background-only hypothesis, where the estimate is the pooled count over 1 + tau
        {
            "setup": "n, m, tau, s = 7, 3, 2.0, 0.0\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # the benchmark Asimov data set, real-valued counts at the background-only hypothesis
        {
            "setup": "n, m, tau, s = 5.8, 1.0, 1.25, 0.0\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # boundary: an empty control region, where the root collapses to max(A, 0) / (1 + tau)
        {
            "setup": "n, m, tau, s = 6, 0, 1.0, 2.0\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # boundary: an empty control region with a signal strength above the observed count
        {
            "setup": "n, m, tau, s = 2, 0, 1.0, 5.0\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # edge: both regions empty, where the estimate is zero for any signal strength
        {
            "setup": "n, m, tau, s = 0, 0, 3.0, 1.5\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # edge: a large signal hypothesis that pulls the profiled background well below the pooled value
        {
            "setup": "n, m, tau, s = 12, 30, 0.5, 40.0\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # edge: a signal hypothesis far above the observed counts, where the root formula cancels
        {
            "setup": "n, m, tau, s = 50, 20, 0.5, 5.0e9\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # edge: the same cancellation with equal-size regions
        {
            "setup": "n, m, tau, s = 30, 10, 1.0, 2.0e9\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # edge: the same cancellation with a control region three times the signal region
        {
            "setup": "n, m, tau, s = 9, 40, 3.0, 1.0e9\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        # edge: the same cancellation with a control region a quarter of the signal region
        {
            "setup": "n, m, tau, s = 60, 8, 0.25, 2.0e9\n",
            "call": "profile_background_estimate(n, m, tau, s)",
            "gold_call": "_oracle_profile_background_estimate(n, m, tau, s)",
        },
        {
            "setup": "# invalid: a scale factor of zero, for which the control region carries no information\n"
                     "args = (5, 3, 0.0, 1.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative signal-region count\n"
                     "args = (-1, 3, 1.0, 1.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative signal strength\n"
                     "args = (5, 3, 1.0, -0.5)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative scale factor\n"
                     "args = (5, 3, -2.0, 1.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
