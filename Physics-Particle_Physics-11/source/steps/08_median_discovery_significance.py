"""
Compute the exact median discovery significance of the on/off experiment for a nominal signal strength.

The sensitivity of a planned search is characterised by the median of the discovery significance under the assumption that the signal is present with strength $s$. Because the $p$-value and the significance are related by a monotonic map, the median significance is the significance of the median $p$-value, which is why the median rather than the mean is reported. The observed pair $(n, m)$ is distributed as $N \sim \mathrm{Poisson}(s + b)$ and $M \sim \mathrm{Poisson}(\tau b)$ independently, and each outcome carries the exact profile-construction significance $Z(n, m)$ of the plug-in null with the background profiled at $s = 0$. The distribution of $Z$ is discrete, and its median is the smallest attainable value $z$ for which

$$P(Z \le z) = \sum_{(n, m)\,:\, Z(n, m) \le z} \mathrm{Poisson}(n; s + b)\, \mathrm{Poisson}(m; \tau b) \ge \tfrac{1}{2} .$$

This is the quantity that toy Monte Carlo estimates approach as the number of pseudo-experiments grows, and the structure it shows as a function of $b$ comes from the discreteness of the data. The two Poisson sums over the outcomes are each truncated where the omitted upper tail has probability below $10^{-12}$, and the probabilities are used as they are without renormalisation, so the truncation cannot move the median.

Returns
-------
float, the smallest attainable profile-construction significance whose cumulative probability under the nominal joint Poisson distribution reaches one half.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def median_discovery_significance(s: float, b: float, tau: float) -> float:
    r"""Return the exact median of the profile-construction discovery significance for a nominal signal.

    Parameters
    ----------
    s : float
        Nominal expected number of signal events, finite and at least zero.
    b : float
        Expected number of background events in the signal region, finite
        and above zero.
    tau : float
        Scale factor between the control and signal regions, finite and
        above zero.

    Returns
    -------
    z_median : float
        The median discovery significance as a native Python float, the
        smallest attainable significance whose cumulative probability under
        the joint Poisson distribution of the two counts reaches one half.
        Each Poisson sum over the outcomes omits an upper tail of probability
        below 1e-12.

    Raises
    ------
    ValueError
        If s is negative or not finite, or if b or tau is not finite or not
        above zero.
    """
    return z_median

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from scipy.stats import poisson


def _oracle_median_discovery_significance(s: float, b: float, tau: float) -> float:
    s = float(s)
    b = float(b)
    tau = float(tau)
    if not (math.isfinite(s) and s >= 0.0):
        raise ValueError("s must be a finite signal strength of at least zero")
    if not (math.isfinite(b) and b > 0.0):
        raise ValueError("b must be a finite background above zero")
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be a finite scale factor above zero")

    n_grid = _poisson_grid(s + b, 0)
    m_grid = _poisson_grid(tau * b, 0)
    probability = (poisson.pmf(n_grid, s + b)[:, None] * poisson.pmf(m_grid, tau * b)[None, :]).ravel()
    significance = np.array([
        _oracle_profile_construction_significance(int(n), int(m), tau)
        for n in n_grid for m in m_grid
    ])
    # Accumulate probability in increasing order of Z and stop at the first outcome
    # whose cumulative probability reaches one half.
    order = np.argsort(significance, kind="stable")
    cumulative = np.cumsum(probability[order])
    index = int(np.searchsorted(cumulative, 0.5, side="left"))
    return float(significance[order][min(index, len(order) - 1)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "median_discovery_significance"),
                           ("run_gold", "_oracle_median_discovery_significance")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    return [
        # the benchmark configuration, s = 5, b = 0.8 and a control region 1.25 times the signal region
        {
            "setup": "s, b, tau = 5.0, 0.8, 1.25\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # a small signal over a unit background with a large control region
        {
            "setup": "s, b, tau = 2.0, 1.0, 4.0\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # a configuration with a fractional expected control count
        {
            "setup": "s, b, tau = 4.5, 1.25, 1.25\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # boundary: no signal, where at least half of the outcomes show no excess
        {
            "setup": "s, b, tau = 0.0, 0.8, 1.25\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # edge: a control region smaller than the signal region
        {
            "setup": "s, b, tau = 6.0, 2.0, 0.5\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # edge: a background well below one expected event
        {
            "setup": "s, b, tau = 3.0, 0.2, 5.0\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # edge: no signal at a small scale factor, where the median is carried by an empty-control outcome and is not zero
        {
            "setup": "s, b, tau = 0.0, 3.0, 0.1\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # edge: a small control region, where the median outcome has an empty control region
        {
            "setup": "s, b, tau = 3.0, 1.0, 0.3\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # edge: no signal with a control region a fifth of the signal region, where the median is zero by a narrow margin
        {
            "setup": "s, b, tau = 0.0, 5.0, 0.2\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        # edge: expected counts of a few tens, where thousands of outcomes each need their own tail sum
        {
            "setup": "s, b, tau = 12.0, 10.0, 2.0\n",
            "call": "median_discovery_significance(s, b, tau)",
            "gold_call": "_oracle_median_discovery_significance(s, b, tau)",
            "tol": 1e-8,
        },
        {
            "setup": "# invalid: a background of zero, for which the control count has no expectation\n"
                     "args = (5.0, 0.0, 1.25)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative signal strength\n"
                     "args = (-0.5, 0.8, 1.25)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a scale factor of zero\n"
                     "args = (5.0, 0.8, 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a signal strength that is not finite\n"
                     "args = (float('inf'), 0.8, 1.25)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
