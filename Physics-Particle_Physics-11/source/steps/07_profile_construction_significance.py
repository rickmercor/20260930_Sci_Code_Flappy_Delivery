"""
Evaluate the exact profile-construction discovery significance of an observed on/off outcome by Poisson tail summation.

The reference against which the asymptotic significances are judged is the profile construction, also called hybrid resampling. For an observed pair $(n, m)$ the background-only $p$-value is found from the distribution of the discovery statistic at $s = 0$ with the background fixed to its profiled value $\hat{\hat{b}}_0 = (n + m)/(1 + \tau)$, so that pseudo-data are distributed as $N \sim \mathrm{Poisson}(\hat{\hat{b}}_0)$ and $M \sim \mathrm{Poisson}(\tau \hat{\hat{b}}_0)$ independently. Rather than sampling toys, the tail probability is summed exactly over the discrete sample space. Outcomes are ordered by the first-order signed likelihood-ratio root $r(0)$, so the $p$-value is the probability of all pairs whose root is at least the observed one,

$$p(n, m) = \sum_{(n', m')\,:\, r(n', m') \ge r(n, m)} \mathrm{Poisson}\big(n'; \hat{\hat{b}}_0\big)\, \mathrm{Poisson}\big(m'; \tau \hat{\hat{b}}_0\big) ,$$

which includes the observed pair itself, in keeping with the convention that a discrete $p$-value contains the probability of the observed outcome. The significance is $Z = \max\{0, \Phi^{-1}(1 - p)\}$, and an outcome with no excess, $n \le m/\tau$, has $Z = 0$ by the discovery convention without any summation.

Each Poisson factor is summed over $0, 1, \ldots$ up to the point where the omitted upper tail has probability below $10^{-12}$, always extending at least to the observed count, so that the truncation changes the $p$-value by no more than the omitted mass. The comparison of the roots on the grid with the observed root has to be guarded against round-off, so that the observed pair is never dropped from its own tail.

Returns
-------
float, the profile-construction discovery significance of the observed count pair from deterministic Poisson tail summation, zero when n <= m/tau.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def profile_construction_significance(n: int, m: int, tau: float) -> float:
    r"""Return the exact profile-construction discovery significance of an observed pair of counts.

    Parameters
    ----------
    n : int
        Observed count in the signal region, an integer of at least zero.
        An integral value carried by a float or a NumPy integer is accepted.
    m : int
        Observed count in the control region, an integer of at least zero.
        An integral value carried by a float or a NumPy integer is accepted.
    tau : float
        Scale factor between the control and signal regions, finite and
        above zero.

    Returns
    -------
    z_reference : float
        The discovery significance as a native Python float, the larger of
        zero and the standard-normal quantile of one minus the background-only
        tail probability of the first-order signed root under the profiled
        null with b fixed to (n + m) / (1 + tau). It is zero when n is at most
        m / tau. Each Poisson sum omits an upper tail of probability below
        1e-12 and extends at least to the observed count.

    Raises
    ------
    ValueError
        If n or m is negative or not an integer value, or if tau is not
        finite or not above zero.
    """
    return z_reference

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from scipy.stats import norm, poisson


def _poisson_grid(mean: float, observed: int) -> "np.ndarray":
    """Support 0..K with K at least the observed count and omitting less than 1e-12 of mass."""
    upper = int(math.ceil(poisson.ppf(1.0 - 1e-12, mean))) if mean > 0.0 else 0
    return np.arange(max(upper, int(observed)) + 1)


def _oracle_profile_construction_significance(n: int, m: int, tau: float) -> float:
    for name, count in (("n", n), ("m", m)):
        value = float(count)
        if not (math.isfinite(value) and value >= 0.0 and value == math.floor(value)):
            raise ValueError(f"{name} must be an integer count of at least zero")
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be a finite scale factor above zero")
    n = int(n)
    m = int(m)

    r_observed = float(_oracle_signed_likelihood_root(np.array(float(n)), np.array(float(m)), tau))
    if r_observed <= 0.0:
        return 0.0

    b_null = _oracle_profile_background_estimate(n, m, tau, 0.0)
    n_grid = _poisson_grid(b_null, n)
    m_grid = _poisson_grid(tau * b_null, m)
    probability = poisson.pmf(n_grid, b_null)[:, None] * poisson.pmf(m_grid, tau * b_null)[None, :]
    roots = _oracle_signed_likelihood_root(n_grid[:, None], m_grid[None, :], tau)
    # The observed pair belongs to its own tail whatever the round-off of its root.
    in_tail = roots >= r_observed - 1e-12 * max(1.0, r_observed)
    p_value = float(probability[in_tail].sum())
    return float(max(0.0, norm.isf(p_value)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "profile_construction_significance"),
                           ("run_gold", "_oracle_profile_construction_significance")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    return [
        # a moderate excess near the benchmark Asimov point
        {
            "setup": "n, m, tau = 7, 1, 1.25\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # a larger excess with an equal-size control region
        {
            "setup": "n, m, tau = 12, 3, 1.0\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # an excess of one count above the background expectation, where the tail is heavy
        {
            "setup": "n, m, tau = 3, 4, 2.0\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # boundary: an empty control region, where the profiled background comes from n alone
        {
            "setup": "n, m, tau = 5, 0, 1.25\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # boundary: counts exactly at the background expectation, no excess
        {
            "setup": "n, m, tau = 4, 8, 2.0\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # edge: a deficit in the signal region, which carries no evidence for a signal
        {
            "setup": "n, m, tau = 2, 9, 1.0\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # edge: larger counts, where the reference approaches the Gaussian approximation
        {
            "setup": "n, m, tau = 45, 60, 2.0\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # edge: counts of a few hundred, where the null grid holds tens of thousands of outcomes
        {
            "setup": "n, m, tau = 260, 400, 1.5\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # edge: an excess near four standard deviations, where a tail coarser than the documented 1e-12 shifts the value
        {
            "setup": "n, m, tau = 20, 4, 1.25\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # edge: an empty control region with an excess near four standard deviations
        {
            "setup": "n, m, tau = 12, 0, 1.0\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # boundary: integral counts supplied as floats, which the contract accepts
        {
            "setup": "n, m, tau = 7.0, 1.0, 1.25\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        # boundary: counts supplied as NumPy integers
        {
            "setup": "import numpy as np\nn, m, tau = np.int64(12), np.int64(3), 1.0\n",
            "call": "profile_construction_significance(n, m, tau)",
            "gold_call": "_oracle_profile_construction_significance(n, m, tau)",
            "tol": 1e-8,
        },
        {
            "setup": "# invalid: a signal-region count that is not an integer\n"
                     "args = (5.5, 1, 1.25)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative control-region count\n"
                     "args = (5, -1, 1.25)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a scale factor of zero\n"
                     "args = (5, 1, 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: a negative scale factor\n"
                     "args = (5, 1, -1.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
