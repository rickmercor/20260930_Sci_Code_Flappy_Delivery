"""
Apply the higher-order likelihood-root correction and the discovery convention to obtain the corrected significance.

The signed likelihood-ratio root $r(0)$ of the background-only test is asymptotically standard Gaussian, and the first-order discovery significance is $Z = \sqrt{q_0} = \max\{0, r(0)\}$, zero whenever the data show no excess. At small expected counts the Gaussian approximation to $r(0)$ incurs non-negligible errors, especially in the far tail that matters for discovery. The higher-order correction of Barndorff-Nielsen refines the root as

$$r^*(0) = r(0) + \frac{1}{r(0)} \ln \frac{u(0)}{r(0)} ,$$

where $u(0)$ is the model-dependent auxiliary statistic that approaches $r(0)$ in the large-sample limit, so that $r^*$ reverts to $r$ there. The corrected discovery statistic is $q_0^* = [\max\{0, r^*(0)\}]^2$, so the requirement for discovery-like evidence is applied to $r^*(0)$ rather than to $r(0)$, and the corrected significance is $Z = \sqrt{q_0^*} = \max\{0, r^*(0)\}$.

At the sample-space boundaries the auxiliary statistic is zero and the logarithmic adjustment is undefined. The same happens when the root itself vanishes. In both situations the adjustment is dropped and $r^*(0) = r(0)$ is adopted, so that the statistic stays well defined numerically. The auxiliary statistic and the root always share their sign for the counting models considered, so a pair of opposite signs indicates inconsistent inputs.

Returns
-------
float, the corrected discovery significance max(0, r*), with r* = r + ln(u/r)/r and r* = r when r or u is zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def corrected_discovery_significance(r: float, u: float) -> float:
    r"""Return the discovery significance max(0, r*) from a signed root and its auxiliary statistic.

    Parameters
    ----------
    r : float
        Signed likelihood-ratio root r(0) of the background-only test, finite.
    u : float
        Auxiliary statistic u(0) of the higher-order correction, finite. It is
        zero at a sample-space boundary and otherwise has the sign of r.

    Returns
    -------
    z : float
        The corrected discovery significance as a native Python float, the
        larger of zero and r*, with r* = r + ln(u / r) / r. When u is zero
        or r is zero the adjustment is undefined and r* equals r.

    Raises
    ------
    ValueError
        If r or u is not finite, or if r and u are both non-zero with
        opposite signs, for which the logarithm is undefined.
    """
    return z

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math


def _oracle_corrected_discovery_significance(r: float, u: float) -> float:
    r = float(r)
    u = float(u)
    if not (math.isfinite(r) and math.isfinite(u)):
        raise ValueError("r and u must be finite")
    if r == 0.0 or u == 0.0:
        # The logarithmic adjustment is undefined, so r* = r.
        r_star = r
    else:
        if (r > 0.0) != (u > 0.0):
            raise ValueError("r and u must share their sign")
        r_star = r + math.log(u / r) / r
    return float(max(0.0, r_star))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "corrected_discovery_significance"),
                           ("run_gold", "_oracle_corrected_discovery_significance")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    return [
        # the benchmark Asimov pair, where the auxiliary statistic is below the root
        {
            "setup": "r, u = 2.2143554, 1.8295505\n",
            "call": "corrected_discovery_significance(r, u)",
            "gold_call": "_oracle_corrected_discovery_significance(r, u)",
        },
        # an auxiliary statistic above the root, which raises the significance
        {
            "setup": "r, u = 1.4, 1.65\n",
            "call": "corrected_discovery_significance(r, u)",
            "gold_call": "_oracle_corrected_discovery_significance(r, u)",
        },
        # a deficit, both quantities negative, where the corrected root stays negative
        {
            "setup": "r, u = -1.3, -1.1\n",
            "call": "corrected_discovery_significance(r, u)",
            "gold_call": "_oracle_corrected_discovery_significance(r, u)",
        },
        # boundary: an auxiliary statistic of zero, where the adjustment is dropped
        {
            "setup": "r, u = 1.7, 0.0\n",
            "call": "corrected_discovery_significance(r, u)",
            "gold_call": "_oracle_corrected_discovery_significance(r, u)",
        },
        # boundary: a root of zero, where the significance is zero
        {
            "setup": "r, u = 0.0, 0.0\n",
            "call": "corrected_discovery_significance(r, u)",
            "gold_call": "_oracle_corrected_discovery_significance(r, u)",
        },
        # edge: a small positive root whose adjustment drives the corrected root below zero
        {
            "setup": "r, u = 0.2, 0.1\n",
            "call": "corrected_discovery_significance(r, u)",
            "gold_call": "_oracle_corrected_discovery_significance(r, u)",
        },
        # edge: a large-sample pair in which the two statistics nearly coincide
        {
            "setup": "r, u = 4.9987, 4.9991\n",
            "call": "corrected_discovery_significance(r, u)",
            "gold_call": "_oracle_corrected_discovery_significance(r, u)",
        },
        {
            "setup": "# invalid: a root and an auxiliary statistic of opposite sign\n"
                     "args = (1.5, -1.2)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: an auxiliary statistic that is not finite\n"
                     "args = (1.5, float('inf'))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "# invalid: an auxiliary statistic that is not a number\n"
                     "args = (1.5, float('nan'))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
