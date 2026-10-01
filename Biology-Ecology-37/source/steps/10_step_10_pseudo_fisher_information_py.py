"""
Expected pseudo-Fisher information about the six demographic parameters from all pairwise comparisons of a sampling design with estimated ages.

Close-kin designs can be judged before any sample is taken. When the population is

large and the sampling fraction small, each pairwise comparison is close to an

independent rare event, so the pseudo-likelihood treats each kinship outcome as a

Poisson count whose mean is the kinship probability. The expected information about

the demographic parameters is then a sum over comparisons, and grouping comparisons

by their covariates, here the estimated ages the laboratory will report, turns it into

a sum over classes weighted by the numbers of comparisons. Self-recaptures on their

own amount to individual mark-recapture in the same framework.

Returns
-------
np.ndarray of shape (6, 6), the expected pseudo-Fisher information about theta
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pseudo_fisher_information(theta: "np.ndarray", ref_year: int, entry_age: int, numbers: "np.ndarray",
                              sample_years: "np.ndarray", first_year: int, use_kin: bool) -> "np.ndarray":
    '''Expected pseudo-Fisher information about the six demographic parameters from all pairwise comparisons of a sampling design with estimated ages.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    as in the second step, and the information is about theta on exactly
    this scale. numbers describes the design by sampling year, sex, true age
    and estimated age, as returned by the fifth step for sample_years; it is
    fixed by the design and does not depend on theta. Each comparison's
    kinship outcome is a Poisson count with mean equal to its kinship
    probability for the two classes of estimated age (the sixth, seventh and
    eighth steps, with first_year the first modelled birth year), in a
    pseudo-likelihood over all compared pairs (the numbers of comparisons
    from the ninth step). Return the expected information of that
    pseudo-likelihood, summed over the mother-offspring, half-sibling and
    self-recapture comparisons when use_kin is True, and over the
    self-recapture comparisons alone when it is False. Derivatives must be
    exact or accurate to a relative 1e-8.

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    entry_age : int
        Age at which females enter the breeding cycle, resting, an integer
        from 1 to 20.
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers, with A + max(sample_years) - min(sample_years) <= 200.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(sample_years).
    use_kin : bool
        True to include the mother-offspring and half-sibling comparisons.

    Returns
    -------
    information : np.ndarray
        Shape (6, 6), symmetric positive semi-definite, entries in the order
        of theta.

    Raises
    ------
    ValueError
        If any argument is invalid as in the sixth to ninth steps, or if
        use_kin is not a bool.
    '''
    return information  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pfi_complex_step(fun, theta: "np.ndarray") -> tuple:
    """Value and exact first derivatives of an array-valued function of theta (complex step, h = 1e-30)."""
    base = np.real(fun(theta.astype(float)))
    grads = []
    for k in range(theta.size):
        th = theta.astype(complex)
        th[k] += 1e-30j
        grads.append(np.imag(fun(th)) / 1e-30)
    return base, np.stack(grads, axis=-1)


def _oracle_pseudo_fisher_information(theta: "np.ndarray", ref_year: int, entry_age: int, numbers: "np.ndarray",
                                      sample_years: "np.ndarray", first_year: int, use_kin: bool) -> "np.ndarray":
    th = np.asarray(theta, dtype=float)
    if th.shape != (6,) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    if not isinstance(use_kin, (bool, np.bool_)):
        raise ValueError("use_kin must be a bool")
    ys = np.asarray(sample_years)
    if ys.ndim != 1 or ys.size == 0 or not np.issubdtype(ys.dtype, np.integer):
        raise ValueError("sample_years must be a non-empty 1-D integer array")
    if isinstance(first_year, bool) or not isinstance(first_year, (int, np.integer)) or not 1900 <= first_year < ys.max():
        raise ValueError("first_year must be an integer from 1900 to 2200 below max(sample_years)")
    J = np.asarray(numbers, dtype=float)
    parts = [("SP", lambda t: _oracle_self_recapture_probabilities(t, ref_year, ys, J))]
    if use_kin:
        parts += [("MOP", lambda t: _oracle_mother_offspring_probabilities(t, ref_year, entry_age, ys, first_year, J)),
                  ("HSP", lambda t: _oracle_half_sibling_probabilities(t, ref_year, ys, first_year, J))]
    info = np.zeros((6, 6))
    for kind, fun in parts:
        n = _oracle_comparison_counts(J, ys, first_year, entry_age, kind)
        p, dp = _pfi_complex_step(fun, th)
        use = (n > 0) & (p > 0)
        d = dp[use] * (np.sqrt(n[use]) / np.sqrt(p[use]))[:, None]      # no overflow when n and p are both tiny
        info += d.T @ d
    return 0.5 * (info + info.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    theta = ("theta = np.array([np.log(48000.0), -0.012, np.log(0.955 / 0.045), np.log(0.85 / 0.15), "
             "np.log(0.16 / 0.84), np.log(0.38 / 0.62)])\n")
    design = ("years = np.array([2014, 2015, 2016, 2017, 2018, 2024, 2025, 2026, 2027])\n"
              "ages = np.arange(1, 38)\n"
              "lam = np.exp(-0.012)\n"
              "rel = np.where(ages < 6, lam ** (-ages) * 0.85 ** (ages - 1), lam ** (-ages) * 0.85 ** 5 * 0.955 ** (ages - 6))\n"
              "rel_m = np.where(ages <= 5, rel, 0.0)\n"
              "share = np.array([rel, rel_m]) / (rel.sum() + rel_m.sum())\n"
              "samples = np.array([800.0] * 5 + [1000.0] * 4)[:, None, None] * share[None, :, :]\n"
              "a = ages[:, None]\ne = ages[None, :]\n"
              "from scipy.stats import norm\n"
              "P = norm.cdf((e + 0.5 - a) / SD) - norm.cdf((e - 0.5 - a) / SD)\nP = P / P.sum(axis=1, keepdims=True)\n"
              "numbers = samples[:, :, :, None] * P[None, None, :, :]\n")
    return [
        # --- normal: all three kinship types in the shipped two-period design with the fitted clock ---
        {
            "setup": "import numpy as np\n" + theta + design.replace("SD", "1.6625615194893861"),
            "call": "pseudo_fisher_information(theta, 2016, 4, numbers, years, 2002, True)",
            "gold_call": "_oracle_pseudo_fisher_information(theta, 2016, 4, numbers, years, 2002, True)",
            "tol": 1e-6,
        },
        # --- boundary: self-recaptures alone, so the breeding parameters carry no information ---
        {
            "setup": "import numpy as np\n" + theta + design.replace("SD", "1.6625615194893861"),
            "call": "pseudo_fisher_information(theta, 2016, 4, numbers, years, 2002, False)",
            "gold_call": "_oracle_pseudo_fisher_information(theta, 2016, 4, numbers, years, 2002, False)",
            "tol": 1e-6,
        },
        # --- edge: three consecutive sampling years of a small, fast-growing population and an imprecise clock ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(3000.0), 0.05, np.log(0.93 / 0.07), np.log(0.7 / 0.3), "
                     "np.log(0.25 / 0.75), np.log(0.5 / 0.5)])\n"
                     "a = np.arange(1, 16)[:, None]\ne = np.arange(1, 16)[None, :]\n"
                     "from scipy.stats import norm\n"
                     "P = norm.cdf((e + 0.5 - a) / 3.0) - norm.cdf((e - 0.5 - a) / 3.0)\nP = P / P.sum(axis=1, keepdims=True)\n"
                     "samples = np.linspace(5.0, 40.0, 3 * 2 * 15).reshape(3, 2, 15)\n"
                     "numbers = samples[:, :, :, None] * P[None, None, :, :]\n",
            "call": "pseudo_fisher_information(theta, 2012, 3, numbers, np.array([2010, 2011, 2012]), 1998, True)",
            "gold_call": "_oracle_pseudo_fisher_information(theta, 2012, 3, numbers, np.array([2010, 2011, 2012]), 1998, True)",
            "tol": 1e-6,
        },
    ]
