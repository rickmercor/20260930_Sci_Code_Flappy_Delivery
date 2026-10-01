"""
Expected precision of a sampling design with estimated ages for adult female abundance, adult survival and the calving proportion, and the expected numbers of kin pairs of each type.

Inverting the expected information gives the covariance a design can expect for the

parameter estimates, and the delta method carries it to any quantity of interest:

abundance in a chosen year, a survival rate, or a derived breeding quantity. Running

the same calculation with self-recaptures alone shows what the kinship comparisons

add over individual mark-recapture. The expected numbers of kin pairs of each type are

the direct check a design needs before sampling starts, since they are what the

genotyping will find.

Returns
-------
np.ndarray of shape (7,), the two abundance CVs, the survival and calf-proportion standard errors and the three expected kin-pair numbers
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def design_precision(theta: "np.ndarray", ref_year: int, entry_age: int, numbers: "np.ndarray",
                     sample_years: "np.ndarray", first_year: int, target_year: int) -> "np.ndarray":
    '''Expected precision of a sampling design with estimated ages for adult female abundance, adult survival and the calving proportion, and the expected numbers of kin pairs of each type.

    Arguments are as in the tenth step, with target_year the year of the
    abundance of interest. The covariance of the estimates is the inverse of
    the expected information of the tenth step; with self-recaptures alone
    only ln N_ref, r, logit phi_A and logit phi_J are estimated, so the
    inverse of that 4 x 4 block is used. Standard errors of derived
    quantities follow from the delta method, and the CV of an abundance
    estimate is its standard error divided by the abundance. Expected numbers
    of kin pairs are sums, over classes of estimated age, of numbers of
    comparisons (ninth step) times kinship probabilities (sixth to eighth
    steps).

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
    target_year : int
        Year of the adult female abundance of interest, an integer from 1900
        to 2200.

    Returns
    -------
    precision : np.ndarray
        Shape (7,): [0] the expected CV of adult female abundance in
        target_year from all three kinship types; [1] the same CV from
        self-recaptures alone; [2] the expected standard error of adult
        survival phi_A and [3] of the long-run proportion of adult females
        with a calf, both from all three kinship types; [4], [5] and [6] the
        expected numbers of mother-offspring pairs, half-sibling pairs and
        self-recaptures among the compared pairs.

    Raises
    ------
    ValueError
        If any argument is invalid as in the tenth step, if target_year is
        not an integer from 1900 to 2200, or if either information matrix
        used is not positive definite.
    '''
    return precision  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _dp_inverse(info: "np.ndarray") -> "np.ndarray":
    """Inverse of a positive definite information matrix, or ValueError."""
    try:
        np.linalg.cholesky(info)
    except np.linalg.LinAlgError:
        raise ValueError("the information matrix is not positive definite") from None
    return np.linalg.inv(info)


def _oracle_design_precision(theta: "np.ndarray", ref_year: int, entry_age: int, numbers: "np.ndarray",
                             sample_years: "np.ndarray", first_year: int, target_year: int) -> "np.ndarray":
    if isinstance(target_year, bool) or not isinstance(target_year, (int, np.integer)) or not 1900 <= target_year <= 2200:
        raise ValueError("target_year must be an integer from 1900 to 2200")
    th = np.asarray(theta, dtype=float)
    info_all = _oracle_pseudo_fisher_information(th, ref_year, entry_age, numbers, sample_years, first_year, True)
    info_sp = _oracle_pseudo_fisher_information(th, ref_year, entry_age, numbers, sample_years, first_year, False)
    cov = _dp_inverse(info_all)
    cov_sp = _dp_inverse(info_sp[:4, :4])
    grad_n = np.array([1.0, float(target_year - ref_year), 0.0, 0.0, 0.0, 0.0])   # of ln N(target_year)
    cv_all = np.sqrt(grad_n @ cov @ grad_n)
    cv_sp = np.sqrt(grad_n[:4] @ cov_sp @ grad_n[:4])
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    se_phi_a = phi_a * (1.0 - phi_a) * np.sqrt(cov[2, 2])
    grad_b = np.zeros(6)
    for k in (4, 5):
        t = th.astype(complex)
        t[k] += 1e-30j
        grad_b[k] = np.imag(_oracle_breeding_cycle_fecundity(1.0 / (1.0 + np.exp(-t[4])), 1.0 / (1.0 + np.exp(-t[5])),
                                                             entry_age, np.array([0]))[0]) / 1e-30
    se_beta = np.sqrt(grad_b @ cov @ grad_b)
    ys = np.asarray(sample_years)
    J = np.asarray(numbers, dtype=float)
    expected = [
        np.sum(_oracle_comparison_counts(J, ys, first_year, entry_age, "MOP")
               * _oracle_mother_offspring_probabilities(th, ref_year, entry_age, ys, first_year, J)),
        np.sum(_oracle_comparison_counts(J, ys, first_year, entry_age, "HSP")
               * _oracle_half_sibling_probabilities(th, ref_year, ys, first_year, J)),
        np.sum(_oracle_comparison_counts(J, ys, first_year, entry_age, "SP")
               * _oracle_self_recapture_probabilities(th, ref_year, ys, J)),
    ]
    return np.array([cv_all, cv_sp, se_phi_a, se_beta] + expected, dtype=float)

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
              "samples = np.array([800.0] * 5 + [NEW] * 4)[:, None, None] * share[None, :, :]\n"
              "a = ages[:, None]\ne = ages[None, :]\n"
              "from scipy.stats import norm\n"
              "P = norm.cdf((e + 0.5 - a) / 1.6625615194893861) - norm.cdf((e - 0.5 - a) / 1.6625615194893861)\n"
              "P = P / P.sum(axis=1, keepdims=True)\n"
              "numbers = samples[:, :, :, None] * P[None, None, :, :]\n")
    return [
        # --- normal: a two-period design with 600 samples per new year, abundance in the last year ---
        {
            "setup": "import numpy as np\n" + theta + design.replace("NEW", "600.0"),
            "call": "design_precision(theta, 2016, 4, numbers, years, 2002, 2027)",
            "gold_call": "_oracle_design_precision(theta, 2016, 4, numbers, years, 2002, 2027)",
            "tol": 1e-6,
        },
        # --- boundary: abundance of interest in the reference year itself ---
        {
            "setup": "import numpy as np\n" + theta + design.replace("NEW", "1500.0"),
            "call": "design_precision(theta, 2016, 4, numbers, years, 2002, 2016)",
            "gold_call": "_oracle_design_precision(theta, 2016, 4, numbers, years, 2002, 2016)",
            "tol": 1e-6,
        },
        # --- edge: four consecutive years of a small, fast-growing population, projected ten years ahead ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(3000.0), 0.05, np.log(0.93 / 0.07), np.log(0.7 / 0.3), "
                     "np.log(0.25 / 0.75), np.log(0.5 / 0.5)])\n"
                     "a = np.arange(1, 16)[:, None]\ne = np.arange(1, 16)[None, :]\n"
                     "from scipy.stats import norm\n"
                     "P = norm.cdf((e + 0.5 - a) / 3.0) - norm.cdf((e - 0.5 - a) / 3.0)\nP = P / P.sum(axis=1, keepdims=True)\n"
                     "samples = np.linspace(5.0, 40.0, 4 * 2 * 15).reshape(4, 2, 15)\n"
                     "numbers = samples[:, :, :, None] * P[None, None, :, :]\n",
            "call": "design_precision(theta, 2012, 3, numbers, np.array([2010, 2011, 2012, 2013]), 1998, 2023)",
            "gold_call": "_oracle_design_precision(theta, 2012, 3, numbers, np.array([2010, 2011, 2012, 2013]), 1998, 2023)",
            "tol": 1e-6,
        },
    ]
