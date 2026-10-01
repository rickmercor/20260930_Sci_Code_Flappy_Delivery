"""
Total expected number of mother-offspring pairs, half-sibling pairs and self-recaptures at the smallest annual sample size of a new survey programme that reaches a target CV for adult female abundance, with ages from a fitted clock.

This step returns the deliverable. A new survey programme is to be added to past

sampling, and the design question is how many animals it must sample each year for

the estimate of adult female abundance to reach a target precision when self-

recaptures and close kin are analysed together and every age comes from an epigenetic

clock fitted to known-age animals. The fit fixes the age error, the smallest adequate

annual sample size fixes the design, and the design fixes what the genotyping of its

samples can be expected to find: the numbers of mother-offspring pairs, maternal

half-sibling pairs and self-recaptures on which every estimate will rest.

Returns
-------
float, the total expected number of mother-offspring pairs, half-sibling pairs and self-recaptures at the smallest adequate annual sample size, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def smallest_design_expected_kin_pairs(theta: "np.ndarray", ref_year: int, entry_age: int, first_year: int,
                                       past_years: "np.ndarray", past_size: float, new_years: "np.ndarray",
                                       size_step: int, max_size: int, max_female_age: int, max_male_age: int,
                                       target_year: int, cv_target: float, reference_true_ages: "np.ndarray",
                                       reference_estimated_ages: "np.ndarray") -> float:
    '''Total expected number of mother-offspring pairs, half-sibling pairs and self-recaptures at the smallest annual sample size of a new survey programme that reaches a target CV for adult female abundance, with ages from a fitted clock.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    holds the true values, as in the second step. The standard deviation of
    the clock's error is the maximum-likelihood value of the fourth step from
    the reference animals' true and estimated ages, with max_age =
    max_female_age. The design samples past_size animals in each of
    past_years and n animals in each of new_years, each year's sample
    composed by true age as in the third step at the true growth rate and
    survival rates, with females aged 1 to max_female_age and males aged 1
    to max_male_age, and each animal's estimated age distributed as in the
    fifth step with that standard deviation. For n = size_step,
    2 size_step, ..., max_size, compute the design precision of the eleventh
    step with first_year as the first modelled birth year and entry_age as
    the age at which females enter the breeding cycle, and take the smallest
    n whose expected CV of adult female abundance in target_year from all
    three kinship types is at most cv_target. Return the sum of the expected
    numbers of mother-offspring pairs, half-sibling pairs and self-recaptures
    of the eleventh step at that n.

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    entry_age : int
        Age at which females enter the breeding cycle, resting, an integer
        from 1 to 20.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(new_years).
    past_years : np.ndarray
        Shape (P,), P >= 1; strictly increasing integers from 1900 to 2200.
    past_size : float
        Animals sampled in each past year, a finite number from 0 to 1e6.
    new_years : np.ndarray
        Shape (Q,), Q >= 1; strictly increasing integers from 1900 to 2200,
        all later than every past year.
    size_step : int
        Spacing of the annual sample sizes tried, an integer from 1 to 10000.
    max_size : int
        Largest annual sample size tried, a multiple of size_step from
        size_step to 100000.
    max_female_age : int
        Oldest age at which females are sampled, an integer from 6 to 100,
        with max_female_age + max(new_years) - min(past_years) <= 200.
    max_male_age : int
        Oldest age at which males are sampled, an integer from 0 to 5.
    target_year : int
        Year of the adult female abundance of interest, an integer from 1900
        to 2200.
    cv_target : float
        Target CV, a finite number with 0 < cv_target < 1.
    reference_true_ages : np.ndarray
        Shape (K,), K >= 1; true ages of the reference animals, as in the
        fourth step.
    reference_estimated_ages : np.ndarray
        Shape (K,); their estimated ages, as in the fourth step.

    Returns
    -------
    expected_pairs : float
        The total expected number of mother-offspring pairs, half-sibling
        pairs and self-recaptures at the smallest adequate annual sample size,
        as a native Python float.

    Raises
    ------
    ValueError
        If any argument is outside its stated range or invalid as in the
        third, fourth, fifth or eleventh step, if a past year is not earlier
        than every new year, or if no annual sample size up to max_size
        reaches cv_target.
    '''
    return expected_pairs  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_smallest_design_expected_kin_pairs(theta: "np.ndarray", ref_year: int, entry_age: int, first_year: int,
                                               past_years: "np.ndarray", past_size: float, new_years: "np.ndarray",
                                               size_step: int, max_size: int, max_female_age: int, max_male_age: int,
                                               target_year: int, cv_target: float, reference_true_ages: "np.ndarray",
                                               reference_estimated_ages: "np.ndarray") -> float:
    th = np.asarray(theta, dtype=float)
    if th.shape != (6,) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    past = np.asarray(past_years)
    new = np.asarray(new_years)
    for name, v in (("past_years", past), ("new_years", new)):
        if v.ndim != 1 or v.size == 0 or not np.issubdtype(v.dtype, np.integer) or np.any(np.diff(v) <= 0):
            raise ValueError(f"{name} must be a non-empty strictly increasing 1-D integer array")
    if past.max() >= new.min():
        raise ValueError("every past year must be earlier than every new year")
    if isinstance(past_size, bool) or not isinstance(past_size, (int, float, np.integer, np.floating)) \
            or not np.isfinite(past_size) or not 0 <= past_size <= 1e6:
        raise ValueError("past_size must be a finite number from 0 to 1e6")
    for name, v, lo, hi in (("size_step", size_step, 1, 10000), ("max_size", max_size, 1, 100000)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if max_size < size_step or max_size % size_step:
        raise ValueError("max_size must be a multiple of size_step")
    if isinstance(cv_target, bool) or not isinstance(cv_target, (int, float, np.integer, np.floating)) \
            or not np.isfinite(cv_target) or not 0 < cv_target < 1:
        raise ValueError("cv_target must be a finite number strictly between 0 and 1")
    error_sd = _oracle_age_error_sd(reference_true_ages, reference_estimated_ages, max_female_age)
    years = np.concatenate([past, new]).astype(np.int64)
    r = float(th[1])
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    phi_j = 1.0 / (1.0 + np.exp(-th[3]))
    for n in range(size_step, max_size + 1, size_step):
        sizes = np.array([float(past_size)] * past.size + [float(n)] * new.size)
        samples = _oracle_sample_composition(sizes, r, phi_a, phi_j, max_female_age, max_male_age)
        numbers = _oracle_estimated_age_numbers(samples, error_sd)
        precision = _oracle_design_precision(th, ref_year, entry_age, numbers, years, first_year, target_year)
        if precision[0] <= cv_target:
            return float(precision[4] + precision[5] + precision[6])
    raise ValueError("no annual sample size up to max_size reaches cv_target")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    theta = ("theta = np.array([np.log(48000.0), -0.012, np.log(0.955 / 0.045), np.log(0.85 / 0.15), "
             "np.log(0.16 / 0.84), np.log(0.38 / 0.62)])\n")
    calib = ("true = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 31, 34, 36])\n"
             "est = np.array([2, 2, 4, 1, 3, 4, 2, 4, 5, 6, 4, 3, 3, 5, 4, 9, 6, 8, 9, 14, 15, 17, 19, 18, 20, 28, 28, 31, 32, 36])\n")
    return [
        # --- normal: five past years of 800 and four new years, CV target 0.10 for abundance in the last new year ---
        {
            "setup": "import numpy as np\n" + theta + calib,
            "call": "smallest_design_expected_kin_pairs(theta, 2016, 4, 2002, np.arange(2014, 2019), 800.0, np.arange(2024, 2028), 50, 3000, 37, 5, 2027, 0.10, true, est)",
            "gold_call": "_oracle_smallest_design_expected_kin_pairs(theta, 2016, 4, 2002, np.arange(2014, 2019), 800.0, np.arange(2024, 2028), 50, 3000, 37, 5, 2027, 0.10, true, est)",
        },
        # --- boundary: no past sampling and no males, a stationary population and a precise clock ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(2000.0), 0.0, np.log(0.9 / 0.1), np.log(0.75 / 0.25), np.log(0.2 / 0.8), np.log(0.45 / 0.55)])\n"
                     "true = np.array([2, 5, 9, 14, 20])\nest = np.array([2, 5, 10, 14, 20])\n",
            "call": "smallest_design_expected_kin_pairs(theta, 2010, 4, 1995, np.array([2008]), 0.0, np.array([2009, 2010, 2011]), 50, 5000, 25, 0, 2011, 0.15, true, est)",
            "gold_call": "_oracle_smallest_design_expected_kin_pairs(theta, 2010, 4, 1995, np.array([2008]), 0.0, np.array([2009, 2010, 2011]), 50, 5000, 25, 0, 2011, 0.15, true, est)",
        },
        # --- edge: a tight target and an imprecise clock, abundance of interest years after the last sample ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(9000.0), -0.03, np.log(0.94 / 0.06), np.log(0.8 / 0.2), np.log(0.1 / 0.9), np.log(0.35 / 0.65)])\n"
                     "true = np.array([1, 3, 3, 6, 8, 12, 15, 20, 24, 27])\nest = np.array([3, 1, 6, 9, 5, 16, 12, 24, 21, 30])\n",
            "call": "smallest_design_expected_kin_pairs(theta, 2005, 4, 1996, np.array([2003, 2004]), 300.0, np.array([2010, 2012]), 250, 20000, 30, 5, 2020, 0.06, true, est)",
            "gold_call": "_oracle_smallest_design_expected_kin_pairs(theta, 2005, 4, 1996, np.array([2003, 2004]), 300.0, np.array([2010, 2012]), 250, 20000, 30, 5, 2020, 0.06, true, est)",
        },
    ]
