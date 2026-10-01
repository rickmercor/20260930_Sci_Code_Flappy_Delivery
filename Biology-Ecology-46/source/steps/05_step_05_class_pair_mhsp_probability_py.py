"""
Probability that an animal of one survey class and a different animal of another class have the same mother.

Only an animal's survey year and the bin of its age score are recorded; its age is not, and neither is its

birth year, which is what a half-sibling probability depends on. A female that breeds on a fixed cycle can be

the mother of two animals only if their birth years sit a whole number of cycles apart, and only the females

whose cycle falls due in a year can be the mothers of that year's newborns, so the cycle both thins the birth

gaps that a maternal half-sibling pair can have and concentrates the motherhood of each year on part of the

adult females. What the surveys can measure is the rate at which pairs drawn from two classes turn out to be

maternal half-siblings, which is the probability this step returns.

Returns
-------
float, the probability that the two animals have the same mother, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def class_pair_mhsp_probability(class_1: "np.ndarray", class_2: "np.ndarray", calibration: "np.ndarray", demography: "np.ndarray", reference_year: int, breeding_period: int, max_age: int) -> float:
    '''Probability that an animal of one survey class and a different animal of another class have the same mother.

    A class is given as [year, lower, upper, min_age]: its animals were
    sampled in survey year `year` by a gear retaining only animals at least
    min_age whole years old, and their scores fell in the bin (lower, upper].
    calibration = [alpha, beta, dispersion]: the score of an animal aged a
    whole years is gamma distributed with mean alpha + beta * a and
    coefficient of variation sqrt(dispersion), the same at every age.
    demography = [mortality, reference_females, growth,
    within_cohort_factor, juvenile_mortality, maturity_age]:
    - an animal dies at the annual rate juvenile_mortality until it matures,
      which it does on reaching the age maturity_age, and at the annual rate
      mortality from then on; a mother is mature whenever she breeds;
    - the number of births, and with it the number of adult females, has
      grown by the factor exp(growth) every year for as long as the
      population has existed, the adult females alive in year t numbering
      reference_females * exp(growth * (t - reference_year));
    - each adult female breeds once every breeding_period years, the years
      in which she breeds are fixed for life, and the breeding_period
      breeding schedules are equally common among adult females at every
      age;
    - each newborn's mother is equally likely to be any adult female that
      breeds in its birth year;
    - two animals born in the same year have the same mother
      within_cohort_factor times as often as that rule alone implies.
    Each survey samples at random among the animals alive in its year that
    its gear retains, and an animal sampled in year y at age a was born in
    year y - a. Neither animal's age is recorded: its class is all that is
    known about it, and the two animals are drawn independently. Ages run
    over the whole years 0, 1, ..., max_age, where max_age is a numerical
    cutoff. Return the probability that the two animals have the same
    mother.

    Parameters
    ----------
    class_1, class_2 : np.ndarray
        Shape (4,) each: [year, lower, upper, min_age], the year and min_age
        whole numbers, 0 <= lower < upper and 0 <= min_age <= max_age.
    calibration : np.ndarray
        Shape (3,): alpha, beta and dispersion, each positive and finite.
    demography : np.ndarray
        Shape (6,): mortality (positive), reference_females (positive),
        growth (finite, with mortality + growth positive),
        within_cohort_factor (non-negative), juvenile_mortality (positive)
        and maturity_age (a whole number from 0 to max_age), all finite.
    reference_year : int
        The year in which the adult females number reference_females.
    breeding_period : int
        The number of years between one breeding and the next, at least 1.
    max_age : int
        The oldest age carried, a whole number from 1 to 400.

    Returns
    -------
    float
        The probability that the two animals have the same mother, as a
        native Python float.

    Raises
    ------
    ValueError
        If class_1 or class_2 does not have shape (4,), calibration shape
        (3,) or demography shape (6,); if a class year or min_age is not a
        whole number, a bin is not 0 <= lower < upper, or min_age is not in
        0 <= min_age <= max_age; if an entry of calibration is not positive
        and finite; if mortality or reference_females is not positive,
        mortality + growth is not positive, or within_cohort_factor is
        negative, or juvenile_mortality is not positive; if maturity_age is
        not a whole number in 0 <= maturity_age <= max_age; if reference_year
        is not an integer; if breeding_period is
        not an integer of at least 1; if max_age is not an integer from 1 to
        400; or if the class's score bin has probability zero at every age
        its gear retains.
    '''
    return x  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _cpp_age_given_class(cls: "np.ndarray", calibration: "np.ndarray", demography: "np.ndarray", max_age: int) -> "np.ndarray":
    """Conditional distribution of age given the class's survey gear and score bin."""
    year, lower, upper, min_age = cls
    prior = _oracle_sampled_age_distribution(float(demography[0]), float(demography[4]), int(demography[5]),
                                             float(demography[2]), int(min_age), max_age)
    bins = _oracle_score_bin_probabilities(float(lower), float(upper), float(calibration[0]), float(calibration[1]),
                                           float(calibration[2]), max_age)
    weights = prior * bins
    total = weights.sum()
    if not total > 0.0:
        raise ValueError("the class's bin has probability zero at every retained age")
    return weights / total


def _oracle_class_pair_mhsp_probability(class_1: "np.ndarray", class_2: "np.ndarray", calibration: "np.ndarray", demography: "np.ndarray", reference_year: int, breeding_period: int, max_age: int) -> float:
    c1 = np.asarray(class_1, dtype=float)
    c2 = np.asarray(class_2, dtype=float)
    cal = np.asarray(calibration, dtype=float)
    dem = np.asarray(demography, dtype=float)
    if c1.shape != (4,) or c2.shape != (4,) or cal.shape != (3,) or dem.shape != (6,):
        raise ValueError("class_1 and class_2 need shape (4,), calibration (3,) and demography (6,)")
    if isinstance(reference_year, bool) or not isinstance(reference_year, (int, np.integer)):
        raise ValueError("reference_year must be an integer")
    if isinstance(breeding_period, bool) or not isinstance(breeding_period, (int, np.integer)) or breeding_period < 1:
        raise ValueError("breeding_period must be an integer of at least 1")
    if isinstance(max_age, bool) or not isinstance(max_age, (int, np.integer)) or not 1 <= max_age <= 400:
        raise ValueError("max_age must be an integer from 1 to 400")
    for c in (c1, c2):
        if not np.all(np.isfinite(c)) or c[0] != np.round(c[0]) or c[3] != np.round(c[3]):
            raise ValueError("a class needs a finite whole year and minimum age")
        if not (0.0 <= c[1] < c[2]) or not (0 <= c[3] <= max_age):
            raise ValueError("a class needs 0 <= lower < upper and 0 <= min_age <= max_age")
    if not (np.all(np.isfinite(cal)) and np.all(cal > 0.0)):
        raise ValueError("alpha, beta and dispersion must be positive and finite")
    if not (np.all(np.isfinite(dem)) and dem[0] > 0.0 and dem[1] > 0.0 and dem[0] + dem[2] > 0.0
            and dem[3] >= 0.0 and dem[4] > 0.0):
        raise ValueError("demography must be finite with mortality > 0, reference_females > 0, "
                         "mortality + growth > 0, within_cohort_factor >= 0 and juvenile_mortality > 0")
    if dem[5] != np.round(dem[5]) or not 0 <= dem[5] <= max_age:
        raise ValueError("maturity_age must be a whole number from 0 to max_age")
    ages_1 = _cpp_age_given_class(c1, cal, dem, max_age)
    ages_2 = _cpp_age_given_class(c2, cal, dem, max_age)
    kin = _oracle_maternal_half_sibling_matrix(int(c1[0]), int(c2[0]), float(dem[0]), float(dem[1]), float(dem[2]),
                                               int(reference_year), float(dem[3]), int(breeding_period), max_age)
    return float(ages_1 @ kin @ ages_2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: young animals of the first survey against young animals of a survey with a size-limited gear ---
        {
            "setup": "import numpy as np\n"
                     "c1 = np.array([2016.0, 3.0, 5.0, 0.0])\nc2 = np.array([2024.0, 3.0, 5.0, 4.0])\n"
                     "cal = np.array([1.49, 0.82, 0.009])\ndem = np.array([0.16, 3000.0, 0.05, 1.6, 0.3, 8])\n",
            "call": "class_pair_mhsp_probability(c1, c2, cal, dem, 2024, 3, 150)",
            "gold_call": "_oracle_class_pair_mhsp_probability(c1, c2, cal, dem, 2024, 3, 150)",
            "tol": 1e-12,
        },
        # --- boundary: both animals from one class, a yearly cycle, constant abundance, no within-cohort effect ---
        {
            "setup": "import numpy as np\n"
                     "c1 = np.array([2016.0, 3.0, 5.0, 0.0])\n"
                     "cal = np.array([1.49, 0.82, 0.009])\ndem = np.array([0.16, 3000.0, 0.0, 1.0, 0.42, 5])\n",
            "call": "class_pair_mhsp_probability(c1, c1, cal, dem, 2016, 1, 150)",
            "gold_call": "_oracle_class_pair_mhsp_probability(c1, c1, cal, dem, 2016, 1, 150)",
            "tol": 1e-12,
        },
        # --- edge: old animals against young ones of a later survey, a noisy score, no littermates, a four-year cycle ---
        {
            "setup": "import numpy as np\n"
                     "c1 = np.array([2016.0, 17.0, 19.0, 0.0])\nc2 = np.array([2024.0, 7.0, 9.0, 4.0])\n"
                     "cal = np.array([1.2, 0.9, 0.05])\ndem = np.array([0.1, 2000.0, 0.03, 0.0, 0.28, 9])\n",
            "call": "class_pair_mhsp_probability(c1, c2, cal, dem, 2020, 4, 200)",
            "gold_call": "_oracle_class_pair_mhsp_probability(c1, c2, cal, dem, 2020, 4, 200)",
            "tol": 1e-12,
        },
    ]
