"""
Probability that a sampled female is the mother of a sampled animal, for samples classified by sampling year, sex and estimated age.

Close-kin mark-recapture turns genetic kinship between samples into information about

abundance: the chance that a given female is the mother of a given sampled animal is

her expected share of all the calves produced in the year that animal was born. In a

skip-breeding population that share depends on whether she was likely to have a calf

in that year, which the breeding cycle predicts from her age, and on how many adult

females competed with her; a female sampled before the birth year also had to live

long enough to be a mother in that year. When ages are only estimated, the probability

for two samples is a weighted sum of these probabilities over the true ages the two

animals may have, weighted by how likely each true age is given its estimate.

Returns
-------
np.ndarray of shape (Y, A, Y, 2, A), the probability that a female of a given year and estimated age is the mother of an animal of a given year, sex and estimated age
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mother_offspring_probabilities(theta: "np.ndarray", ref_year: int, entry_age: int, sample_years: "np.ndarray",
                                   first_year: int, numbers: "np.ndarray") -> "np.ndarray":
    '''Probability that a sampled female is the mother of a sampled animal, for samples classified by sampling year, sex and estimated age.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    as in the second step, with the breeding cycle of the first step
    (entering it resting at entry_age). numbers[i, s, a - 1, e - 1] is the
    expected number of animals of sex s (0 female, 1 male) sampled in
    sample_years[i] with true age a and estimated age e, as returned by the
    fifth step, for ages 1 to A. Entry [i, e - 1, j, s, f - 1] of the result
    is the probability that a female sampled in sample_years[i] with
    estimated age e is the mother of an animal of sex s sampled in
    sample_years[j] with estimated age f.

    Given its sampling year, sex and estimated age, an animal's true age a
    has probability proportional to numbers[i, s, a - 1, e - 1], and the two
    animals' true ages are independent. The entry is the average, over both
    true ages, of the probability for those true ages, and it is zero when
    either class has no expected animals. For a female of true age a sampled
    in year y and an animal whose true birth year is b (its sampling year
    minus its true age), the probability is zero when b is before
    first_year; otherwise it is by expected relative reproductive output:
    her expected number of calves of the year in year b, divided by the
    total for all adult females, which is taken as the long-run proportion
    of adult females with a calf (first step) times adult female abundance
    in year b (second step). A female sampled before year b must survive from
    her sampling year to year b, at the juvenile rate for each year she
    spends at ages 1 to 5 and the adult rate after; a female sampled in or
    after year b needs no survival term.

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    entry_age : int
        Age at which a female enters the breeding cycle, resting, an integer
        from 1 to 20.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(sample_years).
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers, with A + max(sample_years) - min(sample_years) <= 200.

    Returns
    -------
    probabilities : np.ndarray
        Shape (Y, A, Y, 2, A), as described above; a probability too small
        for double precision may be returned as zero.

    Raises
    ------
    ValueError
        If theta is not an array of shape (6,) inside the domain of the second
        step, if ref_year is not an integer from 1900 to 2200, if entry_age is
        not an integer from 1 to 20, if sample_years is not a non-empty,
        strictly increasing one-dimensional array of integers from 1900 to
        2200, if first_year is not an integer from 1900 to 2200 below
        max(sample_years), or if numbers is not a finite, non-negative array
        of shape (Y, 2, A, A) with 6 <= A <= 100 and
        A + max(sample_years) - min(sample_years) <= 200.
    '''
    return probabilities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _mop_inputs(sample_years: "np.ndarray", first_year: int, numbers: "np.ndarray") -> tuple:
    ys = np.asarray(sample_years)
    if ys.ndim != 1 or ys.size == 0 or not np.issubdtype(ys.dtype, np.integer) or ys.min() < 1900 or ys.max() > 2200 \
            or np.any(np.diff(ys) <= 0):
        raise ValueError("sample_years must be a non-empty strictly increasing 1-D array of integers from 1900 to 2200")
    ys = ys.astype(np.int64)
    if isinstance(first_year, bool) or not isinstance(first_year, (int, np.integer)) or not 1900 <= first_year < ys.max():
        raise ValueError("first_year must be an integer from 1900 to 2200 below max(sample_years)")
    J = np.asarray(numbers, dtype=float)
    if J.ndim != 4 or J.shape[0] != ys.size or J.shape[1] != 2 or J.shape[2] != J.shape[3] or not 6 <= J.shape[2] <= 100 \
            or not np.all(np.isfinite(J)) or np.any(J < 0):
        raise ValueError("numbers must be a finite non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100")
    if J.shape[2] + int(ys.max()) - int(ys.min()) > 200:
        raise ValueError("A + max(sample_years) - min(sample_years) must not exceed 200")
    return ys, int(first_year), J


def _mop_true_age_weights(J: "np.ndarray") -> "np.ndarray":
    """W[i, s, a - 1, e - 1]: probability of true age a given sampling year i, sex s and estimated age e."""
    total = J.sum(axis=2, keepdims=True)
    return np.where(total > 0, J / np.where(total > 0, total, 1.0), 0.0)


def _oracle_mother_offspring_probabilities(theta: "np.ndarray", ref_year: int, entry_age: int, sample_years: "np.ndarray",
                                           first_year: int, numbers: "np.ndarray") -> "np.ndarray":
    th = np.asarray(theta)
    if th.shape != (6,) or not np.issubdtype(th.dtype, np.number) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    ys, first, J = _mop_inputs(sample_years, first_year, numbers)
    n_a = J.shape[2]
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    phi_j = 1.0 / (1.0 + np.exp(-th[3]))
    psi2 = 1.0 / (1.0 + np.exp(-th[4]))
    psi3 = 1.0 / (1.0 + np.exp(-th[5]))
    births = np.arange(first, int(ys.max()))                      # modelled birth years a sampled animal can have
    adult = _oracle_stage_abundances(th, ref_year, births)[0]      # validates theta and ref_year
    ages = np.arange(1, n_a + 1)
    yi = ys[:, None, None, None]
    ai = ages[None, :, None, None]
    b = ys[None, None, :, None] - ages[None, None, None, :]         # true birth year of the other animal
    age_at_birth = ai + (b - yi)
    oldest = int(age_at_birth.max())
    fec = _oracle_breeding_cycle_fecundity(psi2, psi3, entry_age, np.arange(0, max(oldest, 0) + 1))[1:]
    before = yi < b
    dt = np.where(before, b - yi, 0)                               # survival only from sampling up to the birth year
    juv_years = np.clip(np.minimum(6 - ai, dt), 0, None)           # years spent at ages 1 to 5
    surv = phi_j ** juv_years * phi_a ** (dt - juv_years)
    modelled = (b >= first) & (age_at_birth >= 1)
    idx_b = np.clip(b - first, 0, births.size - 1)
    true_prob = np.where(modelled, surv * fec[np.clip(age_at_birth, 0, oldest)] / adult[idx_b], 0.0 * surv)
    W = _mop_true_age_weights(J)
    half = np.einsum("iae,iajc->iejc", W[:, 0], true_prob, optimize=True)          # mother's true age averaged out
    return np.einsum("iejc,jscf->iejsf", half, W, optimize=True)                   # the other animal's true age averaged out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    theta = ("theta = np.array([np.log(48000.0), -0.012, np.log(0.955 / 0.045), np.log(0.85 / 0.15), "
             "np.log(0.16 / 0.84), np.log(0.38 / 0.62)])\n")
    numbers = ("ages = np.arange(1, 38)\n"
               "lam = np.exp(-0.012)\n"
               "rel = np.where(ages < 6, lam ** (-ages) * 0.85 ** (ages - 1), lam ** (-ages) * 0.85 ** 5 * 0.955 ** (ages - 6))\n"
               "rel_m = np.where(ages <= 5, rel, 0.0)\n"
               "share = np.array([rel, rel_m]) / (rel.sum() + rel_m.sum())\n"
               "a = np.arange(1, 38)[:, None]\ne = np.arange(1, 38)[None, :]\n"
               "from scipy.stats import norm\n"
               "P = norm.cdf((e + 0.5 - a) / SD) - norm.cdf((e - 0.5 - a) / SD)\nP = P / P.sum(axis=1, keepdims=True)\n"
               "samples = np.array(SIZES)[:, None, None] * share[None, :, :]\n"
               "numbers = samples[:, :, :, None] * P[None, None, :, :]\n")
    return [
        # --- normal: two past years and two new years of the shipped design, the fitted clock ---
        {
            "setup": "import numpy as np\n" + theta + numbers.replace("SD", "1.6625615194893861").replace(
                "SIZES", "[800.0, 800.0, 1000.0, 1000.0]") + "years = np.array([2014, 2016, 2024, 2027])\n",
            "call": "mother_offspring_probabilities(theta, 2016, 4, years, 2002, numbers)",
            "gold_call": "_oracle_mother_offspring_probabilities(theta, 2016, 4, years, 2002, numbers)",
        },
        # --- boundary: a single sampling year, first modelled birth year just below it, a precise clock ---
        {
            "setup": "import numpy as np\n" + theta + numbers.replace("SD", "0.3").replace("SIZES", "[500.0]")
                     + "years = np.array([2020])\n",
            "call": "mother_offspring_probabilities(theta, 2016, 4, years, 2019, numbers)",
            "gold_call": "_oracle_mother_offspring_probabilities(theta, 2016, 4, years, 2019, numbers)",
        },
        # --- edge: an imprecise clock and a small, growing population with early entry to the cycle ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(3000.0), 0.05, np.log(0.93 / 0.07), np.log(0.7 / 0.3), "
                     "np.log(0.25 / 0.75), np.log(0.5 / 0.5)])\n"
                     "a = np.arange(1, 16)[:, None]\ne = np.arange(1, 16)[None, :]\n"
                     "from scipy.stats import norm\n"
                     "P = norm.cdf((e + 0.5 - a) / 3.0) - norm.cdf((e - 0.5 - a) / 3.0)\nP = P / P.sum(axis=1, keepdims=True)\n"
                     "samples = np.linspace(5.0, 40.0, 3 * 2 * 15).reshape(3, 2, 15)\n"
                     "numbers = samples[:, :, :, None] * P[None, None, :, :]\nyears = np.array([2010, 2011, 2013])\n",
            "call": "mother_offspring_probabilities(theta, 2012, 3, years, 1998, numbers)",
            "gold_call": "_oracle_mother_offspring_probabilities(theta, 2012, 3, years, 1998, numbers)",
        },
    ]
