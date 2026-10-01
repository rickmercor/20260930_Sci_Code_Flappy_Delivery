"""
Probability that two sampled animals are maternal half-siblings, for samples classified by sampling year, sex and estimated age.

Two animals born in different years are maternal half-siblings when one female is the

mother of both. The older sibling marks its mother, so the question is how likely

that female was to produce the younger one: she had to survive the intervening years

and to be in a position to calve again, competing with every other adult female that

calved in that year. With skip-breeding, what is known about the mother at the first

birth changes her chances at the second, so the probability does not simply follow

the long-run calving share. With estimated ages the birth years themselves are

uncertain, and even which animal is the older can be.

Returns
-------
np.ndarray of shape (Y, 2, A, Y, 2, A), the probability that two animals of given years, sexes and estimated ages are maternal half-siblings
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def half_sibling_probabilities(theta: "np.ndarray", ref_year: int, sample_years: "np.ndarray", first_year: int,
                               numbers: "np.ndarray") -> "np.ndarray":
    '''Probability that two sampled animals are maternal half-siblings, for samples classified by sampling year, sex and estimated age.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    as in the second step, with the breeding cycle of the first step.
    numbers[i, s, a - 1, e - 1] is as returned by the fifth step for animals
    sampled in sample_years (ages 1 to A). Entry [i, s, e - 1, j, t, f - 1]
    of the result is the probability that an animal of sex s sampled in
    sample_years[i] with estimated age e and an animal of sex t sampled in
    sample_years[j] with estimated age f have the same mother.

    Given its sampling year, sex and estimated age, an animal's true age a
    has probability proportional to numbers[i, s, a - 1, e - 1], and the two
    animals' true ages are independent. The entry is the average, over both
    true ages, of the probability for those true ages, and it is zero when
    either class has no expected animals. For true birth years b and b'
    (sampling year minus true age), the probability is zero when b = b' or
    when either is before first_year. Otherwise let b < b' be the earlier and
    the later of the two, whichever animal was born first. The earlier-born
    animal's mother had a calf of the year in year b and was alive one year
    later. She is the later-born animal's mother with her expected share of
    the calves of the year born in year b': she must survive, at the adult
    rate, from year b + 1 to year b' + 1, and have a calf of the year in
    year b'; the total over all adult females is taken as the long-run
    proportion of adult females with a calf (first step) times adult female
    abundance in year b' (second step).

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(sample_years).
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers.

    Returns
    -------
    probabilities : np.ndarray
        Shape (Y, 2, A, Y, 2, A), as described above; a probability too
        small for double precision may be returned as zero.

    Raises
    ------
    ValueError
        If theta is not an array of shape (6,) inside the domain of the second
        step, if ref_year is not an integer from 1900 to 2200, if sample_years
        is not a non-empty, strictly increasing one-dimensional array of
        integers from 1900 to 2200, if first_year is not an integer from 1900
        to 2200 below max(sample_years), or if numbers is not a finite,
        non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100.
    '''
    return probabilities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _hsp_calf_return(psi2: complex, psi3: complex, max_gap: int) -> "np.ndarray":
    """Probability of being in the calf state g years after being in it, g = 0..max_gap."""
    zero = 0.0 * psi2
    T = np.array([[zero, psi2, psi3], [zero + 1.0, zero, zero], [zero, 1.0 - psi2, 1.0 - psi3]])
    state = np.array([zero, zero + 1.0, zero])
    out = []
    for _ in range(max_gap + 1):
        out.append(state[1])
        state = T @ state
    return np.array(out)


def _hsp_inputs(sample_years: "np.ndarray", first_year: int, numbers: "np.ndarray") -> tuple:
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
    return ys, int(first_year), J


def _oracle_half_sibling_probabilities(theta: "np.ndarray", ref_year: int, sample_years: "np.ndarray", first_year: int,
                                       numbers: "np.ndarray") -> "np.ndarray":
    th = np.asarray(theta)
    if th.shape != (6,) or not np.issubdtype(th.dtype, np.number) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    ys, first, J = _hsp_inputs(sample_years, first_year, numbers)
    n_a = J.shape[2]
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    psi2 = 1.0 / (1.0 + np.exp(-th[4]))
    psi3 = 1.0 / (1.0 + np.exp(-th[5]))
    births = np.arange(first, int(ys.max()))                      # modelled birth years a sampled animal can have
    adult = _oracle_stage_abundances(th, ref_year, births)[0]      # validates theta and ref_year
    beta2 = _oracle_breeding_cycle_fecundity(psi2, psi3, 1, np.array([0]))[0]
    ret = _hsp_calf_return(psi2, psi3, births.size)
    b = (ys[:, None] - np.arange(1, n_a + 1)[None, :])             # (Y, A): true birth year of each (year, true age)
    b1 = b[:, :, None, None]
    b2 = b[None, None, :, :]
    early, late = np.minimum(b1, b2), np.maximum(b1, b2)
    gap = late - early
    modelled = (early >= first) & (gap > 0)
    g = np.where(modelled, gap, 0)
    idx_late = np.clip(late - first, 0, births.size - 1)
    true_prob = np.where(modelled, phi_a ** g * ret[g] / (adult[idx_late] * beta2), 0.0 * phi_a)
    total = J.sum(axis=2, keepdims=True)
    W = np.where(total > 0, J / np.where(total > 0, total, 1.0), 0.0)     # P(true age | year, sex, estimated age)
    half = np.einsum("isae,iajc->isejc", W, true_prob, optimize=True)
    return np.einsum("isejc,jtcf->isejtf", half, W, optimize=True)

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
            "call": "half_sibling_probabilities(theta, 2016, years, 2002, numbers)",
            "gold_call": "_oracle_half_sibling_probabilities(theta, 2016, years, 2002, numbers)",
        },
        # --- boundary: one sampling year, so only the two animals' true ages separate their birth years ---
        {
            "setup": "import numpy as np\n" + theta + numbers.replace("SD", "0.9").replace("SIZES", "[500.0]")
                     + "years = np.array([2020])\n",
            "call": "half_sibling_probabilities(theta, 2016, years, 2000, numbers)",
            "gold_call": "_oracle_half_sibling_probabilities(theta, 2016, years, 2000, numbers)",
        },
        # --- edge: an imprecise clock in a small, growing population, so the estimated order often differs ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(3000.0), 0.05, np.log(0.93 / 0.07), np.log(0.7 / 0.3), "
                     "np.log(0.25 / 0.75), np.log(0.5 / 0.5)])\n"
                     "a = np.arange(1, 16)[:, None]\ne = np.arange(1, 16)[None, :]\n"
                     "from scipy.stats import norm\n"
                     "P = norm.cdf((e + 0.5 - a) / 3.0) - norm.cdf((e - 0.5 - a) / 3.0)\nP = P / P.sum(axis=1, keepdims=True)\n"
                     "samples = np.linspace(5.0, 40.0, 3 * 2 * 15).reshape(3, 2, 15)\n"
                     "numbers = samples[:, :, :, None] * P[None, None, :, :]\nyears = np.array([2010, 2011, 2013])\n",
            "call": "half_sibling_probabilities(theta, 2012, years, 1998, numbers)",
            "gold_call": "_oracle_half_sibling_probabilities(theta, 2012, years, 1998, numbers)",
        },
    ]
