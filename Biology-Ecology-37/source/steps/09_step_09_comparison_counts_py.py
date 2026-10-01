"""
Number of pairwise comparisons of one kinship type in each class of estimated ages, leaving out the pairs that are not compared.

The information a design carries comes from pairwise comparisons, and pairs with the

same covariates can be grouped: the number of comparisons in a class is the product

of the numbers of samples in the two classes that form it. With aging error the

covariates are what the genotyping laboratory will see, the estimated ages, so the

classes and the rules that leave pairs out are set on estimated ages. A young calf

sampled beside its mother is not an independent draw, so potential offspring still

dependent on their mothers are not compared with mothers sampled in the same year;

and at long birth gaps a detected second-order kinship could as well be grandparent

and grandchild, so half-sibling comparisons are limited to the birth gaps at which

such pairs are rare. Only animals born within the modelled years can be offspring or

half-siblings.

Returns
-------
np.ndarray shaped like the probabilities of the requested kinship type, the number of comparisons in each class of estimated ages
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def comparison_counts(numbers: "np.ndarray", sample_years: "np.ndarray", first_year: int, entry_age: int,
                      kind: str) -> "np.ndarray":
    '''Number of pairwise comparisons of one kinship type in each class of estimated ages, leaving out the pairs that are not compared.

    numbers[i, s, a - 1, e - 1] is the expected number of animals of sex s (0
    female, 1 male) sampled in sample_years[i] with true age a and estimated
    age e, as returned by the fifth step, for ages 1 to A. Every expected
    sample is a distinct animal and enters every comparison type it is
    eligible for; the number of comparisons between two classes is the
    product of their expected numbers of animals by estimated age (summed
    over true ages). Only females can be mothers or self-recaptures. The
    rules use estimated ages: an animal's estimated birth year is its
    sampling year minus its estimated age, and the age at first birth is
    entry_age + 2.
    kind "MOP": shape (Y, A, Y, 2, A); entry [i, e - 1, j, s, f - 1] counts
    comparisons of a female sampled in sample_years[i] with estimated age e
    with an animal of sex s sampled in sample_years[j] with estimated age f,
    when the latter's estimated birth year is first_year or later, leaving
    out every pair sampled in the same year in which f is below the age at
    first birth.
    kind "HSP": shape (Y, 2, A, Y, 2, A); entry [i, s, e - 1, j, t, f - 1]
    counts comparisons of an animal of sex s sampled in sample_years[i] with
    estimated age e with an animal of sex t sampled in sample_years[j] with
    estimated age f, each pair once, when both estimated birth years are
    first_year or later and the second animal's estimated birth year minus
    the first's is positive and below twice the age at first birth plus two
    years, that is, below 2 (entry_age + 2) + 2; zero otherwise.
    kind "SP": shape (Y, A, Y, 2); entry [i, e - 1, m, d] counts comparisons
    of a female sampled in sample_years[i] with estimated age e with a female
    sampled in a later year sample_years[m] whose stage is d (d = 0 true ages
    1 to 5, d = 1 true ages 6 or older; the stage is known without error),
    whatever their birth years; zero when sample_years[m] <= sample_years[i].

    Parameters
    ----------
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200.
    entry_age : int
        Age at which females enter the breeding cycle, an integer from 1 to
        20.
    kind : str
        "MOP", "HSP" or "SP".

    Returns
    -------
    counts : np.ndarray
        Shape as given for kind; non-negative.

    Raises
    ------
    ValueError
        If numbers is not a finite, non-negative array of shape (Y, 2, A, A)
        with 6 <= A <= 100 matching sample_years, if sample_years is not a
        non-empty, strictly increasing one-dimensional array of integers from
        1900 to 2200, if first_year is not an integer from 1900 to 2200, if
        entry_age is not an integer from 1 to 20, or if kind is not "MOP",
        "HSP" or "SP".
    '''
    return counts  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_comparison_counts(numbers: "np.ndarray", sample_years: "np.ndarray", first_year: int, entry_age: int,
                              kind: str) -> "np.ndarray":
    ys = np.asarray(sample_years)
    if ys.ndim != 1 or ys.size == 0 or not np.issubdtype(ys.dtype, np.integer) or ys.min() < 1900 or ys.max() > 2200 \
            or np.any(np.diff(ys) <= 0):
        raise ValueError("sample_years must be a non-empty strictly increasing 1-D array of integers from 1900 to 2200")
    ys = ys.astype(np.int64)
    J = np.asarray(numbers, dtype=float)
    if J.ndim != 4 or J.shape[0] != ys.size or J.shape[1] != 2 or J.shape[2] != J.shape[3] or not 6 <= J.shape[2] <= 100 \
            or not np.all(np.isfinite(J)) or np.any(J < 0):
        raise ValueError("numbers must be a finite non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100")
    if isinstance(first_year, bool) or not isinstance(first_year, (int, np.integer)) or not 1900 <= first_year <= 2200:
        raise ValueError("first_year must be an integer from 1900 to 2200")
    if isinstance(entry_age, bool) or not isinstance(entry_age, (int, np.integer)) or not 1 <= entry_age <= 20:
        raise ValueError("entry_age must be an integer from 1 to 20")
    if kind not in ("MOP", "HSP", "SP"):
        raise ValueError('kind must be "MOP", "HSP" or "SP"')
    n_y, n_a = ys.size, J.shape[2]
    first_birth_age = entry_age + 2
    est = J.sum(axis=2)                                             # (Y, 2, A): expected numbers by estimated age
    e = np.arange(1, n_a + 1)
    birth_est = ys[:, None] - e[None, :]                            # (Y, A): estimated birth years
    if kind == "MOP":
        usable = est * (birth_est >= first_year)[:, None, :]        # potential offspring, either sex
        same_year_young = np.eye(n_y, dtype=bool)[:, None, :, None, None] & (e < first_birth_age)[None, None, None, None, :]
        pairs = est[:, 0, :][:, :, None, None, None] * usable[None, None, :, :, :]
        return np.where(same_year_young, 0.0, pairs)
    if kind == "HSP":
        b1 = birth_est[:, None, :, None, None, None]
        b2 = birth_est[None, None, None, :, None, :]
        gap = b2 - b1
        admitted = (b1 >= first_year) & (b2 >= first_year) & (gap > 0) & (gap < 2 * first_birth_age + 2)
        return np.where(admitted, est[:, :, :, None, None, None] * est[None, None, None, :, :, :], 0.0)
    true_f = J[:, 0].sum(axis=2)                                    # (Y, A) females by true age
    stage_tot = np.stack([true_f[:, :5].sum(axis=1), true_f[:, 5:].sum(axis=1)], axis=1)      # (Y, 2)
    later = ys[None, :] > ys[:, None]
    return est[:, 0, :][:, :, None, None] * stage_tot[None, None, :, :] * later[:, None, :, None]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    numbers = ("ages = np.arange(1, 38)\n"
               "lam = np.exp(-0.012)\n"
               "rel = np.where(ages < 6, lam ** (-ages) * 0.85 ** (ages - 1), lam ** (-ages) * 0.85 ** 5 * 0.955 ** (ages - 6))\n"
               "rel_m = np.where(ages <= 5, rel, 0.0)\n"
               "share = np.array([rel, rel_m]) / (rel.sum() + rel_m.sum())\n"
               "a = np.arange(1, 38)[:, None]\ne = np.arange(1, 38)[None, :]\n"
               "from scipy.stats import norm\n"
               "P = norm.cdf((e + 0.5 - a) / 1.6625615194893861) - norm.cdf((e - 0.5 - a) / 1.6625615194893861)\n"
               "P = P / P.sum(axis=1, keepdims=True)\n"
               "samples = np.array([800.0, 800.0, 1000.0, 1000.0])[:, None, None] * share[None, :, :]\n"
               "numbers = samples[:, :, :, None] * P[None, None, :, :]\n"
               "years = np.array([2014, 2018, 2024, 2027])\n")
    return [
        # --- normal: mother-offspring comparisons, same-year pairs present ---
        {
            "setup": "import numpy as np\n" + numbers,
            "call": "comparison_counts(numbers, years, 2002, 4, 'MOP')",
            "gold_call": "_oracle_comparison_counts(numbers, years, 2002, 4, 'MOP')",
        },
        # --- boundary: half-sibling comparisons, the gap limit and the first modelled year on estimated births ---
        {
            "setup": "import numpy as np\n" + numbers,
            "call": "comparison_counts(numbers, years, 2002, 4, 'HSP')",
            "gold_call": "_oracle_comparison_counts(numbers, years, 2002, 4, 'HSP')",
        },
        # --- edge: self-recaptures, first samples by estimated age and later stages by true age ---
        {
            "setup": "import numpy as np\n" + numbers,
            "call": "comparison_counts(numbers, years, 2002, 4, 'SP')",
            "gold_call": "_oracle_comparison_counts(numbers, years, 2002, 4, 'SP')",
        },
        # --- edge: a later entry to the breeding cycle and a later first year move both rules ---
        {
            "setup": "import numpy as np\nsamples = np.linspace(1.0, 30.0, 3 * 2 * 12).reshape(3, 2, 12)\n"
                     "a = np.arange(1, 13)[:, None]\ne = np.arange(1, 13)[None, :]\n"
                     "from scipy.stats import norm\n"
                     "P = norm.cdf((e + 0.5 - a) / 2.5) - norm.cdf((e - 0.5 - a) / 2.5)\nP = P / P.sum(axis=1, keepdims=True)\n"
                     "numbers = samples[:, :, :, None] * P[None, None, :, :]\nyears = np.array([2010, 2011, 2015])\n",
            "call": "comparison_counts(numbers, years, 2004, 6, 'HSP')",
            "gold_call": "_oracle_comparison_counts(numbers, years, 2004, 6, 'HSP')",
        },
    ]
