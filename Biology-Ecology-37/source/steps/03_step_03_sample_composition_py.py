"""
Expected number of sampled animals of each sex and age in each sampling year, for sampling that does not select by sex or age among the animals available.

A design calculation needs the number of samples in every covariate class before

any sample is taken. When biopsies are taken from whichever animals are encountered,

the expected numbers per sex and age follow the age structure of the animals that

can be encountered. Here adult males do not visit the survey area, so the available

animals are females of every sampled age and juvenile males, and the expected

numbers, fractional as they are, stand in for the sample sizes in every later count.

Returns
-------
np.ndarray of shape (Y, 2, A), the expected number of sampled animals by year, sex (0 female, 1 male) and true age 1 to A
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sample_composition(sample_sizes: "np.ndarray", growth_rate: float, adult_survival: float,
                       juvenile_survival: float, max_female_age: int, max_male_age: int) -> "np.ndarray":
    '''Expected number of sampled animals of each sex and age in each sampling year, for sampling that does not select by sex or age among the animals available.

    In each sampling year, sample_sizes[i] animals are taken from the
    animals available: females aged 1 to max_female_age and males aged 1 to
    max_male_age. The numbers at age are those of the quasi-equilibrium age
    structure with annual survival juvenile_survival at ages 1 to 5 (the year
    from age 5 to age 6 included) and adult_survival from age 6 on, and
    population change exp(growth_rate) per year; each sex has the same
    numbers at every juvenile age. The expected number sampled in a class is
    sample_sizes[i] times that class's share of the available animals.

    Parameters
    ----------
    sample_sizes : np.ndarray
        Shape (Y,), Y >= 1; number of animals sampled in each sampling year,
        finite and non-negative.
    growth_rate : float
        Annual rate of change r, a finite number from -0.5 to 0.5.
    adult_survival : float
        Annual survival from age 6 on, a finite number strictly between 0
        and 1 with adult_survival < exp(growth_rate).
    juvenile_survival : float
        Annual survival at ages 1 to 5, a finite number strictly between 0
        and 1.
    max_female_age : int
        Oldest age at which females are sampled, an integer from 6 to 100.
    max_male_age : int
        Oldest age at which males are sampled, an integer from 0 to 5 (0
        means that no males are sampled).

    Returns
    -------
    samples : np.ndarray
        Shape (Y, 2, max_female_age). Entry [i, 0, a - 1] is the expected
        number of females of age a and entry [i, 1, a - 1] the expected
        number of males of age a sampled in year i (zero above max_male_age).

    Raises
    ------
    ValueError
        If sample_sizes is not a non-empty one-dimensional array of finite
        non-negative numbers, if growth_rate is not a finite number from -0.5
        to 0.5, if either survival is not a finite number strictly between 0
        and 1, if adult_survival >= exp(growth_rate), if max_female_age is
        not an integer from 6 to 100, or if max_male_age is not an integer
        from 0 to 5.
    '''
    return samples  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sc_number(name: str, value: float, low: float, high: float, open_ends: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    v = float(value)
    if not np.isfinite(v) or v < low or v > high or (open_ends and (v <= low or v >= high)):
        raise ValueError(f"{name} must be a finite number in the stated range")
    return v


def _oracle_sample_composition(sample_sizes: "np.ndarray", growth_rate: float, adult_survival: float,
                               juvenile_survival: float, max_female_age: int, max_male_age: int) -> "np.ndarray":
    n = np.asarray(sample_sizes, dtype=float)
    if n.ndim != 1 or n.size == 0 or not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("sample_sizes must be a non-empty 1-D array of finite non-negative numbers")
    r = _sc_number("growth_rate", growth_rate, -0.5, 0.5, False)
    phi_a = _sc_number("adult_survival", adult_survival, 0.0, 1.0, True)
    phi_j = _sc_number("juvenile_survival", juvenile_survival, 0.0, 1.0, True)
    lam = np.exp(r)
    if phi_a >= lam:
        raise ValueError("adult_survival must be below exp(growth_rate)")
    for name, v, lo, hi in (("max_female_age", max_female_age, 6, 100), ("max_male_age", max_male_age, 0, 5)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    ages = np.arange(1, max_female_age + 1)
    rel = np.where(ages < 6, lam ** (-ages) * phi_j ** (ages - 1),
                   lam ** (-ages) * phi_j ** 5 * phi_a ** (ages - 6))
    rel_m = np.where(ages <= max_male_age, rel, 0.0)
    share = np.array([rel, rel_m]) / (rel.sum() + rel_m.sum())
    return n[:, None, None] * share[None, :, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: five past years and four new years, females to 37 and juvenile males ---
        {
            "setup": "import numpy as np\nsizes = np.array([800.0] * 5 + [950.0] * 4)\n",
            "call": "sample_composition(sizes, -0.012, 0.955, 0.85, 37, 5)",
            "gold_call": "_oracle_sample_composition(sizes, -0.012, 0.955, 0.85, 37, 5)",
        },
        # --- boundary: no males sampled, stationary population, females only to the first adult age ---
        {
            "setup": "import numpy as np\nsizes = np.array([100.0, 0.0])\n",
            "call": "sample_composition(sizes, 0.0, 0.8, 0.6, 6, 0)",
            "gold_call": "_oracle_sample_composition(sizes, 0.0, 0.8, 0.6, 6, 0)",
        },
        # --- edge: fast growth, survival close to one, long female age range ---
        {
            "setup": "import numpy as np\nsizes = np.array([1234.5])\n",
            "call": "sample_composition(sizes, 0.08, 0.99, 0.95, 60, 3)",
            "gold_call": "_oracle_sample_composition(sizes, 0.08, 0.99, 0.95, 60, 3)",
        },
    ]
