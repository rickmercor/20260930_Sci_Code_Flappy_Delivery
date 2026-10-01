"""
Probability that two sampled animals of given ages have the same mother.

Two sampled animals are maternal half-siblings when they have the same mother, and in close-kin

mark-recapture the chance of that depends only on their birth years: the older animal's mother

must still be alive when the younger is born, and must then be drawn as the younger's mother

from the females that breed that year. A female that breeds on a fixed cycle can be the mother

of both only if their birth years sit a whole number of cycles apart, and only part of the adult

females fall due in any one year. Animals born in the same year can be littermates, which one

within-cohort factor allows for. A sampled animal's birth year is its survey year minus its age,

and in a growing population the number of adult females differs from year to year.

Returns
-------
np.ndarray, shape (max_age + 1, max_age + 1): rows indexed by the age of animal 1, columns by the age of animal 2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def maternal_half_sibling_matrix(year_1: int, year_2: int, mortality: float, reference_females: float, growth: float, reference_year: int, within_cohort_factor: float, breeding_period: int, max_age: int) -> "np.ndarray":
    '''Probability that two sampled animals of given ages have the same mother.

    Animal 1 is sampled in year_1 and animal 2 in year_2; an animal sampled
    in year y at age a was born in year y - a. The adult females alive in year
    t number reference_females * exp(growth * (t - reference_year)), and a
    female that has matured dies at the annual rate mortality from then on. Each adult female
    breeds once every breeding_period years, the years in which she breeds
    are fixed for her lifetime, and the breeding_period breeding schedules
    are equally common among the adult females of every age. Each newborn's
    mother is equally likely to be any adult female that breeds in its birth
    year, and two animals born in the same year have the same mother
    within_cohort_factor times as often as that rule alone implies. Return
    the matrix whose entry [a1, a2] is the probability that animal 1, aged a1,
    and animal 2, aged a2, have the same mother, for a1, a2 = 0, 1, ..., max_age.

    Parameters
    ----------
    year_1 : int
        Survey year of animal 1.
    year_2 : int
        Survey year of animal 2.
    mortality : float
        Annual mortality rate of adult females, positive and finite.
    reference_females : float
        Number of adult females alive in reference_year, positive and finite.
    growth : float
        Annual growth rate of the number of adult females, finite.
    reference_year : int
        Year in which the number of adult females equals reference_females.
    within_cohort_factor : float
        Multiplier for animals born in the same year, finite and >= 0.
    max_age : int
        Oldest age, an integer from 0 to 400.

    Returns
    -------
    probabilities : np.ndarray
        Shape (max_age + 1, max_age + 1): rows indexed by the age of animal 1,
        columns by the age of animal 2.

    Raises
    ------
    ValueError
        If a year is not an integer, if mortality or reference_females is not
        positive and finite, if growth is not finite, if within_cohort_factor
        is negative or not finite, if breeding_period is not an integer of at
        least 1, or if max_age is not an integer from 0 to 400.
    '''
    return probabilities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_maternal_half_sibling_matrix(year_1: int, year_2: int, mortality: float, reference_females: float, growth: float, reference_year: int, within_cohort_factor: float, breeding_period: int, max_age: int) -> "np.ndarray":
    for v in (year_1, year_2, reference_year):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("years must be integers")
    if not (np.isfinite(mortality) and mortality > 0.0):
        raise ValueError("mortality must be positive and finite")
    if not (np.isfinite(reference_females) and reference_females > 0.0):
        raise ValueError("reference_females must be positive and finite")
    if not np.isfinite(growth):
        raise ValueError("growth must be finite")
    if not (np.isfinite(within_cohort_factor) and within_cohort_factor >= 0.0):
        raise ValueError("within_cohort_factor must be finite and non-negative")
    if isinstance(breeding_period, bool) or not isinstance(breeding_period, (int, np.integer)) or breeding_period < 1:
        raise ValueError("breeding_period must be an integer of at least 1")
    if isinstance(max_age, bool) or not isinstance(max_age, (int, np.integer)) or not 0 <= max_age <= 400:
        raise ValueError("max_age must be an integer from 0 to 400")
    ages = np.arange(max_age + 1, dtype=float)
    birth_1 = (float(year_1) - ages)[:, None]
    birth_2 = (float(year_2) - ages)[None, :]
    gap = np.abs(birth_2 - birth_1)
    younger = np.maximum(birth_1, birth_2)                 # the mother is drawn among the females breeding in the later year
    breeders = reference_females * np.exp(growth * (younger - float(reference_year))) / float(breeding_period)
    probabilities = np.exp(-mortality * gap) / breeders    # the earlier mother survives the gap, then is drawn
    probabilities = np.where(gap == 0.0, within_cohort_factor * probabilities, probabilities)
    on_cycle = np.remainder(gap, float(breeding_period)) == 0.0     # a female breeds only in her own years
    return np.where(on_cycle, probabilities, 0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: surveys eight years apart in a growing population, littermates more often related ---
        {
            "setup": "",
            "call": "maternal_half_sibling_matrix(2016, 2024, 0.16, 3000.0, 0.05, 2024, 1.6, 3, 40)",
            "gold_call": "_oracle_maternal_half_sibling_matrix(2016, 2024, 0.16, 3000.0, 0.05, 2024, 1.6, 3, 40)",
            "tol": 1e-12,
        },
        # --- boundary: one survey year, constant abundance, no within-cohort effect ---
        {
            "setup": "",
            "call": "maternal_half_sibling_matrix(2020, 2020, 0.2, 1500.0, 0.0, 2020, 1.0, 1, 12)",
            "gold_call": "_oracle_maternal_half_sibling_matrix(2020, 2020, 0.2, 1500.0, 0.0, 2020, 1.0, 1, 12)",
            "tol": 1e-12,
        },
        # --- edge: the later survey listed first, a declining population, and no same-year half-siblings ---
        {
            "setup": "",
            "call": "maternal_half_sibling_matrix(2024, 2016, 0.12, 800.0, -0.03, 2010, 0.0, 4, 30)",
            "gold_call": "_oracle_maternal_half_sibling_matrix(2024, 2016, 0.12, 800.0, -0.03, 2010, 0.0, 4, 30)",
            "tol": 1e-12,
        },
    ]
