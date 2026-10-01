"""
Age distribution of animals sampled at random from a stable population, at or above a gear's minimum age.

A sampled animal's age is unknown, so every kinship probability is averaged over the ages it

could have, weighted by how common each age is among the animals a survey catches. When the number of births has changed at a steady rate for a long time, the ages of the living

animals settle into a fixed stable pattern, and random sampling reproduces it. Few wild animals

die at one rate all their lives: the young die faster, and the rate settles once they mature, so

the survivorship that shapes the catch is built from two rates rather than one. A survey gear

that cannot retain small animals removes the youngest ages from its catch.

Returns
-------
np.ndarray, shape (max_age + 1,): the probability of each age 0, 1, ..., max_age, zero below min_age, summing to 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sampled_age_distribution(mortality: float, juvenile_mortality: float, maturity_age: int, growth: float, min_age: int, max_age: int) -> "np.ndarray":
    '''Age distribution of animals sampled at random from a stable population, at or above a gear's minimum age.

    An animal dies at the annual rate juvenile_mortality until it matures,
    which it does on reaching the age maturity_age, and at the annual rate
    mortality from then on. The number of births per year has grown by the
    factor exp(growth) every year for as long as the population has existed,
    and a survey samples at random among the animals alive in its year that
    are at least min_age whole years old.
    Ages are whole years 0, 1, ..., max_age, where max_age is a numerical
    cutoff: older animals are ignored and the probabilities are normalised
    over the ages retained.

    Parameters
    ----------
    mortality : float
        Annual mortality rate of mature animals, positive and finite.
    juvenile_mortality : float
        Annual mortality rate before maturity, positive and finite.
    maturity_age : int
        Age at which an animal matures, a whole number from 0 to max_age.
    growth : float
        Annual growth rate of the number of births, finite, with
        mortality + growth > 0.
    min_age : int
        Youngest age the survey gear retains, an integer from 0 to max_age.
    max_age : int
        Oldest age kept, an integer from 1 to 400.

    Returns
    -------
    distribution : np.ndarray
        Shape (max_age + 1,): the probability of each age 0, 1, ..., max_age,
        zero below min_age, summing to 1.

    Raises
    ------
    ValueError
        If mortality or juvenile_mortality is not positive and finite, if
        growth is not finite, if mortality + growth is not positive, if
        maturity_age is not an integer from 0 to max_age, or if min_age and
        max_age are not integers with 0 <= min_age <= max_age and
        1 <= max_age <= 400.
    '''
    return distribution  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sampled_age_distribution(mortality: float, juvenile_mortality: float, maturity_age: int, growth: float, min_age: int, max_age: int) -> "np.ndarray":
    if not (np.isfinite(mortality) and mortality > 0.0):
        raise ValueError("mortality must be positive and finite")
    if not (np.isfinite(juvenile_mortality) and juvenile_mortality > 0.0):
        raise ValueError("juvenile_mortality must be positive and finite")
    if not np.isfinite(growth):
        raise ValueError("growth must be finite")
    if not mortality + growth > 0.0:
        raise ValueError("mortality + growth must be positive")
    for v in (min_age, max_age, maturity_age):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("min_age, max_age and maturity_age must be integers")
    if not 0 <= maturity_age <= max_age:
        raise ValueError("need 0 <= maturity_age <= max_age")
    if not (1 <= max_age <= 400 and 0 <= min_age <= max_age):
        raise ValueError("need 0 <= min_age <= max_age and 1 <= max_age <= 400")
    ages = np.arange(max_age + 1, dtype=float)
    lived = (juvenile_mortality * np.minimum(ages, float(maturity_age))      # the juvenile rate, then the adult one
             + mortality * np.maximum(ages - float(maturity_age), 0.0))
    weights = np.where(ages >= min_age, np.exp(-lived - growth * ages), 0.0)  # survival times the birth cohort's size
    return weights / weights.sum()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a growing population, gear retaining ages 4 and over ---
        {
            "setup": "",
            "call": "sampled_age_distribution(0.17, 0.38, 7, 0.05, 4, 150)",
            "gold_call": "_oracle_sampled_age_distribution(0.17, 0.38, 7, 0.05, 4, 150)",
        },
        # --- boundary: constant births and no gear limit ---
        {
            "setup": "",
            "call": "sampled_age_distribution(0.2, 0.45, 5, 0.0, 0, 30)",
            "gold_call": "_oracle_sampled_age_distribution(0.2, 0.45, 5, 0.0, 0, 30)",
        },
        # --- edge: births declining almost as fast as animals die, so the cutoff at 60 matters ---
        {
            "setup": "",
            "call": "sampled_age_distribution(0.09, 0.3, 9, -0.07, 2, 60)",
            "gold_call": "_oracle_sampled_age_distribution(0.09, 0.3, 9, -0.07, 2, 60)",
        },
    ]
