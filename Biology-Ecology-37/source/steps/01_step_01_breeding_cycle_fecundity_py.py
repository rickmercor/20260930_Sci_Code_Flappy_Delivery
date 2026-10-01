"""
Long-run proportion of adult females with a calf of the year, followed by the fecundity of a female at each requested age relative to an average adult female.

Females of many long-lived marine mammals cannot give birth every year: gestation

takes more than a year, a calf depends on its mother for most of the next year, and

some females rest for several years between calves. A first-order Markov chain over

annual breeding states captures this skip-breeding. Two quantities of the chain feed

close-kin kinship probabilities: the long-run share of adult females that have a calf

of the year, which scales the total reproductive output of the population, and the

chance that a female of a given age has a calf, which ramps up and oscillates over

the first years after maturity before it settles at that long-run share.

Returns
-------
np.ndarray of shape (K + 1,), the long-run proportion of adult females with a calf followed by the relative fecundity at each requested age
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def breeding_cycle_fecundity(psi2: float, psi3: float, entry_age: int, ages: "np.ndarray") -> "np.ndarray":
    '''Long-run proportion of adult females with a calf of the year, followed by the fecundity of a female at each requested age relative to an average adult female.

    Adult females move once a year between three breeding states: pregnant,
    with a calf born that year, and resting. A pregnant female always has a
    calf the next year; a female with a calf becomes pregnant again the next
    year with probability psi2 and otherwise rests; a resting female becomes
    pregnant with probability psi3 and otherwise keeps resting. A female
    enters the cycle in the resting state at age entry_age. The long-run
    proportion is the calf-state probability of the stationary distribution
    of the cycle. The fecundity at age a is the probability that a female of
    age a is in the calf state, divided by that long-run proportion; it is
    zero at every age below entry_age.

    Parameters
    ----------
    psi2 : float
        Probability of becoming pregnant in the year after a calf, a finite
        number with 0 < psi2 < 1.
    psi3 : float
        Probability that a resting female becomes pregnant, a finite number
        with 0 < psi3 < 1.
    entry_age : int
        Age in years at which a female enters the cycle, resting, an integer
        from 1 to 20.
    ages : np.ndarray
        Shape (K,), K >= 1; ages in years, integers from 0 to 200, in any
        order.

    Returns
    -------
    summary : np.ndarray
        Shape (K + 1,). Element 0 is the long-run proportion of adult females
        with a calf of the year; element 1 + k is the fecundity at age
        ages[k].

    Raises
    ------
    ValueError
        If psi2 or psi3 is not a finite number strictly between 0 and 1, if
        entry_age is not an integer from 1 to 20, or if ages is not a
        non-empty one-dimensional array of integers from 0 to 200.
    '''
    return summary  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _bcf_probability(name: str, value: float) -> complex:
    """A breeding probability strictly inside (0, 1); complex values with that real part pass (complex-step use)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, complex, np.number)):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value) or not 0.0 < float(np.real(value)) < 1.0:
        raise ValueError(f"{name} must be a finite number strictly between 0 and 1")
    return value


def _oracle_breeding_cycle_fecundity(psi2: float, psi3: float, entry_age: int, ages: "np.ndarray") -> "np.ndarray":
    p2 = _bcf_probability("psi2", psi2)
    p3 = _bcf_probability("psi3", psi3)
    if isinstance(entry_age, bool) or not isinstance(entry_age, (int, np.integer)) or not 1 <= entry_age <= 20:
        raise ValueError("entry_age must be an integer from 1 to 20")
    a = np.asarray(ages)
    if a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 200:
        raise ValueError("ages must be a non-empty 1-D array of integers from 0 to 200")
    # states (pregnant, calf, resting); column-stochastic: next = T @ current
    zero = 0.0 * p2
    T = np.array([[zero, p2, p3], [zero + 1.0, zero, zero], [zero, 1.0 - p2, 1.0 - p3]])
    beta2 = p3 / (2.0 * p3 + (1.0 - p2))        # calf-state share of the stationary distribution (1 - p2 exact)
    steps = int(a.max()) - entry_age
    calf = [zero] * (max(steps, 0) + 1)
    state = np.array([zero, zero, zero + 1.0])   # resting at entry
    for t in range(max(steps, 0) + 1):
        calf[t] = state[1]
        state = T @ state
    fec = [(calf[int(x) - entry_age] if x >= entry_age else zero) / beta2 for x in a]
    return np.array([beta2] + fec)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a skip-breeding cycle, ages across the first years after maturity and later ---
        {
            "setup": "import numpy as np\n",
            "call": "breeding_cycle_fecundity(0.16, 0.38, 4, np.array([4, 5, 6, 7, 8, 9, 10, 12, 20]))",
            "gold_call": "_oracle_breeding_cycle_fecundity(0.16, 0.38, 4, np.array([4, 5, 6, 7, 8, 9, 10, 12, 20]))",
        },
        # --- boundary: ages up to the first age at which a calf is possible, in reverse order ---
        {
            "setup": "import numpy as np\n",
            "call": "breeding_cycle_fecundity(0.3, 0.6, 3, np.array([5, 4, 3, 2, 0]))",
            "gold_call": "_oracle_breeding_cycle_fecundity(0.3, 0.6, 3, np.array([5, 4, 3, 2, 0]))",
        },
        # --- edge: calving almost never repeated at two years, fast return from rest, late entry, long horizon ---
        {
            "setup": "import numpy as np\n",
            "call": "breeding_cycle_fecundity(0.02, 0.9, 7, np.arange(7, 70, 3))",
            "gold_call": "_oracle_breeding_cycle_fecundity(0.02, 0.9, 7, np.arange(7, 70, 3))",
        },
    ]
