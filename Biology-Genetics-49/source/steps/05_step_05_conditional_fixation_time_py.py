"""
Expected number of generations until a mutant fixes, conditional on its fixation, in the diffusion approximation for a haploid population whose census size depends on the mutant's frequency.

Whether a mutant fixes is only half of what matters for the pace of adaptation; the

other half is how long a successful sweep takes. The time spent at each frequency

depends on the balance of selection and drift there, so a census size that rises or

falls as the mutant spreads lengthens or shortens different stages of the sweep:

the rare stage near the ancestral size and the common stage near the size of a

population fixed for the mutant. This step computes the expected duration of the

sweep, counted only over the histories in which the mutant fixes, in the diffusion

approximation of the previous steps.

Returns
-------
float, the conditional mean time to fixation in generations, as a native Python float, accurate to a relative error below 1e-8
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def conditional_fixation_time(initial_frequency: float, selection: float, ancestral_size: float, mutant_size: float,
                              response: str, offspring_variance: float) -> float:
    '''Expected number of generations until a mutant fixes, conditional on its fixation, in the diffusion approximation for a haploid population whose census size depends on the mutant's frequency.

    The population, its census size N(p) for `response`, ancestral_size and
    mutant_size, the effective size N(p) / offspring_variance and the change
    selection * p * (1 - p) per generation are those of the previous steps.
    Starting from initial_frequency, return the mean number of generations
    until the mutant is fixed, averaged over the histories that end in
    fixation, in the diffusion approximation.

    Parameters
    ----------
    initial_frequency : float
        Mutant frequency at the start, a finite number with
        0 < initial_frequency <= 0.1.
    selection : float
        Selective advantage of the mutant per generation, a finite number from
        0 to 0.05.
    ancestral_size : float
        Census size with the mutant absent, a finite number from 2 to 1e6.
    mutant_size : float
        Census size with the mutant fixed, a finite number from 2 to 1e6, at
        most 100 times and at least 1/100 of ancestral_size.
    response : str
        One of "arithmetic", "geometric", "harmonic", "root_mean_square" and
        "s_shaped", as in the first step.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10. The product
        selection * max(ancestral_size, mutant_size) / offspring_variance must
        not exceed 200.

    Returns
    -------
    generations : float
        The conditional mean time to fixation in generations, as a native
        Python float, accurate to a relative error below 1e-8.

    Raises
    ------
    ValueError
        If initial_frequency is not a finite number with
        0 < initial_frequency <= 0.1, if selection is not a finite number from
        0 to 0.05, if ancestral_size or mutant_size is not a finite number from
        2 to 1e6, if the larger size exceeds 100 times the smaller, if
        response is not one of the five names, if offspring_variance is not a
        finite number from 0.1 to 10, or if
        selection * max(ancestral_size, mutant_size) / offspring_variance
        exceeds 200.
    '''
    return generations  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy import integrate


def _oracle_conditional_fixation_time(initial_frequency: float, selection: float, ancestral_size: float,
                                      mutant_size: float, response: str, offspring_variance: float) -> float:
    p0, s, na, nm, kind, var = _check_diffusion_inputs(initial_frequency, selection, ancestral_size, mutant_size,
                                                       response, offspring_variance)
    c = 2.0 * s / var
    rates = (c * min(na, nm), c * max(na, nm))
    inner = dict(epsabs=0.0, epsrel=1e-12, limit=500)
    outer = dict(epsabs=0.0, epsrel=1e-11, limit=500)

    def _row(x):
        return _oracle_size_response(np.array([x]), na, nm, kind)[0]

    if s == 0.0:
        # neutral: u(x) = x, the tail ratio is 1 - x and the head ratio is x
        drift = 2.0 / var
        first = drift * (_row(1.0)[1] - _row(p0)[1])
        second = integrate.quad(lambda x: drift * _row(x)[0] * x / (1.0 - x), 0.0, p0, **outer)[0]
        return float(first + (1.0 - p0) / p0 * second)

    def _weight(y):
        return math.exp(-c * _row(y)[1])

    head = integrate.quad(_weight, 0.0, p0, **inner)[0]
    total = head + integrate.quad(_weight, p0, 1.0, points=_break_points(p0, 1.0, rates), **inner)[0]

    def _u(x):
        if x <= p0:
            return integrate.quad(_weight, 0.0, x, **inner)[0] / total
        return (head + integrate.quad(_weight, p0, x, points=_break_points(p0, x, rates), **inner)[0]) / total

    def _tail_ratio(x):
        # integral from x to 1 of the weight, divided by the weight at x: no cancellation, no underflow
        fx = _row(x)[1]
        return integrate.quad(lambda y: math.exp(-c * (_row(y)[1] - fx)), x, 1.0,
                              points=_break_points(x, 1.0, rates), **inner)[0]

    def _sojourn_after(x):
        return 2.0 * _row(x)[0] * _u(x) * _tail_ratio(x) / (var * x * (1.0 - x))

    f_start = _row(p0)[1]

    def _sojourn_before(x):
        # the term below the start frequency, with its factor (1 - u0) = weight(p0) * tail_ratio(p0) / total taken inside
        head_x = integrate.quad(_weight, 0.0, x, **inner)[0]
        return 2.0 * _row(x)[0] * head_x * head_x * math.exp(-c * (f_start - _row(x)[1])) / (var * x * (1.0 - x))

    first = integrate.quad(_sojourn_after, p0, 1.0, points=_break_points(p0, 1.0, rates), **outer)[0]
    second = integrate.quad(_sojourn_before, 0.0, p0, **outer)[0]
    u0 = head / total
    return float(first + _tail_ratio(p0) * second / (u0 * total * total))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a single cell of a strain that raises the census size four-fold, S-shaped response ---
        {
            "setup": "import numpy as np\n",
            "call": "conditional_fixation_time(1.0 / 800.0, 0.003, 800.0, 3200.0, 's_shaped', 2.0)",
            "gold_call": "_oracle_conditional_fixation_time(1.0 / 800.0, 0.003, 800.0, 3200.0, 's_shaped', 2.0)",
            "tol": 1e-7,
        },
        # --- boundary: a neutral mutant, arithmetic response ---
        {
            "setup": "import numpy as np\n",
            "call": "conditional_fixation_time(1.0 / 400.0, 0.0, 400.0, 1600.0, 'arithmetic', 1.0)",
            "gold_call": "_oracle_conditional_fixation_time(1.0 / 400.0, 0.0, 400.0, 1600.0, 'arithmetic', 1.0)",
            "tol": 1e-7,
        },
        # --- edge: stronger selection on a strain whose spread shrinks the population to under a sixth ---
        {
            "setup": "import numpy as np\n",
            "call": "conditional_fixation_time(1.0 / 1000.0, 0.01, 1000.0, 150.0, 's_shaped', 1.5)",
            "gold_call": "_oracle_conditional_fixation_time(1.0 / 1000.0, 0.01, 1000.0, 150.0, 's_shaped', 1.5)",
            "tol": 1e-7,
        },
    ]
