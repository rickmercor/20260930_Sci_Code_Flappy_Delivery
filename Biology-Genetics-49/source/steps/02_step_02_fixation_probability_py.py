"""
Probability that a mutant starting at a given frequency is eventually fixed, in the diffusion approximation for a haploid population whose census size depends on the mutant's frequency.

The probability that a new mutant escapes loss and eventually takes over a

population is the classical measure of how much selection can achieve against

random genetic drift. The standard result treats the strength of drift as fixed

while the mutant spreads. When the census size, and with it the effective size,

changes with the mutant's frequency, drift is weaker or stronger at different

stages of the spread, and the early, low-frequency stage, when the mutant is most

exposed to loss, weighs most. This step computes the fixation probability in the

diffusion approximation, with drift set by the census size of the previous step

and by the variance in offspring number.

Returns
-------
float, the fixation probability, as a native Python float, accurate to a relative error below 1e-10
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fixation_probability(initial_frequency: float, selection: float, ancestral_size: float, mutant_size: float,
                         response: str, offspring_variance: float) -> float:
    '''Probability that a mutant starting at a given frequency is eventually fixed, in the diffusion approximation for a haploid population whose census size depends on the mutant's frequency.

    The population reproduces as a haploid Wright-Fisher population in which
    the number of offspring of an individual has variance offspring_variance,
    so genetic drift acts at the effective size N(p) / offspring_variance,
    where N(p) is the census size of the previous step for `response`,
    ancestral_size and mutant_size at mutant frequency p. Selection changes
    the mutant frequency by selection * p * (1 - p) per generation. Return the
    probability of ultimate fixation of the mutant from initial_frequency, in
    the diffusion approximation.

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
        "s_shaped", as in the previous step.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10. The product
        selection * max(ancestral_size, mutant_size) / offspring_variance must
        not exceed 200.

    Returns
    -------
    probability : float
        The fixation probability, as a native Python float, accurate to a
        relative error below 1e-10.

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
    return probability  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy import integrate


def _check_diffusion_inputs(initial_frequency: float, selection: float, ancestral_size: float, mutant_size: float,
                            response: str, offspring_variance: float) -> tuple:
    for name, value in (("initial_frequency", initial_frequency), ("selection", selection),
                        ("offspring_variance", offspring_variance)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
                or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    p0, s, var = float(initial_frequency), float(selection), float(offspring_variance)
    if not 0.0 < p0 <= 0.1:
        raise ValueError("initial_frequency must satisfy 0 < initial_frequency <= 0.1")
    if not 0.0 <= s <= 0.05:
        raise ValueError("selection must be a finite number from 0 to 0.05")
    if not 0.1 <= var <= 10.0:
        raise ValueError("offspring_variance must be a finite number from 0.1 to 10")
    na = _check_size("ancestral_size", ancestral_size, 2.0)
    nm = _check_size("mutant_size", mutant_size, 2.0)
    if max(na, nm) > 1e6:
        raise ValueError("ancestral_size and mutant_size must be finite numbers from 2 to 1e6")
    if max(na, nm) > 100.0 * min(na, nm):
        raise ValueError("the larger census size must not exceed 100 times the smaller")
    kind = _check_response(response)
    if s * max(na, nm) / var > 200.0:
        raise ValueError("selection * max(ancestral_size, mutant_size) / offspring_variance must not exceed 200")
    return p0, s, na, nm, kind, var


def _break_points(lower: float, upper: float, rates: tuple) -> list:
    """Interior points at a few multiples of each decay length 1/rate, to guide adaptive quadrature."""
    points = set()
    for rate in rates:
        if rate > 0.0:
            for multiple in (0.5, 2.0, 8.0, 32.0, 128.0):
                x = lower + multiple / rate
                if lower < x < upper:
                    points.add(x)
    return sorted(points) or None


def _oracle_fixation_probability(initial_frequency: float, selection: float, ancestral_size: float, mutant_size: float,
                                 response: str, offspring_variance: float) -> float:
    p0, s, na, nm, kind, var = _check_diffusion_inputs(initial_frequency, selection, ancestral_size, mutant_size,
                                                       response, offspring_variance)
    if s == 0.0:
        return p0
    c = 2.0 * s / var

    def _weight(y):
        return math.exp(-c * _oracle_size_response(np.array([y]), na, nm, kind)[0, 1])

    options = dict(epsabs=0.0, epsrel=1e-12, limit=500)
    head = integrate.quad(_weight, 0.0, p0, **options)[0]
    tail = integrate.quad(_weight, p0, 1.0, points=_break_points(p0, 1.0, (c * min(na, nm), c * max(na, nm))),
                          **options)[0]
    return float(head / (head + tail))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a single cell of a strain that raises the census size sixteen-fold, S-shaped response ---
        {
            "setup": "import numpy as np\n",
            "call": "fixation_probability(1.0 / 800.0, 0.002, 800.0, 12800.0, 's_shaped', 2.0)",
            "gold_call": "_oracle_fixation_probability(1.0 / 800.0, 0.002, 800.0, 12800.0, 's_shaped', 2.0)",
        },
        # --- boundary: equal census sizes, so the size never changes ---
        {
            "setup": "import numpy as np\n",
            "call": "fixation_probability(0.05, 0.01, 300.0, 300.0, 'harmonic', 1.0)",
            "gold_call": "_oracle_fixation_probability(0.05, 0.01, 300.0, 300.0, 'harmonic', 1.0)",
        },
        # --- edge: stronger selection on a strain whose spread shrinks the population to under a sixth ---
        {
            "setup": "import numpy as np\n",
            "call": "fixation_probability(1.0 / 1000.0, 0.01, 1000.0, 150.0, 's_shaped', 1.5)",
            "gold_call": "_oracle_fixation_probability(1.0 / 1000.0, 0.01, 1000.0, 150.0, 's_shaped', 1.5)",
        },
    ]
