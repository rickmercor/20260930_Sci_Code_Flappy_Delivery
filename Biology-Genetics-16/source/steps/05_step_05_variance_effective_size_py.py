"""
'''Variance effective population size from census size and offspring variance.




    Use the large-population form N_E = 4 N / (2 + V_o) (Wright 1938), with N the

    census size and V_o the variance in offspring number.




    Parameters

    ----------

    census_size : int

        Census number of diploid individuals, N >= 1.

    offspring_variance : float

        Variance in offspring number in the absence of additive genetic variance

        for fitness, V_o >= 0.




    Returns

    -------

    n_effective : float

        Variance effective size as a native Python float.




    Raises

    ------

    ValueError

        If census_size is not an integer >= 1, or if offspring_variance is not a

        finite real number >= 0.

    '''

Two different effective sizes appear in this analysis and they are not

interchangeable. The size that governs how fast gametic-phase disequilibrium

decays is the inbreeding effective size. The size that governs the variance of

allele-frequency change - and therefore the covariance matrix the replicate

observations must be weighted by - is the VARIANCE effective size, which is set by

the variance in offspring number among individuals:




    N_E = 4N / (2 + V_o)




with N the census size and V_o the variance in offspring number in the absence of

additive genetic variance for fitness. Multinomial (Wright-Fisher) reproduction

gives V_o approximately 2 and hence N_E approximately N, which is why the

distinction is easy to overlook; any real reproductive system with over-dispersed

offspring number makes N_E substantially smaller than N.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variance_effective_size(census_size: int, offspring_variance: float) -> float:
    '''Variance effective population size from census size and offspring variance.

    Use the large-population form N_E = 4 N / (2 + V_o) (Wright 1938), with N the
    census size and V_o the variance in offspring number.

    Parameters
    ----------
    census_size : int
        Census number of diploid individuals, N >= 1.
    offspring_variance : float
        Variance in offspring number in the absence of additive genetic variance
        for fitness, V_o >= 0.

    Returns
    -------
    n_effective : float
        Variance effective size as a native Python float.

    Raises
    ------
    ValueError
        If census_size is not an integer >= 1, or if offspring_variance is not a
        finite real number >= 0.
    '''
    return n_effective  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_variance_effective_size(census_size: int, offspring_variance: float) -> float:
    if isinstance(census_size, bool) or not isinstance(census_size, (int, np.integer)):
        raise ValueError("census_size must be an integer >= 1")
    if int(census_size) < 1:
        raise ValueError("census_size must be an integer >= 1")
    if isinstance(offspring_variance, bool) or not isinstance(
            offspring_variance, (int, float, np.floating, np.integer)):
        raise ValueError("offspring_variance must be a finite real number >= 0")
    v = float(offspring_variance)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError("offspring_variance must be a finite real number >= 0")
    return float(4.0 * int(census_size) / (2.0 + v))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped replicates, V_o = 6 so N_E = N / 2 ---
        {
            "setup": "import numpy as np\n",
            "call": "[variance_effective_size(n, 6.0) for n in (2000, 1200, 700, 400, 250, 150)]",
            "gold_call": "[_oracle_variance_effective_size(n, 6.0) for n in (2000, 1200, 700, 400, 250, 150)]",
        },
        # --- boundary: Wright-Fisher reproduction, V_o = 2, where N_E == N ---
        {
            "setup": "import numpy as np\n",
            "call": "variance_effective_size(500, 2.0)",
            "gold_call": "_oracle_variance_effective_size(500, 2.0)",
        },
        # --- edge: V_o = 0 (every parent contributes equally) doubles N_E ---
        {
            "setup": "import numpy as np\n",
            "call": "variance_effective_size(1, 0.0)",
            "gold_call": "_oracle_variance_effective_size(1, 0.0)",
        },
    ]
