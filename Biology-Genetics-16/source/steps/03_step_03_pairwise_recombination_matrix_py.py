"""
'''Recombination probability between every pair of loci, assuming no interference.




    interval_rates[i] is the crossover probability between locus i and locus i+1,

    for n_loci - 1 intervals along one chromosome in physical order. Assuming

    crossovers occur without interference, convert each interval probability to an

    additive map distance, accumulate distances along the chromosome, and convert

    absolute pairwise distance differences back to recombination probabilities.




    Parameters

    ----------

    interval_rates : np.ndarray

        (n_loci - 1,) array of crossover probabilities, each in [0, 0.5).




    Returns

    -------

    R : np.ndarray

        (n_loci, n_loci) symmetric matrix with a zero diagonal and all entries

        in [0, 0.5).




    Raises

    ------

    ValueError

        If interval_rates is not a 1D array, if it is empty, or if any entry is

        not in [0, 0.5).

    '''

Both the decay of gametic-phase disequilibrium and the covariance of drift across

loci are functions of the recombination probability between every ordered pair of

loci, not just between neighbours. Per-interval crossover probabilities therefore

have to be composed into a full pairwise matrix.




Crossover probabilities do not add. Map distances do, so each interval

probability is converted to a map distance, distances are accumulated along the

chromosome, pairwise distances are differenced, and the result is converted back

to a recombination probability. Under Haldane's mapping function - crossovers as a

Poisson process with no interference - the two conversions are

d = -(1/2) ln(1 - 2r) and r = (1 - exp(-2d)) / 2. The diagonal is exactly zero and

no entry can exceed 1/2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pairwise_recombination_matrix(interval_rates: np.ndarray) -> np.ndarray:
    '''Recombination probability between every pair of loci, assuming no interference.

    interval_rates[i] is the crossover probability between locus i and locus i+1,
    for n_loci - 1 intervals along one chromosome in physical order. Assuming
    crossovers occur without interference, convert each interval probability to an
    additive map distance, accumulate distances along the chromosome, and convert
    absolute pairwise distance differences back to recombination probabilities.

    Parameters
    ----------
    interval_rates : np.ndarray
        (n_loci - 1,) array of crossover probabilities, each in [0, 0.5).

    Returns
    -------
    R : np.ndarray
        (n_loci, n_loci) symmetric matrix with a zero diagonal and all entries
        in [0, 0.5).

    Raises
    ------
    ValueError
        If interval_rates is not a 1D array, if it is empty, or if any entry is
        not in [0, 0.5).
    '''
    return R  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pairwise_recombination_matrix(interval_rates: np.ndarray) -> np.ndarray:
    r = np.asarray(interval_rates, dtype=float)
    if r.ndim != 1:
        raise ValueError("interval_rates must be a 1D array")
    if r.size < 1:
        raise ValueError("interval_rates must contain at least one interval")
    if not np.all(np.isfinite(r)):
        raise ValueError("interval_rates must be finite")
    if np.any(r < 0.0) or np.any(r >= 0.5):
        raise ValueError("every interval rate must lie in [0, 0.5)")

    d = np.concatenate([[0.0], np.cumsum(-0.5 * np.log1p(-2.0 * r))])
    M = np.abs(d[:, None] - d[None, :])
    return 0.5 * (1.0 - np.exp(-2.0 * M))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped map, 9 intervals at 0.05 ---
        {
            "setup": "import numpy as np\nrates = np.full(9, 0.05)\n",
            "call": "pairwise_recombination_matrix(rates)",
            "gold_call": "_oracle_pairwise_recombination_matrix(rates)",
        },
        # --- boundary: complete linkage, every interval rate exactly 0 ---
        {
            "setup": "import numpy as np\nrates = np.zeros(4)\n",
            "call": "pairwise_recombination_matrix(rates)",
            "gold_call": "_oracle_pairwise_recombination_matrix(rates)",
        },
        # --- edge: heterogeneous map with one near-free interval ---
        {
            "setup": "import numpy as np\nrates = np.array([0.001, 0.4999, 0.25, 0.02])\n",
            "call": "pairwise_recombination_matrix(rates)",
            "gold_call": "_oracle_pairwise_recombination_matrix(rates)",
        },
    ]
