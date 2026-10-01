"""
Expected frequency of a favourable mutation in each generation since it arose as a single copy.

A favourable mutation that is destined to spread starts as a single copy, and its early

fate is governed by chance: while it is rare, its copy number behaves like a branching

process rather than following the deterministic effect of selection. Once it is common

enough for selection to dominate drift, its frequency follows the familiar deterministic

curve. Predictions of the effect of a sweep on linked neutral variation need the

expected trajectory of the allele through both phases. This step computes that expected

trajectory from the generation the mutation arose.

Returns
-------
np.ndarray, shape (generations + 1,): q[k] is the expected frequency of the favourable mutation k generations after it arose, in (0, 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sweep_frequency_trajectory(census: float, advantage: float, generations: int) -> "np.ndarray":
    '''Expected frequency of a favourable mutation in each generation since it arose as a single copy.

    A randomly mating population has census diploid individuals, that is 2 *
    census haploid genomes, in every generation. A favourable mutation arises
    as a single copy, so its frequency in the generation it arises is
    q_0 = 1 / (2 * census). Individuals carrying zero, one and two copies have
    relative fitness 1, 1 + advantage and 1 + 2 * advantage. Its expected
    trajectory has two phases. While q_k < 1 / (2 * census * advantage), the
    number of copies grows by exactly one per generation,
    q_{k+1} = q_k + 1 / (2 * census). From the first generation with
    q_k >= 1 / (2 * census * advantage) on, the frequency follows the
    deterministic selection recursion
    q_{k+1} = q_k * (1 + advantage + advantage * q_k) / (1 + 2 * advantage * q_k).
    Return q_k for k = 0 to generations, k counting the generations since the
    mutation arose.

    Parameters
    ----------
    census : float
        Number of breeding diploid individuals, >= 1.
    advantage : float
        Fitness advantage of a heterozygous carrier, > 0.
    generations : int
        Number of generations since the mutation arose, >= 0.

    Returns
    -------
    q : np.ndarray
        Shape (generations + 1,): q[k] is the expected frequency of the
        favourable mutation k generations after it arose, in (0, 1).

    Raises
    ------
    ValueError
        If census is not a finite number >= 1, if advantage is not a finite
        number > 0, if generations is not an integer >= 0, or if the expected
        frequency reaches 1 within the requested generations.
    '''
    return q  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sweep_frequency_trajectory(census: float, advantage: float, generations: int) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(census) or float(census) < 1.0:
        raise ValueError("census must be a finite number >= 1")
    if not _num(advantage) or float(advantage) <= 0.0:
        raise ValueError("advantage must be a finite number > 0")
    if isinstance(generations, bool) or not isinstance(generations, (int, np.integer)) or int(generations) < 0:
        raise ValueError("generations must be an integer >= 0")
    n, a = float(census), float(advantage)
    one_copy = 1.0 / (2.0 * n)
    threshold = 1.0 / (2.0 * n * a)          # the frequency at which selection overtakes the branching phase
    q = np.empty(int(generations) + 1)
    q[0] = one_copy
    for k in range(int(generations)):
        x = q[k]
        if x < threshold:
            q[k + 1] = x + one_copy            # branching phase: one more copy per generation
        else:
            q[k + 1] = x * (1.0 + a + a * x) / (1.0 + 2.0 * a * x)   # deterministic selection
        if not q[k + 1] < 1.0:
            raise ValueError("the expected frequency reaches 1 within the requested generations")
    return q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the recent census and the shipped advantage over the sweep's age ---
        {
            "setup": "import numpy as np\n",
            "call": "sweep_frequency_trajectory(600, 0.3, 21)",
            "gold_call": "_oracle_sweep_frequency_trajectory(600, 0.3, 21)",
        },
        # --- boundary: a threshold just below four copies, where the branching phase ends at four copies ---
        {
            "setup": "import numpy as np\n",
            "call": "sweep_frequency_trajectory(600, 0.2503, 12)",
            "gold_call": "_oracle_sweep_frequency_trajectory(600, 0.2503, 12)",
        },
        # --- edge: a small population under strong selection, carried close to fixation ---
        {
            "setup": "import numpy as np\n",
            "call": "sweep_frequency_trajectory(50, 0.45, 40)",
            "gold_call": "_oracle_sweep_frequency_trajectory(50, 0.45, 40)",
        },
        # --- edge: a weakly favoured allele in a large population, a long branching phase ---
        {
            "setup": "import numpy as np\n",
            "call": "sweep_frequency_trajectory(5000, 0.07, 60)",
            "gold_call": "_oracle_sweep_frequency_trajectory(5000, 0.07, 60)",
        },
    ]
