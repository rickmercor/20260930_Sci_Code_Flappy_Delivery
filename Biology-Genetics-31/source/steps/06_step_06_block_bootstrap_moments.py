"""
Resample the retained windows with replacement to obtain the mean and standard

deviation of the genome-wide introgression statistic.

Sites within a window are correlated through linkage, so a site-level resampling

would understate the uncertainty of a genome-wide estimate. The block bootstrap

keeps each window intact and resamples whole windows with replacement, preserving

the within-window dependence while treating distant windows as exchangeable. Each

replicate draws as many windows as there are windows to draw from and then

re-evaluates the genome-wide statistic over the windows it drew. The mean of the replicate

values is the genome-wide point estimate and their sample standard deviation, with

one degree of freedom removed, is its standard error, and the two parameterize a

normal test of whether the imbalance differs from zero. The resampling is

reproducible because every replicate draws from a single generator seeded once,

one replicate at a time.

Returns
-------
tuple of two native Python floats, (mean, sd) of the replicate values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def block_bootstrap_moments(numerators: list[float], denominators: list[float], n_replicates: int, seed: int) -> tuple[float, float]:
    """Mean and standard deviation of the block bootstrap replicates.

    Parameters
    ----------
    numerators : sequence of float
        Per-window numerator sums of the statistic, one per retained window.
    denominators : sequence of float
        Per-window denominator sums, one per retained window.
    n_replicates : int
        Number of bootstrap replicates to draw.
    seed : int
        Seed passed to numpy.random.default_rng.

    Returns
    -------
    moments : tuple of float
        The pair (mean, sd), where mean is the mean of the replicate values of
        the genome-wide statistic and sd is their sample standard deviation with
        one degree of freedom removed.

    Raises
    ------
    ValueError
        If the two window sequences differ in length or are empty, if any
        denominator is not positive, or if n_replicates is not at least two.
    """
    return moments

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_block_bootstrap_moments(numerators: list[float], denominators: list[float], n_replicates: int, seed: int) -> tuple[float, float]:
    """Reference implementation."""
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if numerators.ndim != 1 or denominators.ndim != 1:
        raise ValueError("window sums must be one dimensional")
    if numerators.size == 0 or numerators.size != denominators.size:
        raise ValueError("numerators and denominators must cover the same windows")
    if not np.all(denominators > 0.0):
        raise ValueError("every retained window needs a positive denominator")
    if int(n_replicates) != n_replicates or n_replicates < 2:
        raise ValueError("n_replicates must be an integer of at least two")

    n_blocks = numerators.size
    rng = np.random.default_rng(seed)
    replicates = np.empty(int(n_replicates), dtype=float)
    for r in range(int(n_replicates)):
        drawn = rng.integers(0, n_blocks, size=n_blocks)
        replicates[r] = numerators[drawn].sum() / denominators[drawn].sum()
    return float(replicates.mean()), float(replicates.std(ddof=1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    numerators_data = [-0.6666666666666667, 1.0, 0.8, 0.8, 0.8, -1.0, 2.0,
                       1.0, 0.0, 1.9, 1.3]
    denominators_data = [1.6373626373626373, 1.9191919191919191,
                         5.13986013986014, 2.2234432234432235,
                         3.7391941391941392, 1.7417582417582418,
                         5.80944055944056, 3.81018981018981,
                         1.923076923076923, 5.895371295371295,
                         3.5531468531468535]
    return [
        # --- Normal scenario: the eleven retained windows of the benchmark ---
        {
            "setup": f"""numerators = {numerators_data!r}
denominators = {denominators_data!r}
""",
            "call": "[round(v, 12) for v in block_bootstrap_moments(numerators, denominators, 2000, 2026)]",
            "gold_call": "[round(v, 12) for v in _oracle_block_bootstrap_moments(numerators, denominators, 2000, 2026)]",
        },
        # --- Boundary case: a single window makes every replicate identical ---
        {
            "setup": """numerators = [1.3]
denominators = [3.5531468531468535]
""",
            "call": "[round(v, 12) for v in block_bootstrap_moments(numerators, denominators, 5, 7)]",
            "gold_call": "[round(v, 12) for v in _oracle_block_bootstrap_moments(numerators, denominators, 5, 7)]",
        },
        # --- Edge case: a zero denominator must raise ValueError ---
        {
            "setup": """numerators = [1.0, 0.5]
denominators = [2.0, 0.0]
def run_model():
    try:
        block_bootstrap_moments(numerators, denominators, 100, 1)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_block_bootstrap_moments(numerators, denominators, 100, 1)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
