"""
Probability that a random pair of haplotypes first coalesces at each generation before the present, at a neutral position linked to a sweeping favourable mutation.

At a neutral position near a sweeping favourable mutation, the chance that two

haplotypes find their common ancestor in a given past generation depends on three

things at once: how many breeders the population had then, how much background

selection at deleterious sites reduced the effective size, and how strongly the sweep

concentrated the ancestry on the few chromosomes that carried the favourable allele.

This step combines the three into the distribution of the first coalescence generation

of a random pair at such a position.

Returns
-------
np.ndarray, shape (horizon,): weights[g - 1] is the probability that the pair first coalesces exactly g generations before the present, for g = 1 to horizon
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def focal_coalescence_weights(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                              since_end: int, mutation_rate: float, selection: float, chromosome_length: float,
                              advantage: float, recombination: float, sweep_age: int, horizon: int) -> "np.ndarray":
    '''Probability that a random pair of haplotypes first coalesces at each generation before the present, at a neutral position linked to a sweeping favourable mutation.

    Generation g before the present (g = 1 is the parents of the present
    generation) has N(g) breeding individuals, with N(g) = n_recent for
    1 <= g <= since_end, N(g) = n_bottleneck for
    since_end < g <= since_end + bottleneck_length, and N(g) = n_ancestral for
    larger g. Background selection multiplies the effective size of the
    neutral position at generation g by F(g), the factor of the
    reduction-factor step after g generations of association for the census
    N(g). A favourable mutation arose as a single copy sweep_age generations
    before the present, after the bottleneck ended, at a site at recombination
    fraction recombination from the neutral position; its expected trajectory
    is that of the trajectory step for census n_recent and the given advantage
    over sweep_age generations, and the sweep multiplies the effective size at
    generation g by S(g), the factor of the sweep size-reduction step for that
    trajectory, census n_recent and recombination fraction, for
    1 <= g <= sweep_age, and by S(g) = 1 for larger g. The effective size of
    the neutral position at generation g is N(g) * F(g) * S(g). Two haplotypes
    sampled in the present that have not coalesced in any generation more
    recent than g coalesce in generation g with probability
    1 / (2 * N(g) * F(g) * S(g)), independently across generations and without
    any continuous-time approximation.

    Parameters
    ----------
    n_ancestral : float
        Number of breeding individuals before the bottleneck, >= 1.
    n_bottleneck : float
        Number of breeding individuals during the bottleneck, >= 1.
    n_recent : float
        Number of breeding individuals since the bottleneck ended, >= 1.
    bottleneck_length : int
        Number of generations the bottleneck lasted, >= 0.
    since_end : int
        Number of generations since the bottleneck ended, >= 1.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    advantage : float
        Fitness advantage of a heterozygous carrier of the favourable
        mutation, > 0.
    recombination : float
        Recombination fraction between the selected site and the neutral
        position, in [0, 0.5].
    sweep_age : int
        Number of generations since the favourable mutation arose, an integer
        in [1, since_end].
    horizon : int
        Last generation before the present that is included, >= 1.

    Returns
    -------
    weights : np.ndarray
        Shape (horizon,): weights[g - 1] is the probability that the pair
        first coalesces exactly g generations before the present, for g = 1 to
        horizon.

    Raises
    ------
    ValueError
        If n_ancestral, n_bottleneck or n_recent is not a finite number >= 1,
        if bottleneck_length is not an integer >= 0, if since_end is not an
        integer >= 1, if sweep_age is not an integer in [1, since_end], if
        horizon is not an integer >= 1, if the coalescence probability of some
        generation exceeds 1, or on any condition raised by the previous steps
        for the other inputs.
    '''
    return weights  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_focal_coalescence_weights(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                                      since_end: int, mutation_rate: float, selection: float, chromosome_length: float,
                                      advantage: float, recombination: float, sweep_age: int, horizon: int) -> "np.ndarray":
    def _size(v):
        return ((not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating))
                and bool(np.isfinite(float(v))) and float(v) >= 1.0)

    def _count(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, np.integer))

    if not (_size(n_ancestral) and _size(n_bottleneck) and _size(n_recent)):
        raise ValueError("n_ancestral, n_bottleneck and n_recent must be finite numbers >= 1")
    if not (_count(bottleneck_length) and int(bottleneck_length) >= 0):
        raise ValueError("bottleneck_length must be an integer >= 0")
    if not (_count(since_end) and int(since_end) >= 1):
        raise ValueError("since_end must be an integer >= 1")
    if not (_count(sweep_age) and 1 <= int(sweep_age) <= int(since_end)):
        raise ValueError("sweep_age must be an integer in [1, since_end]")
    if not (_count(horizon) and int(horizon) >= 1):
        raise ValueError("horizon must be an integer >= 1")
    g = np.arange(1, int(horizon) + 1)
    e, dd, a = int(since_end), int(bottleneck_length), int(sweep_age)
    size = np.where(g <= e, float(n_recent), np.where(g <= e + dd, float(n_bottleneck), float(n_ancestral)))
    background = np.empty(g.size)
    for census in (float(n_recent), float(n_bottleneck), float(n_ancestral)):   # each epoch's census sets its variance
        in_epoch = size == census
        if np.any(in_epoch):
            background[in_epoch] = _oracle_selection_reduction_factor(g[in_epoch], mutation_rate, selection,
                                                                      chromosome_length, census)
    trajectory = _oracle_sweep_frequency_trajectory(n_recent, advantage, a)
    sweep = np.ones(g.size)
    reduction = _oracle_sweep_size_reduction(trajectory, n_recent, recombination)
    reach = min(a, g.size)
    sweep[:reach] = reduction[:reach]
    rate = 1.0 / (2.0 * size * background * sweep)
    if np.any(rate > 1.0):
        raise ValueError("the coalescence probability of some generation exceeds 1")
    # probability of no coalescence in the generations more recent than g, then coalescence at g
    not_yet = np.concatenate([[1.0], np.cumprod(1.0 - rate)[:-1]])
    return rate * not_yet

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped history, chromosome and sweep ---
        {
            "setup": "import numpy as np\n",
            "call": "focal_coalescence_weights(8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 21, 3000)",
            "gold_call": "_oracle_focal_coalescence_weights(8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 21, 3000)",
        },
        # --- boundary: complete linkage and a sweep that arose in the generation the bottleneck ended ---
        {
            "setup": "import numpy as np\n",
            "call": "focal_coalescence_weights(8000, 150, 600, 35, 18, 0.5, 0.02, 1.0, 0.3, 0.0, 18, 400)",
            "gold_call": "_oracle_focal_coalescence_weights(8000, 150, 600, 35, 18, 0.5, 0.02, 1.0, 0.3, 0.0, 18, 400)",
        },
        # --- edge: a horizon shorter than the sweep, loose linkage and another chromosome ---
        {
            "setup": "import numpy as np\n",
            "call": "focal_coalescence_weights(2000, 60, 900, 12, 40, 1.0, 0.05, 2.0, 0.45, 0.2, 30, 20)",
            "gold_call": "_oracle_focal_coalescence_weights(2000, 60, 900, 12, 40, 1.0, 0.05, 2.0, 0.45, 0.2, 30, 20)",
        },
    ]
