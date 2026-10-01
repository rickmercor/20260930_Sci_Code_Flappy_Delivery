"""
Probability that a random pair of haplotypes first coalesces at each generation before the present.

How far back a random pair of haplotypes finds its common ancestor depends on the

history of the population's size: small numbers of breeders make recent coalescence

likely, and a bottleneck that has since been left behind leaves its mark as an excess

of coalescence events in the generations it spanned. This step gives, generation by

generation, the probability that a pair sampled today first coalesces at that

generation, for a history with three constant-size epochs whose sizes are reduced by

background selection according to the age of the association and to the standing

fitness variance that each epoch's census sustains.

Returns
-------
np.ndarray, shape (horizon,): weights[g - 1] is the probability that the pair first coalesces exactly g generations before the present, for g = 1 to horizon
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coalescence_weights(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                        since_end: int, mutation_rate: float, selection: float, chromosome_length: float, horizon: int) -> "np.ndarray":
    '''Probability that a random pair of haplotypes first coalesces at each generation before the present.

    The population is a randomly mating Wright-Fisher population of diploid
    individuals. Generation g before the present (g = 1 is the parents of the
    present generation) has N(g) breeding individuals, that is 2 N(g) haploid
    genomes, with N(g) = n_recent for 1 <= g <= since_end, N(g) = n_bottleneck
    for since_end < g <= since_end + bottleneck_length, and N(g) = n_ancestral
    for larger g. Background selection reduces the effective size of the
    neutral position at generation g to N(g) F(g), where F(g) is the reduction
    factor of the previous step after g generations of association for the
    census N(g) of that generation (so each epoch's census sets its own
    standing variance), for the given mutation_rate, selection and
    chromosome_length. Two haplotypes sampled
    in the present that have not coalesced in any generation more recent than g
    coalesce in generation g with probability 1 / (2 N(g) F(g)), independently
    across generations and without any continuous-time approximation.

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
        Number of generations since the bottleneck ended, >= 0.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    horizon : int
        Last generation before the present that is included, >= 1.

    Returns
    -------
    weights : np.ndarray
        Shape (horizon,): weights[g - 1] is the probability that the pair first
        coalesces exactly g generations before the present, for g = 1 to
        horizon. The entries sum to less than 1 when coalescence beyond the
        horizon has positive probability.

    Raises
    ------
    ValueError
        If any of the three sizes is not a finite number >= 1, if
        bottleneck_length or since_end is not an integer >= 0, if horizon is
        not an integer >= 1, or on any condition raised by the previous step for
        mutation_rate, selection and chromosome_length.
    '''
    return weights  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_coalescence_weights(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                                since_end: int, mutation_rate: float, selection: float, chromosome_length: float, horizon: int) -> "np.ndarray":
    def _size(v):
        return ((not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating))
                and bool(np.isfinite(float(v))) and float(v) >= 1.0)

    def _count(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, np.integer))

    if not (_size(n_ancestral) and _size(n_bottleneck) and _size(n_recent)):
        raise ValueError("n_ancestral, n_bottleneck and n_recent must be finite numbers >= 1")
    if not (_count(bottleneck_length) and int(bottleneck_length) >= 0 and _count(since_end) and int(since_end) >= 0):
        raise ValueError("bottleneck_length and since_end must be integers >= 0")
    if not (_count(horizon) and int(horizon) >= 1):
        raise ValueError("horizon must be an integer >= 1")
    g = np.arange(1, int(horizon) + 1)
    e, dd = int(since_end), int(bottleneck_length)
    size = np.where(g <= e, float(n_recent), np.where(g <= e + dd, float(n_bottleneck), float(n_ancestral)))
    factor = np.empty(g.size)
    for census in (float(n_recent), float(n_bottleneck), float(n_ancestral)):   # each epoch's census sets its own variance
        in_epoch = size == census
        if np.any(in_epoch):
            factor[in_epoch] = _oracle_selection_reduction_factor(g[in_epoch], mutation_rate, selection, chromosome_length, census)
    rate = 1.0 / (2.0 * size * factor)
    # probability of no coalescence in the generations more recent than g, then coalescence at g
    not_yet = np.concatenate([[1.0], np.cumprod(1.0 - rate)[:-1]])
    return rate * not_yet

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped history and chromosome, with the bottleneck end 26 generations ago ---
        {
            "setup": "import numpy as np\n",
            "call": "coalescence_weights(8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 3000)",
            "gold_call": "_oracle_coalescence_weights(8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 3000)",
        },
        # --- boundary: a constant census, where only the selective reduction varies over generations ---
        {
            "setup": "import numpy as np\n",
            "call": "coalescence_weights(500, 500, 500, 10, 5, 0.5, 0.02, 1.0, 50)",
            "gold_call": "_oracle_coalescence_weights(500, 500, 500, 10, 5, 0.5, 0.02, 1.0, 50)",
        },
        # --- edge: a bottleneck that ended beyond the horizon, so only the recent size is seen ---
        {
            "setup": "import numpy as np\n",
            "call": "coalescence_weights(8000, 150, 600, 35, 40, 0.5, 0.02, 1.0, 30)",
            "gold_call": "_oracle_coalescence_weights(8000, 150, 600, 35, 40, 0.5, 0.02, 1.0, 30)",
        },
        # --- edge: a bottleneck still in progress, on a longer chromosome under stronger selection ---
        {
            "setup": "import numpy as np\n",
            "call": "coalescence_weights(2000, 40, 900, 12, 0, 1.0, 0.05, 2.0, 100)",
            "gold_call": "_oracle_coalescence_weights(2000, 40, 900, 12, 0, 1.0, 0.05, 2.0, 100)",
        },
    ]
