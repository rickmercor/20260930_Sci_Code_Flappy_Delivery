"""
Number of generations since the bottleneck ended whose model coverage of a length class is closest to an observed value.

The coverage of a length class of runs of homozygosity is an observable, and the age of

a demographic event is not. When the sizes of the epochs and the length of the

bottleneck are known from other evidence, the one unknown, how long ago the bottleneck

ended, can be read off the coverage of a class whose runs mostly date from around that

time. This step performs that inversion on the integer grid of candidate values.

Returns
-------
int, the selected number of generations since the bottleneck ended, as a native Python int
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bottleneck_end_from_class_coverage(observed: float, lower: float, upper: float, n_ancestral: float,
                                       n_bottleneck: float, n_recent: float, bottleneck_length: int, mutation_rate: float, selection: float, chromosome_length: float, break_rate: float,
                                       marker_spacing: float, heterozygosity: float, horizon: int,
                                       max_since_end: int) -> int:
    '''Number of generations since the bottleneck ended whose model coverage of a length class is closest to an observed value.

    For each candidate number of generations since the end of the bottleneck,
    from 1 to max_since_end, the model coverage of the class [lower, upper] is
    the expected fraction of genome positions of a random individual that lie
    inside a run of homozygosity whose length is in the class: the sum over
    coalescence generations g = 1 to horizon of the first-coalescence
    probability of the coalescence-weights step at g for that candidate, times
    the class probability of the class-mass step for a pair that coalesced g
    generations ago. Return the candidate whose model coverage is closest to
    observed; if two candidates are equally close, return the smaller.

    Parameters
    ----------
    observed : float
        Observed coverage of the class, a fraction of the genome, in [0, 1].
    lower : float
        Lower end of the length class in Morgans, >= 0.
    upper : float
        Upper end of the length class in Morgans, > lower; may be infinite.
    n_ancestral : float
        Number of breeding individuals before the bottleneck, >= 1.
    n_bottleneck : float
        Number of breeding individuals during the bottleneck, >= 1.
    n_recent : float
        Number of breeding individuals since the bottleneck ended, >= 1.
    bottleneck_length : int
        Number of generations the bottleneck lasted, >= 0.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].
    horizon : int
        Last coalescence generation before the present that is counted, >= 1.
    max_since_end : int
        Largest candidate number of generations since the bottleneck ended,
        >= 1.

    Returns
    -------
    since_end : int
        The selected number of generations since the bottleneck ended, as a
        native Python int.

    Raises
    ------
    ValueError
        If observed is not a finite number in [0, 1], if max_since_end is not
        an integer >= 1, or on any condition raised by the previous steps for
        the other inputs.
    '''
    return since_end  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bottleneck_end_from_class_coverage(observed: float, lower: float, upper: float, n_ancestral: float,
                                               n_bottleneck: float, n_recent: float, bottleneck_length: int, mutation_rate: float, selection: float, chromosome_length: float, break_rate: float,
                                               marker_spacing: float, heterozygosity: float, horizon: int,
                                               max_since_end: int) -> int:
    if (isinstance(observed, bool) or not isinstance(observed, (int, float, np.integer, np.floating))
            or not np.isfinite(float(observed)) or not 0.0 <= float(observed) <= 1.0):
        raise ValueError("observed must be a finite number in [0, 1]")
    if isinstance(max_since_end, bool) or not isinstance(max_since_end, (int, np.integer)) or int(max_since_end) < 1:
        raise ValueError("max_since_end must be an integer >= 1")
    generations = np.arange(1, int(horizon) + 1) if (isinstance(horizon, (int, np.integer)) and not isinstance(horizon, bool)
                                                    and int(horizon) >= 1) else None
    if generations is None:
        raise ValueError("horizon must be an integer >= 1")
    # the class probabilities given the coalescence generation do not depend on the candidate,
    # so they are evaluated once; the coalescence weights are evaluated per candidate
    mass = _oracle_class_mass_given_time(lower, upper, generations, break_rate, marker_spacing, heterozygosity)
    best, best_gap = None, None
    for candidate in range(1, int(max_since_end) + 1):
        weights = _oracle_coalescence_weights(n_ancestral, n_bottleneck, n_recent, bottleneck_length, candidate,
                                              mutation_rate, selection, chromosome_length, horizon)
        gap = abs(float(weights @ mass) - float(observed))
        if best is None or gap < best_gap:
            best, best_gap = candidate, gap
    return int(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the observed 2-4 cM coverage of the shipped configuration, over a shorter candidate range ---
        {
            "setup": "import numpy as np\n",
            "call": "bottleneck_end_from_class_coverage(0.03233, 0.02, 0.04, 8000, 150, 600, 35, 0.5, 0.02, 1.0, 1.5, 0.001, 0.30, 3000, 40)",
            "gold_call": "_oracle_bottleneck_end_from_class_coverage(0.03233, 0.02, 0.04, 8000, 150, 600, 35, 0.5, 0.02, 1.0, 1.5, 0.001, 0.30, 3000, 40)",
        },
        # --- boundary: an observed coverage above every model value, which selects the smallest candidate ---
        {
            "setup": "import numpy as np\n",
            "call": "bottleneck_end_from_class_coverage(0.5, 0.02, 0.04, 8000, 150, 600, 35, 0.5, 0.02, 1.0, 1.5, 0.001, 0.30, 3000, 30)",
            "gold_call": "_oracle_bottleneck_end_from_class_coverage(0.5, 0.02, 0.04, 8000, 150, 600, 35, 0.5, 0.02, 1.0, 1.5, 0.001, 0.30, 3000, 30)",
        },
        # --- edge: a class of short runs, a short candidate range and a sparse panel ---
        {
            "setup": "import numpy as np\n",
            "call": "bottleneck_end_from_class_coverage(0.03, 0.005, 0.01, 5000, 80, 400, 15, 1.0, 0.05, 2.0, 2.0, 0.004, 0.25, 2000, 60)",
            "gold_call": "_oracle_bottleneck_end_from_class_coverage(0.03, 0.005, 0.01, 5000, 80, 400, 15, 1.0, 0.05, 2.0, 2.0, 0.004, 0.25, 2000, 60)",
        },
    ]
