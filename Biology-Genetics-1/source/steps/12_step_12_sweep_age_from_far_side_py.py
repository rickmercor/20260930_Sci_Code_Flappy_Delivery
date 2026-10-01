"""
Number of generations since a favourable mutation arose whose model far-side class probability is closest to an observed value.

How long ago a favourable mutation arose is not observed directly, but it leaves a

trace in the runs of homozygosity at linked neutral positions: the older the sweep, the

more of the population carries the favourable allele and the more pairs of haplotypes

share recent ancestry at the neutral position. When the history of the population and

the strength of selection are known from other evidence, the age of the sweep can be

read off the frequency of a length class of the side of the run away from the selected

site. This step performs that inversion on the integer grid of candidate ages.

Returns
-------
int, the selected number of generations since the favourable mutation arose, as a native Python int
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sweep_age_from_far_side(observed: float, lower: float, upper: float, n_ancestral: float, n_bottleneck: float,
                            n_recent: float, bottleneck_length: int, since_end: int, mutation_rate: float,
                            selection: float, chromosome_length: float, advantage: float, recombination: float,
                            break_rate: float, marker_spacing: float, heterozygosity: float, horizon: int) -> int:
    '''Number of generations since a favourable mutation arose whose model far-side class probability is closest to an observed value.

    The favourable mutation arose after the bottleneck ended, so the candidate
    numbers of generations since it arose are the integers from 1 to
    since_end. For each candidate, the model probability that the side of the
    run of homozygosity away from the selected site has a length in
    [lower, upper] Morgans is the first entry of the far-side profile step for
    that candidate and the other inputs. Return the candidate whose model
    probability is closest to observed; if two candidates are equally close,
    return the smaller.

    Parameters
    ----------
    observed : float
        Observed probability of the class, in [0, 1].
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

    Returns
    -------
    sweep_age : int
        The selected number of generations since the favourable mutation
        arose, as a native Python int.

    Raises
    ------
    ValueError
        If observed is not a finite number in [0, 1], if since_end is not an
        integer >= 1, or on any condition raised by the previous steps for the
        other inputs.
    '''
    return sweep_age  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sweep_age_from_far_side(observed: float, lower: float, upper: float, n_ancestral: float, n_bottleneck: float,
                                    n_recent: float, bottleneck_length: int, since_end: int, mutation_rate: float,
                                    selection: float, chromosome_length: float, advantage: float, recombination: float,
                                    break_rate: float, marker_spacing: float, heterozygosity: float, horizon: int) -> int:
    if (isinstance(observed, bool) or not isinstance(observed, (int, float, np.integer, np.floating))
            or not np.isfinite(float(observed)) or not 0.0 <= float(observed) <= 1.0):
        raise ValueError("observed must be a finite number in [0, 1]")
    if isinstance(since_end, bool) or not isinstance(since_end, (int, np.integer)) or int(since_end) < 1:
        raise ValueError("since_end must be an integer >= 1")
    best, best_gap = None, None
    for candidate in range(1, int(since_end) + 1):
        probability = _oracle_far_side_class_profile(lower, upper, n_ancestral, n_bottleneck, n_recent,
                                                     bottleneck_length, since_end, mutation_rate, selection,
                                                     chromosome_length, advantage, recombination, candidate,
                                                     break_rate, marker_spacing, heterozygosity, horizon)[0]
        gap = abs(float(probability) - float(observed))
        if best is None or gap < best_gap:
            best, best_gap = candidate, gap
    return int(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the observed 1-2 cM far-side probability of the shipped configuration ---
        {
            "setup": "import numpy as np\n",
            "call": "sweep_age_from_far_side(0.0792658, 0.01, 0.02, 8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 1.5, 0.001, 0.30, 3000)",
            "gold_call": "_oracle_sweep_age_from_far_side(0.0792658, 0.01, 0.02, 8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 1.5, 0.001, 0.30, 3000)",
        },
        # --- boundary: an observed probability above every model value, which selects the oldest candidate ---
        {
            "setup": "import numpy as np\n",
            "call": "sweep_age_from_far_side(0.9, 0.01, 0.02, 8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 1.5, 0.001, 0.30, 3000)",
            "gold_call": "_oracle_sweep_age_from_far_side(0.9, 0.01, 0.02, 8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 1.5, 0.001, 0.30, 3000)",
        },
        # --- edge: another history and chromosome, a weaker sweep and a sparse panel, on the open class ---
        {
            "setup": "import numpy as np\n",
            "call": "sweep_age_from_far_side(0.085, 0.01, np.inf, 2000, 60, 900, 12, 40, 1.0, 0.05, 2.0, 0.2, 0.02, 2.0, 0.004, 0.25, 800)",
            "gold_call": "_oracle_sweep_age_from_far_side(0.085, 0.01, np.inf, 2000, 60, 900, 12, 40, 1.0, 0.05, 2.0, 0.2, 0.02, 2.0, 0.004, 0.25, 800)",
        },
    ]
