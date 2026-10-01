"""
Probability, mean coalescence time and sweep share of a length class of the side of the run of homozygosity away from the selected site.

A sweep distorts the runs of homozygosity around a linked neutral position

asymmetrically. On the side of the neutral position away from the selected site,

recombination trims the run but does not touch the association between the neutral

position and the favourable allele, so the length of that side depends on the

coalescence time exactly as it does without selection, while the coalescence time

itself carries the sweep's signature. That side can therefore be described with the

ordinary one-sided length distribution and the sweep-dependent coalescence weights.

This step summarises one length class of that side.

Returns
-------
np.ndarray, shape (3,): profile[0] is the probability that the side away from the selected site has a length in the class, profile[1] the mean coalescence generation of the pairs counted in it, and profile[2] the fraction of the probability from coalescence in generations 1 to sweep_age
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def far_side_class_profile(lower: float, upper: float, n_ancestral: float, n_bottleneck: float, n_recent: float,
                           bottleneck_length: int, since_end: int, mutation_rate: float, selection: float,
                           chromosome_length: float, advantage: float, recombination: float, sweep_age: int,
                           break_rate: float, marker_spacing: float, heterozygosity: float,
                           horizon: int) -> "np.ndarray":
    '''Probability, mean coalescence time and sweep share of a length class of the side of the run of homozygosity away from the selected site.

    At a neutral position linked to a sweeping favourable mutation, consider
    only the side of the run of homozygosity away from the selected site. For
    a pair of haplotypes that coalesced g generations ago, the probability
    that this side extends at least a distance x is S_g(x), the one-side
    survival of the flank step for t = g with the given break rate, marker
    spacing and heterozygosity, independently of the sweep; the pair's first
    coalescence generation g has the probabilities of the focal-weights step
    for the given history, background selection and sweep. The probability
    that this side of the run of a random individual has a length in
    [lower, upper] Morgans is the sum over g = 1 to horizon of the
    first-coalescence probability at g times S_g(lower) - S_g(upper), with
    S_g(upper) = 0 for an infinite upper. Return that probability; the mean
    coalescence generation of the pairs it counts, each generation weighted by
    its contribution to the probability; and the fraction of the probability
    due to coalescence in generations 1 to sweep_age, during the sweep.

    Parameters
    ----------
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
    sweep_age : int
        Number of generations since the favourable mutation arose, an integer
        in [1, since_end].
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
    profile : np.ndarray
        Shape (3,): profile[0] is the probability that the side away from the
        selected site has a length in the class, profile[1] the mean
        coalescence generation of the pairs counted in it, and profile[2] the
        fraction of the probability from coalescence in generations 1 to
        sweep_age.

    Raises
    ------
    ValueError
        If lower and upper do not satisfy 0 <= lower < upper with lower
        finite, if the probability of the class within the horizon is zero, or
        on any condition raised by the previous steps for the other inputs.
    '''
    return profile  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_far_side_class_profile(lower: float, upper: float, n_ancestral: float, n_bottleneck: float, n_recent: float,
                                   bottleneck_length: int, since_end: int, mutation_rate: float, selection: float,
                                   chromosome_length: float, advantage: float, recombination: float, sweep_age: int,
                                   break_rate: float, marker_spacing: float, heterozygosity: float,
                                   horizon: int) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(float(v))

    if not (_num(lower) and _num(upper) and np.isfinite(float(lower)) and 0.0 <= float(lower) < float(upper)):
        raise ValueError("lower and upper must satisfy 0 <= lower < upper with lower finite")
    weights = _oracle_focal_coalescence_weights(n_ancestral, n_bottleneck, n_recent, bottleneck_length, since_end,
                                                mutation_rate, selection, chromosome_length, advantage, recombination,
                                                sweep_age, horizon)
    lo, hi = float(lower), float(upper)
    open_class = not np.isfinite(hi)
    ends = np.array([lo]) if open_class else np.array([lo, hi])
    generations = np.arange(1, int(horizon) + 1)
    # the one-side survival after g generations is the survival after one generation to the power g, since the
    # observable breaks of different generations are independent
    one_generation = _oracle_flank_survival_and_density(ends, 1, break_rate, marker_spacing, heterozygosity)[0]
    survival = one_generation[None, :] ** generations[:, None]
    mass = survival[:, 0] - (0.0 if open_class else survival[:, 1])
    contributions = weights * mass
    probability = float(contributions.sum())
    if probability <= 0.0:
        raise ValueError("the probability of the class within the horizon is zero")
    mean_generation = float((generations * contributions).sum() / probability)
    during = float(contributions[generations <= int(sweep_age)].sum() / probability)
    return np.array([probability, mean_generation, during])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the 0.5-1 cM class of the far side in the shipped configuration ---
        {
            "setup": "import numpy as np\n",
            "call": "far_side_class_profile(0.005, 0.01, 8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 21, 1.5, 0.001, 0.30, 3000)",
            "gold_call": "_oracle_far_side_class_profile(0.005, 0.01, 8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 21, 1.5, 0.001, 0.30, 3000)",
        },
        # --- boundary: every break observed where it occurs, the 1-2 cM class ---
        {
            "setup": "import numpy as np\n",
            "call": "far_side_class_profile(0.01, 0.02, 8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 21, 1.5, 0.0, 0.30, 3000)",
            "gold_call": "_oracle_far_side_class_profile(0.01, 0.02, 8000, 150, 600, 35, 26, 0.5, 0.02, 1.0, 0.3, 0.01, 21, 1.5, 0.0, 0.30, 3000)",
        },
        # --- edge: the open class of far sides of 1 cM and longer, a young sweep and a sparse panel ---
        {
            "setup": "import numpy as np\n",
            "call": "far_side_class_profile(0.01, np.inf, 2000, 60, 900, 12, 40, 1.0, 0.05, 2.0, 0.2, 0.02, 9, 2.0, 0.004, 0.25, 800)",
            "gold_call": "_oracle_far_side_class_profile(0.01, np.inf, 2000, 60, 900, 12, 40, 1.0, 0.05, 2.0, 0.2, 0.02, 9, 2.0, 0.004, 0.25, 800)",
        },
    ]
