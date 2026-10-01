"""
Mean coalescence time of the pairs whose run extends a reported length on the side of a swept neutral position away from the selected site, at the recovered bottleneck end and sweep age.

This step returns the deliverable: the mean coalescence time, in generations before

the present, of the pairs of haplotypes whose run of homozygosity at a neutral position

near a sweeping favourable mutation extends a stated length on the side away from the

selected site, once the end of a past bottleneck has been dated from the genome-wide

coverage of a class of runs, itself recovered from the constant population size a study

reported for that class, and the age of the sweep has been dated from the frequency of

another class of that side. It converts the genotyping panel and the break rates from

their laboratory units to the map units of the earlier steps, recovers the coverage

behind the reported size, recovers the number of generations since the bottleneck ended,

recovers the number of generations since the favourable mutation arose, and evaluates

the profile of the reported class of the side away from the selected site.

Returns
-------
float, mean coalescence generation of the pairs counted by the reported class of the side away from the selected site at the recovered bottleneck end and sweep age, in generations before the present, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def swept_focal_mean_coalescence_time(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                                      mutation_rate_per_chromosome: float, selection_coefficient: float,
                                      chromosome_length_morgans: float,
                                      marker_spacing_kb: float, map_cM_per_Mb: float, heterozygosity: float,
                                      mutation_rate_per_bp: float, conversion_rate_per_bp: float,
                                      observed_class_cM: "np.ndarray", reported_size: float,
                                      advantage: float, recombination_fraction: float,
                                      far_observed_class_cM: "np.ndarray", far_observed_probability: float,
                                      far_report_class_cM: "np.ndarray", max_since_end: int, horizon: int) -> float:
    '''Mean coalescence time of the pairs whose run extends a reported length on the side of a swept neutral position away from the selected site, at the recovered bottleneck end and sweep age.

    A randomly mating diploid population had n_ancestral breeding individuals
    until a bottleneck of n_bottleneck individuals that lasted bottleneck_length
    generations, after which it has had n_recent individuals up to the present;
    the number of generations since the bottleneck ended is unknown. Deleterious
    mutations arise along the chromosome, of map length
    chromosome_length_morgans, at mutation_rate_per_chromosome per chromosome
    per generation with a multiplicative fitness effect selection_coefficient in
    heterozygotes, and the background selection they cause reduces the
    effective size of a neutral position as in the reduction-factor step.
    Individuals are genotyped at markers spaced marker_spacing_kb kilobases
    apart on average, on a map of map_cM_per_Mb centiMorgans per megabase, with
    heterozygosity at the typed markers; identity along a lineage is broken by
    recombination and by de novo mutation and gene conversion at the stated
    rates per base pair per generation. For the runs of lengths in
    observed_class_cM the study reports, in place of their coverage, the
    constant population size that the source's closed-form steady-state
    expression for the coverage of runs of a given length on this panel returns
    for that class: reported_size breeding individuals. At one focal position, a
    favourable mutation with heterozygous advantage advantage arose, after the
    bottleneck ended, at a site at recombination fraction
    recombination_fraction from the focal position; the number of generations
    since it arose is unknown. The fraction of individuals whose run of
    homozygosity around the focal position has a length in
    far_observed_class_cM on the side away from the selected site is
    far_observed_probability. Recover the coverage of the observed class from
    reported_size as in the closed-form step; recover the number of generations
    since the bottleneck ended from that coverage as in the bottleneck-end step,
    searching 1 to max_since_end; recover the number of generations since the
    favourable mutation arose from far_observed_probability as in the
    sweep-age step; then return the mean coalescence generation of the pairs
    counted by the class far_report_class_cM of the side away from the selected
    site at those two values, as in the far-side profile step, counting
    coalescence generations 1 to horizon.

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
    mutation_rate_per_chromosome : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection_coefficient : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length_morgans : float
        Map length of the chromosome in Morgans, > 0.
    marker_spacing_kb : float
        Mean spacing of the typed markers in kilobases, >= 0.
    map_cM_per_Mb : float
        Genetic map density in centiMorgans per megabase, > 0.
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].
    mutation_rate_per_bp : float
        De novo mutation rate per base pair per generation, >= 0.
    conversion_rate_per_bp : float
        Rate of gene-conversion breaks per base pair per generation, >= 0.
    observed_class_cM : np.ndarray
        Shape (2,): lower and upper ends of the observed length class in
        centiMorgans, 0 <= lower < upper; the upper end may be infinite.
    reported_size : float
        Constant number of breeding individuals reported for that class, >= 1.
    advantage : float
        Fitness advantage of a heterozygous carrier of the favourable
        mutation, > 0.
    recombination_fraction : float
        Recombination fraction between the selected site and the focal
        position, in [0, 0.5].
    far_observed_class_cM : np.ndarray
        Shape (2,): lower and upper ends, in centiMorgans, of the length class
        of the side of the run away from the selected site whose frequency is
        observed, 0 <= lower < upper; the upper end may be infinite.
    far_observed_probability : float
        Observed fraction of individuals whose run has a length in that class
        on that side, in [0, 1].
    far_report_class_cM : np.ndarray
        Shape (2,): lower and upper ends, in centiMorgans, of the length class
        of the side away from the selected site that is reported,
        0 <= lower < upper; the upper end may be infinite.
    max_since_end : int
        Largest candidate number of generations since the bottleneck ended,
        >= 1.
    horizon : int
        Last coalescence generation before the present that is counted, >= 1.

    Returns
    -------
    mean_generation : float
        Mean coalescence generation of the pairs counted by the reported class
        of the side away from the selected site at the recovered bottleneck end
        and sweep age, in generations before the present, as a native Python
        float.

    Raises
    ------
    ValueError
        If marker_spacing_kb is not a finite number >= 0, if map_cM_per_Mb is
        not a finite number > 0, if either rate per base pair is not a finite
        number >= 0, if any class is not a pair of numbers with
        0 <= lower < upper, or on any condition raised by the earlier steps for
        the derived inputs.
    '''
    return mean_generation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_swept_focal_mean_coalescence_time(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                                              mutation_rate_per_chromosome: float, selection_coefficient: float,
                                              chromosome_length_morgans: float,
                                              marker_spacing_kb: float, map_cM_per_Mb: float, heterozygosity: float,
                                              mutation_rate_per_bp: float, conversion_rate_per_bp: float,
                                              observed_class_cM: "np.ndarray", reported_size: float,
                                              advantage: float, recombination_fraction: float,
                                              far_observed_class_cM: "np.ndarray", far_observed_probability: float,
                                              far_report_class_cM: "np.ndarray", max_since_end: int, horizon: int) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    def _class(c):
        cc = np.asarray(c, dtype=float) if not isinstance(c, (str, bytes)) else None
        if cc is None or cc.shape != (2,) or np.isnan(cc).any() or not np.isfinite(cc[0]) or not 0.0 <= cc[0] < cc[1]:
            raise ValueError("a length class must be a pair of numbers with 0 <= lower < upper")
        return float(cc[0]) / 100.0, float(cc[1]) / 100.0

    if not _num(marker_spacing_kb) or float(marker_spacing_kb) < 0.0:
        raise ValueError("marker_spacing_kb must be a finite number >= 0")
    if not _num(map_cM_per_Mb) or float(map_cM_per_Mb) <= 0.0:
        raise ValueError("map_cM_per_Mb must be a finite number > 0")
    if not _num(mutation_rate_per_bp) or float(mutation_rate_per_bp) < 0.0 or not _num(conversion_rate_per_bp) \
            or float(conversion_rate_per_bp) < 0.0:
        raise ValueError("mutation_rate_per_bp and conversion_rate_per_bp must be finite numbers >= 0")
    lo_obs, hi_obs = _class(observed_class_cM)
    lo_far, hi_far = _class(far_observed_class_cM)
    lo_rep, hi_rep = _class(far_report_class_cM)
    # laboratory units to map units: one Morgan is 100 cM, and map_cM_per_Mb cM span one megabase
    bp_per_morgan = 1.0e8 / float(map_cM_per_Mb)
    marker_spacing = float(marker_spacing_kb) * 1.0e3 / bp_per_morgan
    break_rate = (float(mutation_rate_per_bp) + float(conversion_rate_per_bp)) * bp_per_morgan
    observed_coverage = _oracle_class_coverage_from_reported_size(reported_size, lo_obs, hi_obs, break_rate, marker_spacing,
                                                                  heterozygosity)
    since_end = _oracle_bottleneck_end_from_class_coverage(observed_coverage, lo_obs, hi_obs, n_ancestral, n_bottleneck,
                                                           n_recent, bottleneck_length, mutation_rate_per_chromosome,
                                                           selection_coefficient, chromosome_length_morgans, break_rate,
                                                           marker_spacing, heterozygosity, horizon, max_since_end)
    sweep_age = _oracle_sweep_age_from_far_side(far_observed_probability, lo_far, hi_far, n_ancestral, n_bottleneck,
                                                n_recent, bottleneck_length, since_end, mutation_rate_per_chromosome,
                                                selection_coefficient, chromosome_length_morgans, advantage,
                                                recombination_fraction, break_rate, marker_spacing, heterozygosity,
                                                horizon)
    profile = _oracle_far_side_class_profile(lo_rep, hi_rep, n_ancestral, n_bottleneck, n_recent, bottleneck_length,
                                             since_end, mutation_rate_per_chromosome, selection_coefficient,
                                             chromosome_length_morgans, advantage, recombination_fraction, sweep_age,
                                             break_rate, marker_spacing, heterozygosity, horizon)
    return float(profile[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped configuration ---
        {
            "setup": "import numpy as np\nobs = np.array([2.0, 4.0])\nfar_obs = np.array([1.0, 2.0])\nfar_rep = np.array([0.5, 1.0])\n",
            "call": "swept_focal_mean_coalescence_time(8000, 150, 600, 35, 0.5, 0.02, 1.0, 100.0, 1.0, 0.30, 1.2e-08, 3e-09, obs, 199.753, 0.3, 0.01, far_obs, 0.0792658, far_rep, 200, 3000)",
            "gold_call": "_oracle_swept_focal_mean_coalescence_time(8000, 150, 600, 35, 0.5, 0.02, 1.0, 100.0, 1.0, 0.30, 1.2e-08, 3e-09, obs, 199.753, 0.3, 0.01, far_obs, 0.0792658, far_rep, 200, 3000)",
        },
        # --- boundary: the observed far-side class reported back, at complete linkage ---
        {
            "setup": "import numpy as np\nobs = np.array([2.0, 4.0])\nfar_obs = np.array([1.0, 2.0])\n",
            "call": "swept_focal_mean_coalescence_time(8000, 150, 600, 35, 0.5, 0.02, 1.0, 100.0, 1.0, 0.30, 1.2e-08, 3e-09, obs, 199.753, 0.3, 0.0, far_obs, 0.06, far_obs, 40, 3000)",
            "gold_call": "_oracle_swept_focal_mean_coalescence_time(8000, 150, 600, 35, 0.5, 0.02, 1.0, 100.0, 1.0, 0.30, 1.2e-08, 3e-09, obs, 199.753, 0.3, 0.0, far_obs, 0.06, far_obs, 40, 3000)",
        },
        # --- edge: another history, chromosome, panel and sweep ---
        {
            "setup": "import numpy as np\nobs = np.array([2.0, 4.0])\nfar_obs = np.array([1.0, np.inf])\nfar_rep = np.array([0.5, 1.0])\n",
            "call": "swept_focal_mean_coalescence_time(5000, 80, 400, 15, 1.0, 0.05, 2.0, 150.0, 1.6, 0.25, 1.0e-08, 5e-09, obs, 300.0, 0.12, 0.02, far_obs, 0.16, far_rep, 80, 1500)",
            "gold_call": "_oracle_swept_focal_mean_coalescence_time(5000, 80, 400, 15, 1.0, 0.05, 2.0, 150.0, 1.6, 0.25, 1.0e-08, 5e-09, obs, 300.0, 0.12, 0.02, far_obs, 0.16, far_rep, 80, 1500)",
        },
    ]
