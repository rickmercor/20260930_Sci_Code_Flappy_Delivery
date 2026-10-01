"""
'''Additive genetic variance for relative fitness of the base population.




    Each replicate was founded from the offspring of one round of random mating in

    the base population, a round that passed through a bottleneck with inbreeding

    effective size founding_effective_size. Its allele frequencies were measured

    in that founding generation and again n_generations generations later, and

    delta_p_by_replicate holds the differences. Assume the mean average effects

    for relative fitness are proportional to the reference-allele frequency

    contrast p - q of the base population, using its realised frequencies, with no

    across-replicate variance in the average effects, and correct the resulting

    quadratic form for the sampling error in the fitted proportionality

    coefficient. Treat each replicate's census size as its inbreeding effective

    size in every round of mating after the founding round, and derive its

    variance effective size from that census size and offspring_variance.




    Parameters

    ----------

    delta_p_by_replicate : np.ndarray

        (n_replicates, n_loci) observed allele-frequency changes, one row per

        replicate, loci in the same physical order throughout.

    census_sizes : list

        Length-n_replicates list of int census sizes, one per replicate, in the

        same order as the rows of delta_p_by_replicate.

    founding_effective_size : float

        Inbreeding effective size of the founding round, shared by all

        replicates. Must be > 0.5.

    interval_rates : np.ndarray

        (n_loci - 1,) crossover probabilities between adjacent loci.

    n_generations : int

        Number of generations between the two measurements, >= 1.

    offspring_variance : float

        Variance in offspring number, shared across replicates, >= 0.

    population_spec : dict

        Exactly the keys 'n_loci', 'n_individuals', 'rho', 'p_low', 'p_high' and

        'seed', giving the reference construction of the base population.




    Returns

    -------

    v_a : float

        Estimated additive genetic variance for relative fitness, as a native

        Python float.




    Raises

    ------

    ValueError

        If delta_p_by_replicate is not a finite 2D array with at least one row, if

        census_sizes is not a list whose length matches the number of rows, if

        founding_effective_size is not a finite number > 0.5, if

        population_spec is not a dict holding exactly the six required keys, if

        population_spec['n_loci'] does not equal the number of columns of

        delta_p_by_replicate, or if interval_rates does not have length

        n_loci - 1. Conditions raised by the earlier steps propagate unchanged.

    '''

This step returns the deliverable: the additive genetic variance for relative

fitness of the base population, which by Fisher's fundamental theorem is the

per-generation increase in mean fitness attributable to change in allele

frequency. It is estimated from the replicate allele-frequency changes under the

model for the mean average effects stated in the signature, with the preceding

steps as building blocks.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def additive_genetic_variance_for_fitness(delta_p_by_replicate: np.ndarray,
                                          census_sizes: list,
                                          founding_effective_size: float,
                                          interval_rates: np.ndarray,
                                          n_generations: int,
                                          offspring_variance: float,
                                          population_spec: dict) -> float:
    '''Additive genetic variance for relative fitness of the base population.

    Each replicate was founded from the offspring of one round of random mating in
    the base population, a round that passed through a bottleneck with inbreeding
    effective size founding_effective_size. Its allele frequencies were measured
    in that founding generation and again n_generations generations later, and
    delta_p_by_replicate holds the differences. Assume the mean average effects
    for relative fitness are proportional to the reference-allele frequency
    contrast p - q of the base population, using its realised frequencies, with no
    across-replicate variance in the average effects, and correct the resulting
    quadratic form for the sampling error in the fitted proportionality
    coefficient. Treat each replicate's census size as its inbreeding effective
    size in every round of mating after the founding round, and derive its
    variance effective size from that census size and offspring_variance.

    Parameters
    ----------
    delta_p_by_replicate : np.ndarray
        (n_replicates, n_loci) observed allele-frequency changes, one row per
        replicate, loci in the same physical order throughout.
    census_sizes : list
        Length-n_replicates list of int census sizes, one per replicate, in the
        same order as the rows of delta_p_by_replicate.
    founding_effective_size : float
        Inbreeding effective size of the founding round, shared by all
        replicates. Must be > 0.5.
    interval_rates : np.ndarray
        (n_loci - 1,) crossover probabilities between adjacent loci.
    n_generations : int
        Number of generations between the two measurements, >= 1.
    offspring_variance : float
        Variance in offspring number, shared across replicates, >= 0.
    population_spec : dict
        Exactly the keys 'n_loci', 'n_individuals', 'rho', 'p_low', 'p_high' and
        'seed', giving the reference construction of the base population.

    Returns
    -------
    v_a : float
        Estimated additive genetic variance for relative fitness, as a native
        Python float.

    Raises
    ------
    ValueError
        If delta_p_by_replicate is not a finite 2D array with at least one row, if
        census_sizes is not a list whose length matches the number of rows, if
        founding_effective_size is not a finite number > 0.5, if
        population_spec is not a dict holding exactly the six required keys, if
        population_spec['n_loci'] does not equal the number of columns of
        delta_p_by_replicate, or if interval_rates does not have length
        n_loci - 1. Conditions raised by the earlier steps propagate unchanged.
    '''
    return v_a  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_additive_genetic_variance_for_fitness(delta_p_by_replicate: np.ndarray,
                                                  census_sizes: list,
                                                  founding_effective_size: float,
                                                  interval_rates: np.ndarray,
                                                  n_generations: int,
                                                  offspring_variance: float,
                                                  population_spec: dict) -> float:
    _SPEC_KEYS = {"n_loci", "n_individuals", "rho", "p_low", "p_high", "seed"}
    Y = np.asarray(delta_p_by_replicate, dtype=float)
    if Y.ndim != 2 or Y.shape[0] < 1 or Y.shape[1] < 1:
        raise ValueError("delta_p_by_replicate must be a 2D array with >= 1 row")
    if not np.all(np.isfinite(Y)):
        raise ValueError("delta_p_by_replicate must be finite")
    if not isinstance(census_sizes, list) or len(census_sizes) != Y.shape[0]:
        raise ValueError("census_sizes must be a list with one entry per replicate")
    if isinstance(founding_effective_size, bool) or not isinstance(
            founding_effective_size, (int, float, np.floating, np.integer)):
        raise ValueError("founding_effective_size must be a finite number > 0.5")
    if not np.isfinite(float(founding_effective_size)) or float(founding_effective_size) <= 0.5:
        raise ValueError("founding_effective_size must be a finite number > 0.5")
    if not isinstance(population_spec, dict) or set(population_spec) != _SPEC_KEYS:
        raise ValueError(f"population_spec must be a dict with exactly the keys {sorted(_SPEC_KEYS)}")
    if int(population_spec["n_loci"]) != Y.shape[1]:
        raise ValueError("population_spec['n_loci'] must match the number of loci")
    rates = np.asarray(interval_rates, dtype=float)
    if rates.ndim != 1 or rates.size != Y.shape[1] - 1:
        raise ValueError("interval_rates must have length n_loci - 1")

    haplotypes = _oracle_simulate_base_population(
        population_spec["n_loci"], population_spec["n_individuals"],
        population_spec["rho"], population_spec["p_low"],
        population_spec["p_high"], population_spec["seed"])

    base_diversity = _oracle_individual_diversity_matrix(haplotypes)
    recombination = _oracle_pairwise_recombination_matrix(rates)
    weighted_base = _oracle_weighted_base_diversity_matrix(haplotypes, recombination)

    # Diagnostic-only cross-check: the additive GENIC variance (step 08) ignores
    # linkage disequilibrium and is not the target quantity, but a broken step 08
    # must not be able to pass this orchestrator silently just because its output
    # is never touched.
    replicate_mean_change = Y.mean(axis=0)
    genic_variance = _oracle_genic_variance_from_change(replicate_mean_change, base_diversity)
    if not np.isfinite(genic_variance) or genic_variance < 0.0:
        raise ValueError("genic-variance diagnostic (step 08) returned a non-finite or negative value")

    operators, covariances = [], []
    for census in census_sizes:
        n_eff = _oracle_variance_effective_size(census, offspring_variance)
        operators.append(_oracle_expected_change_operator(
            weighted_base, recombination, n_generations,
            float(founding_effective_size), float(census)))
        covariances.append(_oracle_drift_covariance_matrix(
            weighted_base, recombination, n_generations,
            float(founding_effective_size), float(census), n_eff))

    frequency = np.asarray(haplotypes, dtype=float).mean(axis=0)
    contrast = 2.0 * frequency - 1.0

    coefficient = _oracle_average_effect_scale(Y, operators, covariances, contrast)

    information = 0.0
    for Lm, Dm in zip(operators, covariances):
        z = Lm @ contrast
        information += float(z @ np.linalg.solve(Dm, z))
    sampling_variance = 1.0 / information

    scale = float(contrast @ base_diversity @ contrast)
    return float(coefficient ** 2 * scale - scale * sampling_variance)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _SHIPPED = """import numpy as np
Y = np.array([
 [-0.080508, -0.088382, -0.092691, -0.079427, -0.029581,  0.007478,  0.057723,  0.081924,  0.063215,  0.058229],
 [-0.087222, -0.098856, -0.084800, -0.045451, -0.039817, -0.008562,  0.026788,  0.041907,  0.049555,  0.057158],
 [-0.104796, -0.131458, -0.065336, -0.087962, -0.065417, -0.008970,  0.015046,  0.058879,  0.065851,  0.072945],
 [-0.076533, -0.060456, -0.062452, -0.045712, -0.024884,  0.022225,  0.068221,  0.082374,  0.104465,  0.091738],
 [-0.063602, -0.083285, -0.104352, -0.102128, -0.025376,  0.053171,  0.009892,  0.118097,  0.088889,  0.056976],
 [-0.079714, -0.159038, -0.173297, -0.097500, -0.076308,  0.005407,  0.028781,  0.104331,  0.044838,  0.049813]])
census = [2000, 1200, 700, 400, 250, 150]
rates = np.full(9, 0.05)
spec = {"n_loci": 10, "n_individuals": 2000, "rho": 0.90,
        "p_low": 0.15, "p_high": 0.85, "seed": 2043}
"""
    return [
        # --- normal: the shipped configuration; this is the reported answer ---
        {
            "setup": _SHIPPED,
            "call": "additive_genetic_variance_for_fitness(Y, census, 50.0, rates, 2, 6.0, spec)",
            "gold_call": "_oracle_additive_genetic_variance_for_fitness(Y, census, 50.0, rates, 2, 6.0, spec)",
        },
        # --- boundary: one generation and Wright-Fisher offspring variance, so
        #     N_E == N and the operator is the base matrix after the founding
        #     round alone ---
        {
            "setup": _SHIPPED,
            "call": "additive_genetic_variance_for_fitness(Y, census, 50.0, rates, 1, 2.0, spec)",
            "gold_call": "_oracle_additive_genetic_variance_for_fitness(Y, census, 50.0, rates, 1, 2.0, spec)",
        },
        # --- edge: a single small replicate, where the sampling-variance
        #     correction is a large fraction of the uncorrected quadratic form ---
        {
            "setup": _SHIPPED,
            "call": "additive_genetic_variance_for_fitness(Y[5:], [150], 50.0, rates, 2, 6.0, spec)",
            "gold_call": "_oracle_additive_genetic_variance_for_fitness(Y[5:], [150], 50.0, rates, 2, 6.0, spec)",
        },
    ]
