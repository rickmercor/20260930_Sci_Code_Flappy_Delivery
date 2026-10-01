"""
Chain the sub-problem functions 01-07 end-to-end on the confounded donor cohort and return the gap in base ten logarithms between the associational and the causal communication odds. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (simulate_communication_dataset, compute_naive_communication_score, initialise_gibbs_state, update_exposure_equations, update_outcome_equation, update_communication_state, summarise_communication_evidence) rather than reimplementing them.

This step runs the whole comparison end to end. It (i) draws the donor cohort from the structural model with sub-problem 01, (ii) scores it the way a co-expression screen would, by the joint test of the two ligand terms in a regression on the observed exposures, with sub-problem 02, (iii) starts the sampler from the least squares fits of sub-problem 03, and then (iv) sweeps the sampler, drawing the two exposure equations with sub-problem 04, the four unselected outcome parameters with sub-problem 05 and the communication effect vector together with the selection parameters with sub-problem 06, and finally (v) reduces the retained sweeps to a posterior inclusion probability and two posterior mean effects and hands them to sub-problem 07.


Each sweep visits the blocks in the order the model is written. The exposure equations come first, because everything downstream is expressed in terms of the representations they define. The unselected outcome parameters come next, and the communication effect vector and its selection parameters come last, so that the inclusion indicator is drawn against the effect vector of the same sweep rather than of the previous one. Each of the three sampler steps reads the whole parameter state and writes back only the block it owns, which is what keeps the sweep a Gibbs sweep: no step may condition on a value that a later step in the same sweep has already replaced.

The returned scalar is the gap in orders of magnitude between what an associational screen concludes about this cohort and what a causal analysis concludes about the same cohort. There is no communication in the cohort at all, so an analysis that has defeated the confounding should place its verdict near the low end of the unit interval and an analysis that has not should place it at the top; the logarithm of the ratio of the two odds says by how much the associational verdict overstates the case. A value near zero would mean that the associational screen has nothing to answer for at these confounder loadings. A large positive value means the screen is manufacturing evidence, that the manufacture is invisible from inside the screen because nothing about the fit looks pathological, and that instrumenting both expression traits rather than raising the significance threshold is what removes it. The sign of the gap also matters: a negative value would say the causal analysis is the more confident of the two, which under a cohort built with no communication would indicate a failure of the selection prior rather than a discovery.

Returns
-------
float: the base ten logarithm of the associational communication odds divided by the causal communication odds, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def run_causal_communication_pipeline(n_donors: int = 600, n_instruments: int = 4,
                                      n_covariates: int = 3,
                                      instrument_strength: float = 0.45,
                                      covariate_effect: float = 0.25,
                                      receptor_effect: float = 0.5,
                                      confounder_loading: float = 0.7,
                                      ligand_effect: float = 0.0,
                                      interaction_effect: float = 0.0,
                                      data_seed: int = 20260826,
                                      n_iterations: int = 20000, burn_in: int = 2000,
                                      thin: int = 5, nu_spike: float = 1e-4,
                                      a_sigma: float = 3.0, b_sigma: float = 2.0,
                                      a_rho: float = 3.0, b_rho: float = 1.0,
                                      ridge: float = 1e-6,
                                      sampler_seed: int = 2026) -> float:
    '''Run the whole associational versus causal comparison on one donor cohort.

    The Zellner g value shared by every prior is the smaller of n_donors and
    one hundred. The sampler is driven by numpy.random.default_rng of
    sampler_seed and takes n_iterations sweeps, each visiting the two exposure
    equations, then the four unselected outcome parameters, then the
    communication effect vector and the selection parameters, in that order,
    with every drawn block written back into the parameter state before the
    next step of the same sweep reads it. Sweeps numbered burn_in and above
    are retained whenever their index minus burn_in is a multiple of thin. The
    posterior inclusion probability is the mean inclusion indicator over the
    retained sweeps, and the two posterior effects are the mean ligand main
    effect and the mean interaction effect over the same sweeps. The three
    standard deviations used to standardise those effects are the population
    standard deviations of ligand expression, receptor expression and pathway
    activity in the cohort.

    Parameters
    ----------
    n_donors : int
        Number of donors, at least 2 * n_instruments + n_covariates + 5.
    n_instruments : int
        Number of cis-eQTL instruments for each exposure, n_instruments >= 1.
    n_covariates : int
        Number of observed donor covariates, n_covariates >= 1.
    instrument_strength : float
        Common coefficient of every instrument on its own exposure.
    covariate_effect : float
        Common coefficient of every covariate on each observed quantity.
    receptor_effect : float
        Coefficient of receptor expression on pathway activity.
    confounder_loading : float
        Common coefficient of the unmeasured donor factor on each observed
        quantity.
    ligand_effect : float
        Coefficient of ligand expression on pathway activity.
    interaction_effect : float
        Coefficient of the ligand by receptor product on pathway activity.
    data_seed : int
        Seed of the generator that draws the cohort.
    n_iterations : int
        Number of sampler sweeps, n_iterations > burn_in.
    burn_in : int
        Number of leading sweeps discarded, burn_in >= 0.
    thin : int
        Thinning factor applied after the burn in, thin >= 1.
    nu_spike : float
        Scale factor of the spike component, 0 < nu_spike <= 1.
    a_sigma : float
        Shape of the inverse gamma priors on the residual variances.
    b_sigma : float
        Scale of the inverse gamma priors on the residual variances.
    a_rho : float
        First shape of the beta prior on the inclusion probability.
    b_rho : float
        Second shape of the beta prior on the inclusion probability.
    ridge : float
        Non negative diagonal regularisation used in every matrix inverse.
    sampler_seed : int
        Seed of the generator that drives the sampler.

    Returns
    -------
    evidence_gap : float
        Base ten logarithm of the associational communication odds divided by
        the causal communication odds, as a native Python float.

    Raises
    ------
    ValueError
        If any argument fails the validation of the step it is passed to, if
        n_iterations does not exceed burn_in, if thin is below one, or if the
        retained sweeps leave the posterior inclusion probability at exactly
        zero or exactly one, where the odds are undefined.
    '''
    return evidence_gap  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_causal_communication_pipeline(n_donors: int = 600, n_instruments: int = 4,
                                              n_covariates: int = 3,
                                              instrument_strength: float = 0.45,
                                              covariate_effect: float = 0.25,
                                              receptor_effect: float = 0.5,
                                              confounder_loading: float = 0.7,
                                              ligand_effect: float = 0.0,
                                              interaction_effect: float = 0.0,
                                              data_seed: int = 20260826,
                                              n_iterations: int = 20000, burn_in: int = 2000,
                                              thin: int = 5, nu_spike: float = 1e-4,
                                              a_sigma: float = 3.0, b_sigma: float = 2.0,
                                              a_rho: float = 3.0, b_rho: float = 1.0,
                                              ridge: float = 1e-6,
                                              sampler_seed: int = 2026) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n_iterations, bool) or not isinstance(n_iterations, (int, np.integer)):
        raise ValueError("n_iterations must be an integer")
    if isinstance(burn_in, bool) or not isinstance(burn_in, (int, np.integer)):
        raise ValueError("burn_in must be an integer")
    if isinstance(thin, bool) or not isinstance(thin, (int, np.integer)):
        raise ValueError("thin must be an integer")
    if int(burn_in) < 0:
        raise ValueError("burn_in must not be negative")
    if int(thin) < 1:
        raise ValueError("thin must be at least one")
    if int(n_iterations) <= int(burn_in):
        raise ValueError("n_iterations must exceed burn_in")
    sweeps = int(n_iterations)
    discarded = int(burn_in)
    spacing = int(thin)

    # -- Sub-problem 01: the cohort every later step reads.
    data = _oracle_simulate_communication_dataset(n_donors, n_instruments, n_covariates,
                                                  instrument_strength, covariate_effect,
                                                  receptor_effect, confounder_loading,
                                                  ligand_effect, interaction_effect, data_seed)

    # -- Sub-problem 02: the associational verdict on the same cohort.
    naive_summary = _oracle_compute_naive_communication_score(data, n_instruments, n_covariates)

    n_instrument = int(n_instruments)
    n_covariate = int(n_covariates)
    offset = 2 * n_instrument + n_covariate
    ligand = data[:, offset]
    receptor = data[:, offset + 1]
    pathway = data[:, offset + 2]

    # -- Index of every block inside the flat parameter state.
    exposure_block = slice(0, 2 * (n_instrument + n_covariate + 1))
    intercept_index = exposure_block.stop
    outcome_covariate_slice = slice(intercept_index + 1, intercept_index + 1 + n_covariate)
    receptor_effect_index = outcome_covariate_slice.stop
    outcome_variance_index = receptor_effect_index + 1
    ligand_effect_index = outcome_variance_index + 1
    interaction_index = ligand_effect_index + 1
    inclusion_index = interaction_index + 1
    inclusion_probability_index = inclusion_index + 1

    # -- Sub-problem 03: the starting point of the sampler.
    state = _oracle_initialise_gibbs_state(data, n_instruments, n_covariates, a_rho, b_rho)
    g_prior = float(min(int(n_donors), 100))
    rng = np.random.default_rng(int(sampler_seed))
    inclusion_trace = []
    ligand_trace = []
    interaction_trace = []

    for sweep in range(sweeps):
        # -- Sub-problem 04: both exposure equations.
        exposures = _oracle_update_exposure_equations(data, n_instruments, n_covariates, state,
                                                      g_prior, a_sigma, b_sigma, ridge, rng)
        state[exposure_block] = exposures

        # -- Sub-problem 05: the four outcome parameters that carry no selection.
        outcome = _oracle_update_outcome_equation(data, n_instruments, n_covariates, state,
                                                  g_prior, nu_spike, a_sigma, b_sigma,
                                                  ridge, rng)
        state[intercept_index] = outcome[0]
        state[outcome_covariate_slice] = outcome[1:1 + n_covariate]
        state[receptor_effect_index] = outcome[1 + n_covariate]
        state[outcome_variance_index] = outcome[2 + n_covariate]

        # -- Sub-problem 06: the communication effect vector and its selection.
        communication = _oracle_update_communication_state(data, n_instruments, n_covariates,
                                                           state, g_prior, nu_spike, a_rho,
                                                           b_rho, ridge, rng)
        state[ligand_effect_index] = communication[0]
        state[interaction_index] = communication[1]
        state[inclusion_index] = communication[2]
        state[inclusion_probability_index] = communication[4]

        if sweep >= discarded and (sweep - discarded) % spacing == 0:
            inclusion_trace.append(float(state[inclusion_index]))
            ligand_trace.append(float(state[ligand_effect_index]))
            interaction_trace.append(float(state[interaction_index]))

    posterior_inclusion = float(np.mean(inclusion_trace))
    if not 0.0 < posterior_inclusion < 1.0:
        raise ValueError("the retained sweeps leave the communication odds undefined")

    # -- Sub-problem 07: both verdicts on one odds scale.
    evidence = _oracle_summarise_communication_evidence(naive_summary, posterior_inclusion,
                                                        float(np.mean(ligand_trace)),
                                                        float(np.mean(interaction_trace)),
                                                        float(np.std(ligand)),
                                                        float(np.std(receptor)),
                                                        float(np.std(pathway)))
    return float(evidence[0])

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES (integration tests -- whole pipeline)
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: a small confounded null cohort, whole pipeline ---
        {
            "setup": """import numpy as np
""",
            "call": "float(1.0 + 1.0e6 * run_causal_communication_pipeline(120, 3, 2, n_iterations=600, burn_in=100, thin=2, sampler_seed=5))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_run_causal_communication_pipeline(120, 3, 2, n_iterations=600, burn_in=100, thin=2, sampler_seed=5))",
        },
        # --- Integration: the configuration whose result is the task's final answer ---
        {
            "setup": """import numpy as np
""",
            "call": "float(1.0 + 1.0e6 * run_causal_communication_pipeline())",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_run_causal_communication_pipeline())",
        },
        # --- Integration (boundary): a genuine ligand effect alongside the confounding ---
        {
            "setup": """import numpy as np
""",
            "call": "float(1.0 + 1.0e6 * run_causal_communication_pipeline(240, 3, 2, 0.5, 0.3, 0.5, 0.7, 0.14, 0.0, 77, 1200, 200, 2, sampler_seed=13))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_run_causal_communication_pipeline(240, 3, 2, 0.5, 0.3, 0.5, 0.7, 0.14, 0.0, 77, 1200, 200, 2, sampler_seed=13))",
        },
        # --- Integration (edge): no confounding at all, so there is little for the causal arm to correct ---
        {
            "setup": """import numpy as np
""",
            "call": "float(1.0 + 1.0e6 * run_causal_communication_pipeline(150, 2, 1, 0.6, 0.3, 0.5, 0.0, 0.0, 0.0, 21, 600, 100, 2, sampler_seed=31))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_run_causal_communication_pipeline(150, 2, 1, 0.6, 0.3, 0.5, 0.0, 0.0, 0.0, 21, 600, 100, 2, sampler_seed=31))",
        },
        # --- Invalid: a thinning factor below one ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_causal_communication_pipeline(60, 2, 1, n_iterations=100, burn_in=10, thin=0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_causal_communication_pipeline(60, 2, 1, n_iterations=100, burn_in=10, thin=0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a burn in that consumes every sweep ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_causal_communication_pipeline(60, 2, 1, n_iterations=100, burn_in=100, thin=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_causal_communication_pipeline(60, 2, 1, n_iterations=100, burn_in=100, thin=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
