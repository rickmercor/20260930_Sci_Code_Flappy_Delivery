"""
Step 10: compose the complete posterior predictive recovery calculation.

Compose stages 6 through 9. Sort scenario labels in increasing order, use each scenario's retained-descendant count as its inventory, aggregate the stage-7 population weights by scenario, and fit one GMM per scenario with uniform fragment weights. Forward every public numerical and scientific control unchanged. Defaults use the eight-scenario, two-campaign instance in the statement.

Returns
-------
Return only the finite posterior predictive scalar at index zero of stage 9.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 10: compose the complete posterior predictive recovery calculation."""
import numpy as np

def solve_strewn_probability(seed=761903, runs=8, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2, components=4, iterations=120, regularization=(0.0064, 0.01, 0.0049), balance_realizations=True, survey_response=(0.06, 0.75, 0.035, 0.9), calibration=((13, 2), (17, 5), (11, 3), (19, 6)), hermite_nodes=64, radial_nodes=56, angular_nodes=88, mass_nodes=48, efficiency_nodes=64, beta_prior=((2.3, 5.7), (4.1, 10.3), (1.2, 3.8)), background_model=(0.1, -0.15, 0.05, 0.7, 0.08, 0.12, 0.08, 0.5, -0.06, 0.12, -0.06, 0.9), background_prior=(2.4, 1.6), background_control=(4, 3.0), background_scale=1.0):
    """Compose the complete finite-inventory posterior predictive probability.

    Parameters
    ----------
    seed, runs : int
        Nonboolean finite integers, seed>=0 and 1<=runs<=1000.
    kick_power : float
        Finite nonnegative radius-ratio exponent inside the kick square root.
    momentum_correct, cascade : bool
        Resolved momentum centering and repeated-breakup switches.
    rtol, max_step : float
        Positive finite RK45 controls.
    max_depth : int
        Nonboolean finite nonnegative breakup depth limit.
    components, iterations : int
        Nonboolean integers, components>=1 and no greater than the number
        of retained impacts in any scenario; iterations>=1.
    regularization : array_like, shape (3,)
        Positive finite standardized covariance diagonal additions.
    balance_realizations : bool
        True supplies equal base proposal probability per event scenario;
        False supplies a survivor-count-weighted base proposal. Stage 7's
        normalized row weights are summed within each scenario to form its
        population-conditional prior. Neither choice reweights marks within
        an individual scenario, whose GMM uses uniform fragment weights.
    survey_response : array_like, shape (4,)
        Positive finite first/second half-response masses and log widths,
        with exactly the meaning specified in infer_campaign_recovery.
    calibration : array_like, shape (J,2)
        Integer binomial trial/success batches as in infer_campaign_recovery.
    hermite_nodes, radial_nodes, angular_nodes, mass_nodes, efficiency_nodes : int
        Nonboolean finite quadrature orders >=2 with the stage 9 meanings.

    beta_prior, background_model, background_prior, background_control, background_scale
        Population-linked visibility and accumulated-meteorite nuisance
        parameters with exactly the domains and meanings of stage 9.

    Returns
    -------
    float
        Probability of at least one new second-campaign recovery, using the
        fixed catalog, priors, measurement law, and target in the statement.

    Raises
    ------
    ValueError
        For domain violations specified in the consuming stages, including
        an inventory smaller than the component count or impossible zero-background catalog.
    RuntimeError
        For predecessor propagation, covariance, or probability failures.

    Notes
    -----
    The default composes the physical scenarios and both specified campaigns.
    Differential probability comparisons use absolute tolerance 1e-7.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 10: compose the complete posterior predictive recovery calculation."""
import numpy as np

def _oracle_solve_strewn_probability(seed=761903, runs=8, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2, components=4, iterations=120, regularization=(0.0064, 0.01, 0.0049), balance_realizations=True, survey_response=(0.06, 0.75, 0.035, 0.9), calibration=((13, 2), (17, 5), (11, 3), (19, 6)), hermite_nodes=64, radial_nodes=56, angular_nodes=88, mass_nodes=48, efficiency_nodes=64, beta_prior=((2.3, 5.7), (4.1, 10.3), (1.2, 3.8)), background_model=(0.1, -0.15, 0.05, 0.7, 0.08, 0.12, 0.08, 0.5, -0.06, 0.12, -0.06, 0.9), background_prior=(2.4, 1.6), background_control=(4, 3.0), background_scale=1.0):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    cloud = _oracle_build_impact_cloud(seed, runs, kick_power, momentum_correct, cascade, rtol, max_step, max_depth)
    row_weights = _oracle_compute_population_weights(cloud, balance_realizations)
    labels = np.unique(cloud[:, 3])
    inventory = np.array([np.count_nonzero(cloud[:, 3] == label) for label in labels])
    scenario_priors = np.array([[np.sum(row[cloud[:, 3] == label]) for label in labels] for row in row_weights])
    models = np.vstack([_oracle_fit_ground_gmm(cloud[cloud[:, 3] == label], np.ones(inventory[i]), components, iterations, regularization) for i, label in enumerate(labels)])
    inference = _oracle_infer_campaign_recovery(models, inventory, scenario_priors, survey_response=survey_response, calibration=calibration, hermite_nodes=hermite_nodes, radial_nodes=radial_nodes, angular_nodes=angular_nodes, mass_nodes=mass_nodes, efficiency_nodes=efficiency_nodes, beta_prior=beta_prior, background_model=background_model, background_prior=background_prior, background_control=background_control, background_scale=background_scale)
    return float(inference[0])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Literal positive, boundary, and input-domain fixtures."""
    return [{'setup': 'marker=1.', 'call': 'solve_strewn_probability(runs=1,components=2,iterations=14,hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)', 'gold_call': '_oracle_solve_strewn_probability(runs=1,components=2,iterations=14,hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)', 'tol': 1e-07}, {'setup': 'marker=1.', 'call': 'solve_strewn_probability(runs=2,components=2,iterations=14,balance_realizations=False,hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)', 'gold_call': '_oracle_solve_strewn_probability(runs=2,components=2,iterations=14,balance_realizations=False,hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)', 'tol': 1e-07}, {'setup': 'marker=1.', 'call': 'solve_strewn_probability(seed=47,runs=1,kick_power=.8,momentum_correct=False,cascade=True,rtol=1e-8,max_step=.9,max_depth=1,components=3,iterations=7,regularization=(.02,.03,.04),balance_realizations=True,survey_response=(.08,.6,.02,1.2),calibration=((7,0),(11,11)),hermite_nodes=32,radial_nodes=18,angular_nodes=28,mass_nodes=24,efficiency_nodes=2)', 'gold_call': '_oracle_solve_strewn_probability(seed=47,runs=1,kick_power=.8,momentum_correct=False,cascade=True,rtol=1e-8,max_step=.9,max_depth=1,components=3,iterations=7,regularization=(.02,.03,.04),balance_realizations=True,survey_response=(.08,.6,.02,1.2),calibration=((7,0),(11,11)),hermite_nodes=32,radial_nodes=18,angular_nodes=28,mass_nodes=24,efficiency_nodes=2)', 'tol': 1e-07}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.', 'call': 'rejects(lambda: solve_strewn_probability(seed=np.inf))', 'gold_call': 'rejects(lambda: _oracle_solve_strewn_probability(seed=np.inf))'}, {'setup': 'marker=1.', 'call': 'solve_strewn_probability(runs=2,cascade=False,components=2,iterations=17,hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)', 'gold_call': '_oracle_solve_strewn_probability(runs=2,cascade=False,components=2,iterations=17,hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24)', 'tol': 1e-07}, {'setup': 'marker=1.', 'call': 'solve_strewn_probability(runs=1,components=2,iterations=14,regularization=(.4,.3,.5),hermite_nodes=3,radial_nodes=3,angular_nodes=5,mass_nodes=3,efficiency_nodes=24)', 'gold_call': '_oracle_solve_strewn_probability(runs=1,components=2,iterations=14,regularization=(.4,.3,.5),hermite_nodes=3,radial_nodes=3,angular_nodes=5,mass_nodes=3,efficiency_nodes=24)', 'tol': 1e-07}, {'setup': 'marker=1.', 'call': 'solve_strewn_probability(runs=1,components=2,iterations=14,hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,beta_prior=((.9,1.4),(2.7,3.2),(1.3,2.1)),background_model=(-.2,.3,-.5,.8,.1,-.07,.1,.6,.09,-.07,.09,.5),background_prior=(1.3,2.2),background_control=(2,1.5),background_scale=.4)', 'gold_call': '_oracle_solve_strewn_probability(runs=1,components=2,iterations=14,hermite_nodes=24,radial_nodes=16,angular_nodes=24,mass_nodes=20,efficiency_nodes=24,beta_prior=((.9,1.4),(2.7,3.2),(1.3,2.1)),background_model=(-.2,.3,-.5,.8,.1,-.07,.1,.6,.09,-.07,.09,.5),background_prior=(1.3,2.2),background_control=(2,1.5),background_scale=.4)', 'tol': 1e-07}]
