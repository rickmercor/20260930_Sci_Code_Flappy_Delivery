"""
09_flag_consensus_outlier

This final step runs the whole workflow from the ecological parameters to the single number the study reports, and it exists because the parts only mean something in sequence.

Step 01 advances mean hard coral cover at every site under logistic regrowth and lagged disturbance mortality, laying down the three disturbance fields from fixed integer maps so that the record carries a known mechanism and no random draw. Step 02 turns that record into a supervised problem, building the six covariates, dropping the two years for which the cyclone lags do not exist, and splitting the rows deterministically into training rows, evaluation rows and the background set the attributions average over. Steps 03, 04 and 05 supply three of the four architectures, the penalised additive model, the regression tree that both ensembles are built from, and the two ensembles themselves; step 06 adds the kernel smoother and puts the four on a common footing, fitted on the same rows and evaluated at the same query points. Step 07 explains every evaluation observation under every architecture, computing exact Shapley attributions by enumerating the covariate subsets against the background set, and checks them against the local accuracy identity. Step 08 turns the four attribution vectors at an observation into the standardised explanation discrepancy measure, once for each choice of reference model.

What remains is the reading of that measure, and it is the point of the exercise. A large measure under one reference model says that the observation stands apart when that model is taken as the reference. The statement is not symmetric, because the normalisation divides by the reference model's own explanation length, so an observation at which one model explains little shows a large discrepancy under that model and a smaller one under the others. Taking the smallest of the measures keeps only the part of the discrepancy that survives every choice of reference, which is the strictest form of the rule that an observation flagged under more than one reference is a consistent outlier rather than an artefact of one comparison. It does not apportion the disagreement: a single architecture that contradicts the rest at one observation raises the measure under every reference. The observation the framework directs to a domain expert first is the one whose smallest measure is largest.

That number, the largest over evaluation observations of the smallest measure over reference models, is what this step returns, along with the identity of the observation it belongs to, the four measures at that observation, the runner-up so that the separation can be judged, the out-of-sample skill of the four architectures, which the framework requires to be comparable before any explanation is compared, and the local accuracy residual of the attribution stage. It also returns the global ranking each architecture gives the covariates, ordered by the mean of the absolute attribution over the evaluation set with ties going to the lower covariate index, because the contrast between that ranking and what the architectures do at a single observation is the finding the whole construction exists to make.

Returns
-------
dict holding the native floats consensus_discrepancy, runner_up_discrepancy and local_accuracy_residual, the native integers flagged_row, flagged_site and flagged_year, the float64 arrays flagged_reference_measures of shape (4,), eval_r_squared of shape (4,) and reference_spread of shape (4,), and the int64 array global_ranking of shape (4, 6) holding each architecture's covariate ordering.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def flag_consensus_outlier(
    n_sites: int,
    n_years: int,
    ecology: dict,
    sampling: dict,
    model_config: dict,
    feature_subset: tuple,
) -> dict:
    """Run the whole workflow and report the strength and the identity of the most contested evaluation observation.
 
    Parameters
    ----------
    n_sites : int
        Number of monitoring sites.
    n_years : int
        Number of years in the record.
    ecology : dict
        Simulator settings under the keys growth_rate, carrying_capacity, cover_floor, cyclone, cyclone_lag1, cyclone_lag2, bleaching and other. The last five are the disturbance weights of step 01, with cyclone_lag1 and cyclone_lag2 the relative lag factors r_1 and r_2 inside the cyclone term.
    sampling : dict
        Row selection under the keys eval_stride and background_stride.
    model_config : dict
        Architecture settings under the keys n_knots, penalty_weight, ridge_nugget, n_trees, forest_depth, n_rounds, boost_depth, learning_rate, min_node and bandwidth.
    feature_subset : tuple
        Positions of the covariates the discrepancy is computed on.
 
    Returns
    -------
    dict
        Under the keys consensus_discrepancy, runner_up_discrepancy, local_accuracy_residual, flagged_row, flagged_site, flagged_year, flagged_reference_measures, eval_r_squared, reference_spread and global_ranking. flagged_row is the position of the flagged observation among the evaluation rows, counted from zero, not its index in the full design, and eval_r_squared is zero for every model when the evaluation response has no spread.
 
    Raises
    ------
    ValueError
        When one of the eight ecology keys or one of the two sampling keys is absent, or when any argument fails the validation of the step it is passed to.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
ECOLOGY_KEYS = ("growth_rate", "carrying_capacity", "cover_floor",
                "cyclone", "cyclone_lag1", "cyclone_lag2", "bleaching", "other")
SAMPLING_KEYS = ("eval_stride", "background_stride")
 
 
def _oracle_flag_consensus_outlier(
    n_sites: int,
    n_years: int,
    ecology: dict,
    sampling: dict,
    model_config: dict,
    feature_subset: tuple,
) -> dict:
    """Reference implementation."""
    ECOLOGY_KEYS = ("growth_rate", "carrying_capacity", "cover_floor",
                    "cyclone", "cyclone_lag1", "cyclone_lag2", "bleaching", "other")
    SAMPLING_KEYS = ("eval_stride", "background_stride")
    absent = [key for key in ECOLOGY_KEYS if key not in ecology]
    if absent:
        raise ValueError("ecology lacks the entry %s" % absent[0])
    absent = [key for key in SAMPLING_KEYS if key not in sampling]
    if absent:
        raise ValueError("sampling lacks the entry %s" % absent[0])
 
    weights = {key: ecology[key] for key in
               ("cyclone", "cyclone_lag1", "cyclone_lag2", "bleaching", "other")}
    record = _oracle_simulate_reef_cover(  # noqa: F821
        n_sites, n_years, ecology["growth_rate"], ecology["carrying_capacity"],
        weights, ecology["cover_floor"])
    design = _oracle_assemble_lagged_design(  # noqa: F821
        record["cover"], record["cyclone"], record["bleaching"], record["other"],
        sampling["eval_stride"], sampling["background_stride"])
 
    scored = _oracle_evaluate_model_consensus(  # noqa: F821
        design["train_features"], design["train_response"], design["eval_features"], model_config)
    held = design["eval_response"]
    spread = float(((held - held.mean()) ** 2).sum())
    residual = ((held[None, :] - scored["query_predictions"]) ** 2).sum(axis=1)
    eval_r_squared = 1.0 - residual / spread if spread > 0.0 else np.zeros(scored["n_models"])
 
    explained = _oracle_exact_shapley_attributions(  # noqa: F821
        design["train_features"], design["train_response"], design["eval_features"],
        design["background"], model_config)
    discrepancy = _oracle_explanation_discrepancy(  # noqa: F821
        explained["attributions"], feature_subset)
 
    importance = np.abs(explained["attributions"]).mean(axis=1)
    ranking = np.argsort(-importance, axis=1, kind="stable").astype(np.int64)
 
    measure = discrepancy["discrepancy_measure"]
    weakest = measure.min(axis=0)
    flagged = int(np.argmax(weakest))
    ordered = np.sort(weakest)[::-1]
    return {
        "consensus_discrepancy": float(weakest[flagged]),
        "runner_up_discrepancy": float(ordered[1]) if ordered.size > 1 else float("nan"),
        "local_accuracy_residual": float(explained["local_accuracy_residual"]),
        "flagged_row": flagged,
        "flagged_site": int(design["eval_site"][flagged]),
        "flagged_year": int(design["eval_year"][flagged]),
        "flagged_reference_measures": measure[:, flagged],
        "eval_r_squared": eval_r_squared,
        "reference_spread": discrepancy["reference_spread"],
        "global_ranking": ranking,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, smaller-record, full-projection, and invalid-data cases."""
    setup = """import numpy as np
ECOLOGY = {"growth_rate": 0.8, "carrying_capacity": 0.80, "cover_floor": 0.02,
           "cyclone": 0.55, "cyclone_lag1": 0.45, "cyclone_lag2": 0.20,
           "bleaching": 0.30, "other": 0.18}
SAMPLING = {"eval_stride": 6, "background_stride": 8}
SMALL = {"n_knots": 4, "penalty_weight": 2.0, "ridge_nugget": 1e-8, "n_trees": 4,
         "forest_depth": 4, "n_rounds": 10, "boost_depth": 2, "learning_rate": 0.1,
         "min_node": 4, "bandwidth": 0.12}
ALT = dict(SMALL, n_rounds=6, bandwidth=0.15)
def digest(out):
    return np.concatenate((
        np.asarray([out["consensus_discrepancy"], out["runner_up_discrepancy"],
                    out["local_accuracy_residual"], out["flagged_row"],
                    out["flagged_site"], out["flagged_year"]], dtype=float),
        np.asarray(out["flagged_reference_measures"], dtype=float).ravel(),
        np.asarray(out["eval_r_squared"], dtype=float).ravel(),
        np.asarray(out["reference_spread"], dtype=float).ravel(),
        np.asarray(out["global_ranking"], dtype=float).ravel(),
    ))
def invalid_ecology(fn):
    bad = dict(ECOLOGY); bad.pop("other")
    try:
        fn(12, 16, bad, SAMPLING, SMALL, (0, 1, 2, 3, 4))
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(flag_consensus_outlier(12, 16, ECOLOGY, SAMPLING, SMALL, (0, 1, 2, 3, 4)))",
         "gold_call": "digest(_oracle_flag_consensus_outlier(12, 16, ECOLOGY, SAMPLING, SMALL, (0, 1, 2, 3, 4)))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "digest(flag_consensus_outlier(10, 14, ECOLOGY, {\"eval_stride\": 5, \"background_stride\": 5}, ALT, (0, 1, 2, 3, 4)))",
         "gold_call": "digest(_oracle_flag_consensus_outlier(10, 14, ECOLOGY, {\"eval_stride\": 5, \"background_stride\": 5}, ALT, (0, 1, 2, 3, 4)))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "digest(flag_consensus_outlier(14, 18, ECOLOGY, SAMPLING, ALT, (0, 1, 2, 3, 4, 5)))",
         "gold_call": "digest(_oracle_flag_consensus_outlier(14, 18, ECOLOGY, SAMPLING, ALT, (0, 1, 2, 3, 4, 5)))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "invalid_ecology(flag_consensus_outlier)",
         "gold_call": "invalid_ecology(_oracle_flag_consensus_outlier)"},
    ]
