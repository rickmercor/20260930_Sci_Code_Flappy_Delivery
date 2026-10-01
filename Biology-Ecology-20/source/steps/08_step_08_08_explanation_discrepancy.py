"""
08_explanation_discrepancy

Four architectures that agree on the response of a reef need not agree on why. Each supplies, at every observation, a vector of attributions over the covariates, and the question this step answers is how far those vectors stand apart. The answer has to be a single number per observation per reference model, comparable across observations, and large exactly where the architectures tell conflicting stories about the same reef in the same year.

Three choices turn a set of attribution vectors into such a number, and each is a decision rather than a detail.

The first is what to compare. The attribution vectors carry one entry per covariate, but not every covariate is of interest: the disturbance covariates are what a manager acts on, while a covariate carrying time is a nuisance direction along which the architectures are free to differ without any ecological consequence. Restricting the vectors to a stated subset of their entries before any norm is taken gives a projected measure that answers the question actually asked.

The second is how to normalise a distance between two vectors. A raw distance is not comparable across observations, because the size of an attribution vector varies for reasons unrelated to agreement: an observation whose prediction is far from the mean model output must carry a large vector, since the attributions sum to that gap, while one at the mean may carry a large vector or a small one, since large attributions can cancel. The distance must therefore be referred to a scale drawn from the explanations themselves. Which scale is the substance of the construction, and it is not the only available one, so the choice has consequences that an implementation must get right rather than guess.

The third is how to combine the pairwise comparisons and how to make the result readable. A comparison of one model against the rest leaves one number per model per observation, and the measure is asymmetric in the pair, so the role of reference model is a choice the analyst makes rather than a symmetry to be averaged away. The per-observation numbers are then referred to a scale computed across the whole evaluation set, which turns the measure into a signal-to-noise ratio in discrepancy space: values well above unity mark observations whose explanations disagree far more than the run of the data, and those are the observations that carry an ecological question a model cannot settle.

This step computes all of that from a stack of attribution vectors and a stated subset of covariates, returning the pairwise relative discrepancies, the per-reference summary, the normalising spread, and the final standardised measure. Which of the available scales and which centring convention the construction uses is the content of the step, and the values it returns depend on getting them right.

Returns
-------
dict holding the float64 arrays relative_discrepancy of shape (n_models, n_models, n_observations) with a zero diagonal, mean_discrepancy of shape (n_models, n_observations), reference_spread of shape (n_models,) and discrepancy_measure of shape (n_models, n_observations), and the native floats largest_measure and smallest_projected_norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def explanation_discrepancy(
    attributions: np.ndarray,
    feature_subset: tuple,
) -> dict:
    """Turn a stack of attribution vectors into the standardised explanation discrepancy measure for every reference model.
 
    Parameters
    ----------
    attributions : np.ndarray
        Attribution vectors, shape (n_models, n_observations, n_covariates).
    feature_subset : tuple
        Positions of the covariates the discrepancy is computed on.
 
    Returns
    -------
    dict
        Under the keys relative_discrepancy, mean_discrepancy, reference_spread, discrepancy_measure, largest_measure and smallest_projected_norm. The first axis of relative_discrepancy, mean_discrepancy, reference_spread and discrepancy_measure is the reference model, and the second axis of relative_discrepancy is the model compared with it.
 
    Raises
    ------
    ValueError
        When attributions is not a three-dimensional array with at least two models, at least two observations and at least one covariate, when an entry fails to be finite, when feature_subset is empty, holds a repeated or non-ascending position, or holds a position outside the covariate range, when the projected attribution vector of some model at some observation has zero length, which leaves the relative discrepancy undefined, or when some reference model's mean discrepancy is the same at every observation, which leaves its normalising spread at zero and the standardised measure undefined.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_explanation_discrepancy(
    attributions: np.ndarray,
    feature_subset: tuple,
) -> dict:
    """Reference implementation."""
    stack = np.asarray(attributions, dtype=float)
    if stack.ndim != 3 or stack.shape[0] < 2 or stack.shape[1] < 2 or stack.shape[2] < 1:
        raise ValueError("attributions wants shape (n_models, n_observations, n_covariates) with at least two models and two observations")
    if not np.isfinite(stack).all():
        raise ValueError("no attribution entry may be infinite or undefined")
    kept = tuple(int(position) for position in feature_subset)
    if len(kept) < 1:
        raise ValueError("feature_subset wants at least one covariate position")
    if any(position < 0 or position >= stack.shape[2] for position in kept):
        raise ValueError("every entry of feature_subset wants a position inside the covariate range")
    if any(kept[index] >= kept[index + 1] for index in range(len(kept) - 1)):
        raise ValueError("feature_subset wants ascending positions without repetition")

    projected = stack[:, :, list(kept)]
    lengths = np.linalg.norm(projected, axis=2)
    if lengths.min() <= 0.0:
        raise ValueError("a projected attribution vector of zero length leaves the relative discrepancy undefined")

    n_models, n_observations = lengths.shape
    relative = np.zeros((n_models, n_models, n_observations))
    for reference in range(n_models):
        for other in range(n_models):
            if other == reference:
                continue
            gap = np.linalg.norm(projected[reference] - projected[other], axis=1)
            relative[reference, other] = gap / lengths[reference]
    mean_discrepancy = relative.sum(axis=1) / (n_models - 1)
    spread = mean_discrepancy.std(axis=1, ddof=1)
    # The range is the exact test. A constant vector leaves the sample standard deviation at
    # rounding level rather than at zero, so a test on the spread alone lets it through and the
    # measure returns a number of order 1e16 instead of being rejected.
    if spread.min() <= 0.0 or np.ptp(mean_discrepancy, axis=1).min() <= 0.0:
        raise ValueError("a reference model whose mean discrepancy is constant across observations leaves the measure undefined")
    measure = mean_discrepancy / spread[:, None]
    return {
        "relative_discrepancy": relative,
        "mean_discrepancy": mean_discrepancy,
        "reference_spread": spread,
        "discrepancy_measure": measure,
        "largest_measure": float(measure.max()),
        "smallest_projected_norm": float(lengths.min()),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, smallest-stack, projected-subset, and invalid-data cases."""
    setup = """import numpy as np
def stack(n_models, n_obs, n_cov):
    m = np.arange(n_models)[:, None, None]
    i = np.arange(n_obs)[None, :, None]
    c = np.arange(n_cov)[None, None, :]
    return 0.4 * np.cos(1.7 * m + 0.31 * i + 0.9 * c) + 0.1 * (m + 1) * np.sin(0.13 * i + c)
A = stack(4, 40, 6)
B = stack(3, 7, 5)
SMALL = np.zeros((2, 2, 3))
SMALL[0, 0] = [3.0, 4.0, 0.0]; SMALL[0, 1] = [3.0, 4.0, 0.0]
SMALL[1, 0] = [3.0, 4.0, 1.0]; SMALL[1, 1] = [3.0, 4.0, 4.0]
ZERO = A.copy(); ZERO[0, 0, :] = 0.0
def digest(out):
    return np.concatenate((
        np.asarray([out["largest_measure"], out["smallest_projected_norm"]], dtype=float),
        np.asarray(out["relative_discrepancy"], dtype=float).ravel(),
        np.asarray(out["mean_discrepancy"], dtype=float).ravel(),
        np.asarray(out["reference_spread"], dtype=float).ravel(),
        np.asarray(out["discrepancy_measure"], dtype=float).ravel(),
    ))
def invalid_zero_norm(fn):
    try:
        fn(ZERO, (0, 1, 2, 3, 4))
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(explanation_discrepancy(A, (0, 1, 2, 3, 4)))",
         "gold_call": "digest(_oracle_explanation_discrepancy(A, (0, 1, 2, 3, 4)))",
         "tol": 1e-10},
        {"setup": setup,
         "call": "digest(explanation_discrepancy(SMALL, (0, 1, 2)))",
         "gold_call": "digest(_oracle_explanation_discrepancy(SMALL, (0, 1, 2)))",
         "tol": 1e-10},
        {"setup": setup,
         "call": "digest(explanation_discrepancy(B, (0, 2, 4)))",
         "gold_call": "digest(_oracle_explanation_discrepancy(B, (0, 2, 4)))",
         "tol": 1e-10},
        {"setup": setup,
         "call": "invalid_zero_norm(explanation_discrepancy)",
         "gold_call": "invalid_zero_norm(_oracle_explanation_discrepancy)"},
    ]
