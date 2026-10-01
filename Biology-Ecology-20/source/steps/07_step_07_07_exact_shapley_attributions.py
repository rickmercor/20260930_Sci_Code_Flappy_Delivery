"""
07_exact_shapley_attributions

An explanation of a single prediction, in the additive attribution family, is a vector that divides the gap between the prediction and a baseline among the covariates. The division that is uniquely determined by the standard fairness axioms is the Shapley value of the cooperative game whose players are the covariates and whose payoff is the model output when only a stated subset of them is known.

The game has to be defined before the value can be computed, and the definition turns on what it means for a covariate to be unknown. Here an unknown covariate is drawn from the empirical distribution of a fixed background set independently of the covariates that are known, so the payoff of a subset $S$ at the observation $x$ is the average model output over the background rows with the entries in $S$ overwritten by those of $x$,

$$v_x(S) = (1/|B|) \sum_{b \in B} f(x_S; b_{-S}),$$

with $B$ the background set. The attribution to covariate $j$ is then the weighted average of what $j$ adds to every subset that excludes it,

$$phi_j(x) = \sum_{S \subseteq F - j} [|S|! (K - |S| - 1)! / K!] [v_x(S + j) - v_x(S)],$$

where $K$ is the number of covariates and $F$ the full set. With six covariates the sum has 64 distinct subsets, so the attribution can be computed exactly by enumeration rather than estimated by sampling. Exactness matters twice over: a sampled attribution carries Monte Carlo error that would be indistinguishable from genuine disagreement between architectures, and it would make the result depend on a random draw.

The construction admits one strong check. Summing the attributions telescopes every path through the subset lattice, so for any model and any observation

$$\sum_j phi_j(x) = v_x(F) - v_x(\emptyset) = f(x) - (1/|B|) \sum_{b \in B} f(b),$$

which is the local accuracy property: the attributions account exactly for the gap between the prediction at the observation and the mean prediction over the background. A residual larger than rounding error in this identity means the enumeration, the weights or the payoff is wrong, and the step reports the largest such residual over all models and observations so that the check cannot be skipped.

Every model in the set is interrogated on the same lattice, so the payoffs for all four architectures at all subsets and all observations are gathered into one query matrix and evaluated in a single pass. This matters in practice, since the architectures are refitted on every call and a naive loop over observations would refit them thousands of times.

Returns
-------
dict holding the float64 array attributions of shape (4, n_query, 6) with the model axis ordered as additive model, averaged ensemble, boosted expansion and kernel smoother, the float64 arrays baselines of shape (4,) and query_predictions of shape (4, n_query), and the native float local_accuracy_residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_shapley_attributions(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_rows: np.ndarray,
    background: np.ndarray,
    model_config: dict,
) -> dict:
    """Compute exact Shapley attributions for every architecture at every query row by enumerating the covariate subsets.
 
    Parameters
    ----------
    train_features : np.ndarray
        Training covariates, shape (n_train, 6).
    train_response : np.ndarray
        Training response, shape (n_train,).
    query_rows : np.ndarray
        Observations to be explained, shape (n_query, 6).
    background : np.ndarray
        Background rows over which absent covariates are averaged, shape (n_background, 6).
    model_config : dict
        Settings under the keys n_knots, penalty_weight, ridge_nugget, n_trees, forest_depth, n_rounds, boost_depth, learning_rate, min_node and bandwidth.
 
    Returns
    -------
    dict
        Under the keys attributions, baselines, query_predictions and local_accuracy_residual.
 
    Raises
    ------
    ValueError
        When train_features is not a two-dimensional array of six columns with at least one row, when train_response does not match it in length, when query_rows or background is not a two-dimensional array of six columns with at least one row, when any entry fails to be finite, or when one of the ten configuration keys is absent.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math
 
import numpy as np
 
N_COVARIATES = 6
CONFIG_KEYS = ("n_knots", "penalty_weight", "ridge_nugget", "n_trees", "forest_depth",
               "n_rounds", "boost_depth", "learning_rate", "min_node", "bandwidth")
 
 
def _hat_basis(values, knots):
    """Piecewise-linear hat functions on the given knots, extended flat beyond the end knots."""
    n_knots = knots.size
    basis = np.zeros((values.size, n_knots))
    for a in range(n_knots):
        column = np.zeros(values.size)
        if a > 0:
            rising = (values > knots[a - 1]) & (values <= knots[a])
            column[rising] = (values[rising] - knots[a - 1]) / (knots[a] - knots[a - 1])
        if a < n_knots - 1:
            falling = (values > knots[a]) & (values < knots[a + 1])
            column[falling] = (knots[a + 1] - values[falling]) / (knots[a + 1] - knots[a])
        column[values == knots[a]] = 1.0
        if a == 0:
            column[values < knots[0]] = 1.0
        if a == n_knots - 1:
            column[values > knots[-1]] = 1.0
        basis[:, a] = column
    return basis
 
 
def _subset_lattice(n_covariates):
    """Every subset of the covariates, in non-decreasing order of size."""
    return list(itertools.chain.from_iterable(
        itertools.combinations(range(n_covariates), size) for size in range(n_covariates + 1)))
 
 
def _shapley_weights(n_covariates):
    """Weight carried by a subset of each size in the Shapley sum."""
    total = math.factorial(n_covariates)
    return np.array([math.factorial(size) * math.factorial(n_covariates - size - 1) / total
                     for size in range(n_covariates)])
 
 
def _oracle_exact_shapley_attributions(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_rows: np.ndarray,
    background: np.ndarray,
    model_config: dict,
) -> dict:
    """Reference implementation."""
    N_COVARIATES = 6
    CONFIG_KEYS = ("n_knots", "penalty_weight", "ridge_nugget", "n_trees", "forest_depth",
                   "n_rounds", "boost_depth", "learning_rate", "min_node", "bandwidth")
    fit_rows = np.asarray(train_features, dtype=float)
    target = np.asarray(train_response, dtype=float)
    query = np.asarray(query_rows, dtype=float)
    pool = np.asarray(background, dtype=float)
    if fit_rows.ndim != 2 or fit_rows.shape[1] != N_COVARIATES or fit_rows.shape[0] < 1:
        raise ValueError("train_features wants shape (n_train, 6) with at least one row")
    if target.ndim != 1 or target.shape[0] != fit_rows.shape[0]:
        raise ValueError("train_response wants one finite entry per training row")
    if query.ndim != 2 or query.shape[1] != N_COVARIATES or query.shape[0] < 1:
        raise ValueError("query_rows wants shape (n_query, 6) with at least one row")
    if pool.ndim != 2 or pool.shape[1] != N_COVARIATES or pool.shape[0] < 1:
        raise ValueError("background wants shape (n_background, 6) with at least one row")
    if not (np.isfinite(fit_rows).all() and np.isfinite(target).all()
            and np.isfinite(query).all() and np.isfinite(pool).all()):
        raise ValueError("no covariate or response entry may be infinite or undefined")
    absent = [key for key in CONFIG_KEYS if key not in model_config]
    if absent:
        raise ValueError("model_config lacks the entry %s" % absent[0])
 
    subsets = _subset_lattice(N_COVARIATES)
    n_query, n_background, n_subsets = query.shape[0], pool.shape[0], len(subsets)
    mixture = np.repeat(pool[None, None, :, :], n_query, axis=0).repeat(n_subsets, axis=1)
    for slot, subset in enumerate(subsets):
        for column in subset:
            mixture[:, slot, :, column] = query[:, column][:, None]
    stacked = np.vstack([mixture.reshape(-1, N_COVARIATES), query, pool])
 
    predicted = _oracle_evaluate_model_consensus(  # noqa: F821
        fit_rows, target, stacked, model_config)["query_predictions"]
    n_models = predicted.shape[0]
    split = n_query * n_subsets * n_background
    payoff = predicted[:, :split].reshape(n_models, n_query, n_subsets, n_background).mean(axis=3)
    at_query = predicted[:, split:split + n_query]
    baselines = predicted[:, split + n_query:].mean(axis=1)
 
    slot_of = {subset: slot for slot, subset in enumerate(subsets)}
    weights = _shapley_weights(N_COVARIATES)
    attributions = np.zeros((n_models, n_query, N_COVARIATES))
    for column in range(N_COVARIATES):
        others = [index for index in range(N_COVARIATES) if index != column]
        for size in range(N_COVARIATES):
            for subset in itertools.combinations(others, size):
                with_column = tuple(sorted(subset + (column,)))
                attributions[:, :, column] += weights[size] * (
                    payoff[:, :, slot_of[with_column]] - payoff[:, :, slot_of[subset]])
    residual = np.abs(attributions.sum(axis=2) - (at_query - baselines[:, None])).max()
    return {
        "attributions": attributions,
        "baselines": baselines,
        "query_predictions": at_query,
        "local_accuracy_residual": float(residual),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, single-row, constant-response, and invalid-data cases."""
    setup = """import numpy as np
CONFIG = {"n_knots": 4, "penalty_weight": 2.0, "ridge_nugget": 1e-8, "n_trees": 4,
          "forest_depth": 4, "n_rounds": 8, "boost_depth": 2, "learning_rate": 0.1,
          "min_node": 4, "bandwidth": 0.12}
def grid(n):
    s = np.arange(n, dtype=np.int64)
    return np.stack([((3 * s + 5 * k) % 13) / 12.0 for k in range(6)], axis=1)
TRAIN = grid(90)
Y = 0.45 + 0.3 * TRAIN[:, 0] - 0.4 * (TRAIN[:, 3] > 0.6) + 0.15 * TRAIN[:, 5] ** 2
QUERY = TRAIN[::17]
POOL = TRAIN[::13]
def digest(out):
    return np.concatenate((
        np.asarray([out["local_accuracy_residual"]], dtype=float),
        np.asarray(out["attributions"], dtype=float).ravel(),
        np.asarray(out["baselines"], dtype=float).ravel(),
        np.asarray(out["query_predictions"], dtype=float).ravel(),
    ))
def invalid_empty_query(fn):
    try:
        fn(TRAIN, Y, TRAIN[:0], POOL, CONFIG)
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(exact_shapley_attributions(TRAIN, Y, QUERY, POOL, CONFIG))",
         "gold_call": "digest(_oracle_exact_shapley_attributions(TRAIN, Y, QUERY, POOL, CONFIG))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "digest(exact_shapley_attributions(TRAIN, Y, POOL[:1], POOL[:1], CONFIG))",
         "gold_call": "digest(_oracle_exact_shapley_attributions(TRAIN, Y, POOL[:1], POOL[:1], CONFIG))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "digest(exact_shapley_attributions(TRAIN[:40], np.full(40, 0.37), QUERY[:2], POOL[:3], CONFIG))",
         "gold_call": "digest(_oracle_exact_shapley_attributions(TRAIN[:40], np.full(40, 0.37), QUERY[:2], POOL[:3], CONFIG))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "invalid_empty_query(exact_shapley_attributions)",
         "gold_call": "invalid_empty_query(_oracle_exact_shapley_attributions)"},
    ]
