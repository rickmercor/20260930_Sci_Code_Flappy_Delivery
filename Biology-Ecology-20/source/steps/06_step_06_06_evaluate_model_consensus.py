"""
06_evaluate_model_consensus

A framework that compares explanations across architectures needs a set of architectures that differ in how they represent the response while agreeing on what they are given: the same covariates, the same training rows, and comparable predictive skill. Without the first of those the comparison is trivial, and without the last it is confounded, because a model that fits worse can be dismissed on accuracy grounds before its explanation is examined at all.

Four architectures are used here and they span the usual classes. The additive model is smooth and has no interactions, so its response to one covariate is a curve that does not depend on the others. The averaged tree ensemble is piecewise constant and captures interactions, but only those its splits happen to isolate. The stagewise boosted expansion is also piecewise constant and, at a depth of three, represents interactions of up to third order, though it reaches them by many small corrections rather than by a few deep splits. The kernel smoother is smooth, fully local, and represents interactions of every order through the joint distance in covariate space, with no parametric form at all.

The fourth of these takes the place of the feed-forward network that would otherwise complete the set. A network fitted by stochastic gradient descent from a random initialisation is not reproducible without sharing the initialisation and the batch order, and a graded quantity built on it would be unreachable rather than merely difficult. The kernel smoother preserves what the network contributes to the comparison, a smooth non-parametric response with unrestricted interactions and no additive structure, while remaining a deterministic function of the training data. Its prediction at $x$ is the weight-averaged training response

$$m(x) = \sum_i w_i(x) y_i / \sum_i w_i(x),$$

in which the weight of training row $i$ is $w_i(x) = \exp[-\|x - x_i\|^2 / (2 h^2)]$, computed after subtracting the squared distance to the nearest training row, which cancels between the numerator and the denominator and keeps the weights of a distant query row from underflowing together,

with $h$ the bandwidth and the norm the Euclidean one on the covariate vector. Because the weights vary continuously with $x$, the smoother has no ordering step and therefore no tie-breaking convention, which a nearest-neighbour rule would need: covariates on a coarse grid put many training rows at exactly equal distance, and the prediction would then depend on how equal distances are sorted.

The four predictions are returned stacked in a fixed order, additive model first, then the averaged ensemble, the boosted expansion and the kernel smoother. That order is carried unchanged through the attribution and discrepancy steps, so that a reference model can be named by its position.

Returns
-------
dict holding the float64 arrays query_predictions of shape (4, n_query), train_predictions of shape (4, n_train) and train_r_squared of shape (4,), and the native integer n_models.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_model_consensus(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    model_config: dict,
) -> dict:
    """Fit the four architectures on common training rows and evaluate every one of them at the query rows.
 
    Parameters
    ----------
    train_features : np.ndarray
        Training covariates, shape (n_train, 6).
    train_response : np.ndarray
        Training response, shape (n_train,).
    query_features : np.ndarray
        Rows at which every architecture is evaluated, shape (n_query, 6).
    model_config : dict
        Settings under the keys n_knots, penalty_weight, ridge_nugget, n_trees, forest_depth, n_rounds, boost_depth, learning_rate, min_node and bandwidth.
 
    Returns
    -------
    dict
        Under the keys query_predictions, train_predictions, train_r_squared and n_models, with the model axis ordered as additive model, averaged ensemble, boosted expansion, kernel smoother. train_r_squared is one minus the residual sum of squares over the total sum of squares about the mean training response, and is zero for every model when the training response has no spread.
 
    Raises
    ------
    ValueError
        When train_features is not a two-dimensional array of six columns with at least one row, when train_response does not match it in length, when query_features does not carry six columns, when any entry fails to be finite, when one of the ten configuration keys is absent, when bandwidth is not finite and above zero, or when bandwidth is so small that twice its square underflows to zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
N_COVARIATES = 6
CONFIG_KEYS = ("n_knots", "penalty_weight", "ridge_nugget", "n_trees", "forest_depth",
               "n_rounds", "boost_depth", "learning_rate", "min_node", "bandwidth")
 
 
def _kernel_smoother(fit_rows, target, query, bandwidth, block=4096):
    """Nadaraya-Watson prediction under a Gaussian kernel of the stated bandwidth.
 
    The squared distance to the nearest training row is subtracted before the exponential. The
    shift cancels between the numerator and the denominator, so it changes nothing about the
    estimator, and it keeps the weights of a query row far from every training row from
    underflowing together and leaving the ratio undefined.
    """
    out = np.empty(query.shape[0])
    for start in range(0, query.shape[0], block):
        chunk = query[start:start + block]
        gap = ((chunk[:, None, :] - fit_rows[None, :, :]) ** 2).sum(axis=2)
        gap = gap - gap.min(axis=1, keepdims=True)
        weight = np.exp(-gap / (2.0 * bandwidth * bandwidth))
        out[start:start + block] = (weight @ target) / weight.sum(axis=1)
    return out
 
 
def _oracle_evaluate_model_consensus(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    model_config: dict,
) -> dict:
    """Reference implementation."""
    N_COVARIATES = 6
    CONFIG_KEYS = ("n_knots", "penalty_weight", "ridge_nugget", "n_trees", "forest_depth",
                   "n_rounds", "boost_depth", "learning_rate", "min_node", "bandwidth")
    fit_rows = np.asarray(train_features, dtype=float)
    target = np.asarray(train_response, dtype=float)
    query = np.asarray(query_features, dtype=float)
    if fit_rows.ndim != 2 or fit_rows.shape[1] != N_COVARIATES or fit_rows.shape[0] < 1:
        raise ValueError("train_features wants shape (n_train, 6) with at least one row")
    if target.ndim != 1 or target.shape[0] != fit_rows.shape[0]:
        raise ValueError("train_response wants one finite entry per training row")
    if query.ndim != 2 or query.shape[1] != N_COVARIATES:
        raise ValueError("query_features wants shape (n_query, 6)")
    if not (np.isfinite(fit_rows).all() and np.isfinite(target).all() and np.isfinite(query).all()):
        raise ValueError("no covariate or response entry may be infinite or undefined")
    absent = [key for key in CONFIG_KEYS if key not in model_config]
    if absent:
        raise ValueError("model_config lacks the entry %s" % absent[0])
    bandwidth = float(model_config["bandwidth"])
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth wants a finite value above zero")
    if 2.0 * bandwidth * bandwidth <= 0.0:
        raise ValueError("bandwidth is so small that twice its square underflows, which leaves every kernel weight undefined")
 
    both = np.vstack([query, fit_rows])
    additive = _oracle_fit_additive_spline_model(  # noqa: F821
        fit_rows, target, both, int(model_config["n_knots"]),
        float(model_config["penalty_weight"]), float(model_config["ridge_nugget"]))["predictions"]
    ensembles = _oracle_predict_tree_ensembles(  # noqa: F821
        fit_rows, target, both, int(model_config["n_trees"]), int(model_config["forest_depth"]),
        int(model_config["n_rounds"]), int(model_config["boost_depth"]),
        float(model_config["learning_rate"]), int(model_config["min_node"]))
    smoothed = _kernel_smoother(fit_rows, target, both, bandwidth)
    stacked = np.vstack([additive, ensembles["forest_predictions"],
                         ensembles["boosted_predictions"], smoothed])
 
    n_query = query.shape[0]
    train_block = stacked[:, n_query:]
    spread = float(((target - target.mean()) ** 2).sum())
    residual = ((target[None, :] - train_block) ** 2).sum(axis=1)
    return {
        "query_predictions": stacked[:, :n_query],
        "train_predictions": train_block,
        "train_r_squared": 1.0 - residual / spread if spread > 0.0 else np.zeros(4),
        "n_models": 4,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, single-query, constant-response, and invalid-data cases."""
    setup = """import numpy as np
CONFIG = {"n_knots": 4, "penalty_weight": 2.0, "ridge_nugget": 1e-8, "n_trees": 4,
          "forest_depth": 4, "n_rounds": 8, "boost_depth": 2, "learning_rate": 0.1,
          "min_node": 4, "bandwidth": 0.12}
def grid(n):
    s = np.arange(n, dtype=np.int64)
    return np.stack([((3 * s + 5 * k) % 13) / 12.0 for k in range(6)], axis=1)
TRAIN = grid(90)
Y = 0.45 + 0.3 * TRAIN[:, 0] - 0.4 * (TRAIN[:, 3] > 0.6) + 0.15 * TRAIN[:, 5] ** 2
QUERY = grid(11) * 0.9 + 0.03
def digest(out):
    return np.concatenate((
        np.asarray([out["n_models"]], dtype=float),
        np.asarray(out["query_predictions"], dtype=float).ravel(),
        np.asarray(out["train_predictions"], dtype=float).ravel(),
        np.asarray(out["train_r_squared"], dtype=float).ravel(),
    ))
def invalid_config(fn):
    bad = dict(CONFIG); bad.pop("bandwidth")
    try:
        fn(TRAIN, Y, QUERY, bad)
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(evaluate_model_consensus(TRAIN, Y, QUERY, CONFIG))",
         "gold_call": "digest(_oracle_evaluate_model_consensus(TRAIN, Y, QUERY, CONFIG))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "digest(evaluate_model_consensus(TRAIN[:40], Y[:40], QUERY[:1], CONFIG))",
         "gold_call": "digest(_oracle_evaluate_model_consensus(TRAIN[:40], Y[:40], QUERY[:1], CONFIG))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "digest(evaluate_model_consensus(TRAIN[:40], np.full(40, 0.37), QUERY[:4], CONFIG))",
         "gold_call": "digest(_oracle_evaluate_model_consensus(TRAIN[:40], np.full(40, 0.37), QUERY[:4], CONFIG))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "invalid_config(evaluate_model_consensus)",
         "gold_call": "invalid_config(_oracle_evaluate_model_consensus)"},
    ]
