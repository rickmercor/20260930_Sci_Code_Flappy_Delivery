"""
05_predict_tree_ensembles

Two of the four architectures are ensembles of regression trees, and they differ in how the ensemble is assembled rather than in the trees themselves.

The first averages trees grown on overlapping subsets of the training rows. Variance reduction by averaging needs the members to differ, and the usual way to make them differ is to resample the training rows at random. A random resample cannot be used where the result must be reproducible by someone who does not share the draw, so the subsets are taken deterministically instead: with $n$ trees, tree $k$ is grown on every training row whose position in the training set is not congruent to $k$ modulo $n$. Each tree then sees a fixed fraction $(n-1)/n$ of the rows, the omitted fractions are disjoint across trees, and the ensemble prediction is the plain mean over the members. The members are grown deep, since averaging is what controls their variance.

The second builds an additive expansion in a forward stagewise fashion. The expansion starts at the mean response of the training rows and, at each round, grows a shallow tree on the current residuals and adds a shrunken multiple of it. With $F_0$ the training mean, $h_r$ the tree grown at round $r$ and nu the learning rate,

$$F_r(x) = F_{r-1}(x) + nu h_r(x),$$

in which $h_r$ is grown on the current residual $y - F_{r-1}$.

Shrinkage below one is what makes the procedure work as regularisation: each round corrects only part of the residual, so the expansion approaches the data slowly and many shallow trees combine into a smooth response rather than one deep tree memorising the rows. The members are grown shallow, since depth here controls the order of interaction the expansion can represent rather than the variance.

Both ensembles are evaluated at an arbitrary matrix of query rows, since the attribution step interrogates them away from the training data. Neither contains a random draw at any point, so both are reproducible from the training data and the stated settings alone.

Returns
-------
dict holding the float64 arrays forest_predictions and boosted_predictions of shape (n_query,), and the native integers forest_leaf_total and boosted_leaf_total and the native floats forest_train_rss and boosted_train_rss.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def predict_tree_ensembles(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    n_trees: int,
    forest_depth: int,
    n_rounds: int,
    boost_depth: int,
    learning_rate: float,
    min_node: int,
) -> dict:
    """Fit the deterministically bagged ensemble and the stagewise boosted expansion and evaluate both at the query rows.

    Parameters
    ----------
    train_features : np.ndarray
        Training covariates, shape (n_train, n_covariates).
    train_response : np.ndarray
        Training response, shape (n_train,).
    query_features : np.ndarray
        Rows at which both ensembles are evaluated, shape (n_query, n_covariates).
    n_trees : int
        Members of the averaged ensemble.
    forest_depth : int
        Depth budget of an averaged member.
    n_rounds : int
        Rounds of the stagewise expansion.
    boost_depth : int
        Depth budget of a stagewise member.
    learning_rate : float
        Shrinkage applied to each stagewise member.
    min_node : int
        Minimum rows a node may hold.

    Returns
    -------
    dict
        Under the keys forest_predictions, boosted_predictions, forest_leaf_total, boosted_leaf_total, forest_train_rss and boosted_train_rss.

    Raises
    ------
    ValueError
        When train_features is not a two-dimensional array with at least one row and one covariate, when train_response does not match it in length, when query_features does not share the covariate count, when any entry fails to be finite, when n_trees is not an integer of two or more, when forest_depth, n_rounds, boost_depth or min_node is not an integer of one or more, or when learning_rate is not finite, above zero and at most one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _descend(tree, rows):
    """Prediction of a flattened tree at every row."""
    out = np.empty(rows.shape[0])
    stack = [(0, np.arange(rows.shape[0]))]
    while stack:
        node, index = stack.pop()
        if index.size == 0:
            continue
        if tree["node_feature"][node] < 0:
            out[index] = tree["node_value"][node]
            continue
        below = rows[index, tree["node_feature"][node]] <= tree["node_threshold"][node]
        stack.append((int(tree["node_left"][node]), index[below]))
        stack.append((int(tree["node_right"][node]), index[~below]))
    return out


def _oracle_predict_tree_ensembles(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    n_trees: int,
    forest_depth: int,
    n_rounds: int,
    boost_depth: int,
    learning_rate: float,
    min_node: int,
) -> dict:
    """Reference implementation."""
    fit_rows = np.asarray(train_features, dtype=float)
    target = np.asarray(train_response, dtype=float)
    query = np.asarray(query_features, dtype=float)
    if fit_rows.ndim != 2 or fit_rows.shape[0] < 1 or fit_rows.shape[1] < 1:
        raise ValueError("train_features wants shape (n_train, n_covariates) with at least one row and one covariate")
    if target.ndim != 1 or target.shape[0] != fit_rows.shape[0]:
        raise ValueError("train_response wants one finite entry per training row")
    if query.ndim != 2 or query.shape[1] != fit_rows.shape[1]:
        raise ValueError("query_features wants the covariate count of train_features")
    if not (np.isfinite(fit_rows).all() and np.isfinite(target).all() and np.isfinite(query).all()):
        raise ValueError("no covariate or response entry may be infinite or undefined")
    if not isinstance(n_trees, (int, np.integer)) or int(n_trees) < 2:
        raise ValueError("n_trees wants an integer of two or more")
    for name, value in (("forest_depth", forest_depth), ("n_rounds", n_rounds),
                        ("boost_depth", boost_depth), ("min_node", min_node)):
        if not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError("%s wants an integer of one or more" % name)
    rate = float(learning_rate)
    if not np.isfinite(rate) or rate <= 0.0 or rate > 1.0:
        raise ValueError("learning_rate wants a finite value above zero and at most one")

    members, leaves = int(n_trees), 0
    position = np.arange(fit_rows.shape[0])
    forest_query = np.zeros(query.shape[0])
    forest_train = np.zeros(fit_rows.shape[0])
    for member in range(members):
        kept = position % members != member
        if not kept.any():
            raise ValueError("n_trees leaves one bagged member with no training rows")
        tree = _oracle_grow_regression_tree(  # noqa: F821
            fit_rows[kept], target[kept], int(forest_depth), int(min_node))
        leaves += tree["n_leaves"]
        forest_query += _descend(tree, query)
        forest_train += _descend(tree, fit_rows)
    forest_query /= members
    forest_train /= members

    base = float(target.mean())
    residual = target - base
    boosted_query = np.full(query.shape[0], base)
    boosted_train = np.full(fit_rows.shape[0], base)
    boosted_leaves = 0
    for _ in range(int(n_rounds)):
        tree = _oracle_grow_regression_tree(  # noqa: F821
            fit_rows, residual, int(boost_depth), int(min_node))
        boosted_leaves += tree["n_leaves"]
        step = _descend(tree, fit_rows)
        residual = residual - rate * step
        boosted_train += rate * step
        boosted_query += rate * _descend(tree, query)
    return {
        "forest_predictions": forest_query,
        "boosted_predictions": boosted_query,
        "forest_leaf_total": int(leaves),
        "boosted_leaf_total": int(boosted_leaves),
        "forest_train_rss": float(((target - forest_train) ** 2).sum()),
        "boosted_train_rss": float(((target - boosted_train) ** 2).sum()),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, small, constant-response, and invalid-data cases."""
    setup = """import numpy as np
def grid(n):
    s = np.arange(n, dtype=np.int64)
    return np.stack([((3 * s + 5 * k) % 13) / 12.0 for k in range(6)], axis=1)
TRAIN = grid(96)
Y = 0.45 + 0.3 * TRAIN[:, 0] - 0.4 * (TRAIN[:, 3] > 0.6) + 0.15 * TRAIN[:, 5] ** 2
QUERY = grid(13) * 0.9 + 0.03
def digest(out):
    return np.concatenate((
        np.asarray([out["forest_leaf_total"], out["boosted_leaf_total"],
                    out["forest_train_rss"], out["boosted_train_rss"]], dtype=float),
        np.asarray(out["forest_predictions"], dtype=float),
        np.asarray(out["boosted_predictions"], dtype=float),
    ))
def invalid_tree_count(fn):
    try:
        fn(TRAIN, Y, QUERY, 1, 4, 8, 2, 0.1, 4)
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(predict_tree_ensembles(TRAIN, Y, QUERY, 4, 4, 8, 2, 0.1, 4))",
         "gold_call": "digest(_oracle_predict_tree_ensembles(TRAIN, Y, QUERY, 4, 4, 8, 2, 0.1, 4))",
         "tol": 1e-9},
        {"setup": setup,
         "call": "digest(predict_tree_ensembles(TRAIN[:24], Y[:24], QUERY[:1], 2, 1, 1, 1, 1.0, 1))",
         "gold_call": "digest(_oracle_predict_tree_ensembles(TRAIN[:24], Y[:24], QUERY[:1], 2, 1, 1, 1, 1.0, 1))",
         "tol": 1e-9},
        {"setup": setup,
         "call": "digest(predict_tree_ensembles(TRAIN[:40], np.full(40, 0.37), QUERY[:4], 4, 3, 5, 2, 0.1, 4))",
         "gold_call": "digest(_oracle_predict_tree_ensembles(TRAIN[:40], np.full(40, 0.37), QUERY[:4], 4, 3, 5, 2, 0.1, 4))",
         "tol": 1e-9},
        {"setup": setup,
         "call": "invalid_tree_count(predict_tree_ensembles)",
         "gold_call": "invalid_tree_count(_oracle_predict_tree_ensembles)"},
    ]
