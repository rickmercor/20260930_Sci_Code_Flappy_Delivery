"""
04_grow_regression_tree

Both tree architectures rest on the same object, a regression tree grown by recursive binary splitting. At a node holding a set of rows, every admissible split is scored, the best is taken, and the two children are grown in turn until a depth limit or a size limit stops the recursion. A node that is not split becomes a leaf whose prediction is the mean response of the rows it holds.

The score is the reduction in the within-node sum of squares. Writing $y$ for the responses at the node and $y_L$, $y_R$ for the two parts a candidate split produces, the parent sum of squares is $\sum (y_i - \overline{y})^2$ and the reduction is that quantity less the same sum computed separately in each part. Maximising the reduction is equivalent to minimising the pooled within-part variance, and it is the criterion that makes a regression tree a piecewise-constant least-squares fit.

Candidate thresholds are the midpoints between consecutive distinct values of a covariate among the rows at the node, which is the coarsest set of thresholds that realises every distinct partition of those rows by that covariate. A candidate is admissible only if both parts retain at least the minimum node size. The rule that decides among equally good candidates has to be stated, because covariates measured on a coarse grid produce exact ties routinely and a tree that breaks them arbitrarily is not reproducible: the first candidate found wins, scanning covariates in index order and thresholds in ascending order. Reductions are compared to within rounding: a later candidate displaces the best so far only if its reduction is larger by more than 1e-12 times the node's sum of squares, so an exact tie that floating point separates in the last digits still goes to the first candidate. Recursion stops when the depth budget is exhausted, when the node holds fewer than twice the minimum node size, when every response at the node is the same value, or when no admissible split reduces the sum of squares by more than 1e-12 times its value at the node. The third of those is not implied by the fourth. Summing a constant vector and subtracting its mean leaves a residue at the last bit, so a node whose responses are identical reports an apparent reduction of order 1e-31 and would otherwise be split on nothing.

The grown tree is returned in flattened array form rather than as a nested structure, one entry per node in the order a depth-first walk creates them, a node before its left subtree and the whole left subtree before the right. Every node carries the mean response of the rows it holds, internal nodes included, where it records what the node would predict if the walk stopped there. A leaf carries a covariate index of minus one, a threshold of not-a-number, and the mean response of its rows; an internal node carries its covariate index, its threshold, and the positions of its two children, with rows satisfying covariate at most threshold going left. That representation is what the ensembles consume, and it makes a tree comparable entry by entry between two implementations.

Returns
-------
dict holding the int64 arrays node_feature, node_left and node_right of shape (n_nodes,), the float64 arrays node_threshold and node_value of shape (n_nodes,), and the native integers n_nodes and n_leaves and the native float train_rss.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def grow_regression_tree(
    features: np.ndarray,
    response: np.ndarray,
    max_depth: int,
    min_node: int,
) -> dict:
    """Grow a least-squares regression tree by exhaustive binary splitting and return it in flattened form.

    Parameters
    ----------
    features : np.ndarray
        Covariates at the rows the tree is grown on, shape (n_rows, n_covariates).
    response : np.ndarray
        Response at those rows, shape (n_rows,).
    max_depth : int
        Maximum number of splits on any root-to-leaf path.
    min_node : int
        Minimum number of rows a node may hold.

    Returns
    -------
    dict
        Under the keys node_feature, node_threshold, node_value, node_left, node_right, n_nodes, n_leaves and train_rss. A leaf's node_left and node_right entries are minus one, and train_rss is the residual sum of squares of the tree's predictions at the rows it was grown on.

    Raises
    ------
    ValueError
        When features is not a two-dimensional array with at least one row and one covariate, when response does not match it in length, when any entry fails to be finite, when max_depth is not an integer of one or more, or when min_node is not an integer of one or more. A node whose responses are all equal is returned as a leaf rather than split.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _best_split(rows, target, min_node):
    """Covariate and threshold giving the largest reduction in the within-node sum of squares.

    Reductions are compared to within 1e-12 of the node's sum of squares. Exact ties are routine on coarse grids, and
    floating point can separate two equal reductions in the last digits, which would hand the tie to whichever
    candidate the rounding favours instead of the first one.
    """
    n_rows, n_covariates = rows.shape
    parent = float(((target - target.mean()) ** 2).sum())
    slack = 1e-12 * parent
    best_gain, best_column, best_threshold = 0.0, -1, float("nan")
    for column in range(n_covariates):
        distinct = np.unique(rows[:, column])
        if distinct.size < 2:
            continue
        for threshold in 0.5 * (distinct[:-1] + distinct[1:]):
            left = rows[:, column] <= threshold
            n_left = int(left.sum())
            if n_left < min_node or n_rows - n_left < min_node:
                continue
            below, above = target[left], target[~left]
            child = float(((below - below.mean()) ** 2).sum() + ((above - above.mean()) ** 2).sum())
            gain = parent - child
            if gain > best_gain + slack:
                best_gain, best_column, best_threshold = gain, column, float(threshold)
    return best_column, best_threshold


def _grow(rows, target, depth, min_node, store):
    """Append one node to the flattened store and return its position."""
    position = len(store["feature"])
    store["feature"].append(-1)
    store["threshold"].append(float("nan"))
    store["value"].append(float(target.mean()))
    store["left"].append(-1)
    store["right"].append(-1)
    if depth <= 0 or rows.shape[0] < 2 * min_node or np.ptp(target) <= 0.0:
        return position
    column, threshold = _best_split(rows, target, min_node)
    if column < 0:
        return position
    below = rows[:, column] <= threshold
    store["feature"][position] = column
    store["threshold"][position] = threshold
    store["left"][position] = _grow(rows[below], target[below], depth - 1, min_node, store)
    store["right"][position] = _grow(rows[~below], target[~below], depth - 1, min_node, store)
    return position


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


def _oracle_grow_regression_tree(
    features: np.ndarray,
    response: np.ndarray,
    max_depth: int,
    min_node: int,
) -> dict:
    """Reference implementation."""
    rows = np.asarray(features, dtype=float)
    target = np.asarray(response, dtype=float)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError("features wants shape (n_rows, n_covariates) with at least one row and one covariate")
    if target.ndim != 1 or target.shape[0] != rows.shape[0]:
        raise ValueError("response wants one finite entry per row of features")
    if not np.isfinite(rows).all() or not np.isfinite(target).all():
        raise ValueError("no covariate or response entry may be infinite or undefined")
    if not isinstance(max_depth, (int, np.integer)) or int(max_depth) < 1:
        raise ValueError("max_depth wants an integer of one or more")
    if not isinstance(min_node, (int, np.integer)) or int(min_node) < 1:
        raise ValueError("min_node wants an integer of one or more")

    store = {"feature": [], "threshold": [], "value": [], "left": [], "right": []}
    _grow(rows, target, int(max_depth), int(min_node), store)
    tree = {
        "node_feature": np.array(store["feature"], dtype=np.int64),
        "node_threshold": np.array(store["threshold"], dtype=float),
        "node_value": np.array(store["value"], dtype=float),
        "node_left": np.array(store["left"], dtype=np.int64),
        "node_right": np.array(store["right"], dtype=np.int64),
    }
    tree["n_nodes"] = int(tree["node_feature"].size)
    tree["n_leaves"] = int((tree["node_feature"] < 0).sum())
    tree["train_rss"] = float(((target - _descend(tree, rows)) ** 2).sum())
    return tree

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, leaf-boundary, tie, and invalid-data cases."""
    setup = """import numpy as np
def grid(n):
    s = np.arange(n, dtype=np.int64)
    return np.stack([((3 * s + 5 * k) % 13) / 12.0 for k in range(4)], axis=1)
X = grid(96)
Y = 0.5 * (X[:, 0] > 0.4) + 0.3 * X[:, 2] - 0.2 * (X[:, 1] > 0.7)
TIED = np.zeros((24, 3)); TIED[12:, 0] = 1.0; TIED[12:, 1] = 1.0; TIED[:, 2] = 0.5
TIED_Y = np.concatenate([np.zeros(12), np.ones(12)])
def digest(out):
    threshold = np.asarray(out["node_threshold"], dtype=float)
    threshold = np.where(np.isnan(threshold), -1.0e300, threshold)
    return np.concatenate((
        np.asarray([out["n_nodes"], out["n_leaves"], out["train_rss"]], dtype=float),
        np.asarray(out["node_feature"], dtype=float), threshold,
        np.asarray(out["node_value"], dtype=float),
        np.asarray(out["node_left"], dtype=float),
        np.asarray(out["node_right"], dtype=float),
    ))
def invalid_feature_rank(fn):
    try:
        fn(X[:, 0], Y, 3, 6)
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(grow_regression_tree(X, Y, 3, 6))",
         "gold_call": "digest(_oracle_grow_regression_tree(X, Y, 3, 6))",
         "tol": 1e-10},
        {"setup": setup,
         "call": "digest(grow_regression_tree(X[:5], np.full(5, 0.37), 3, 3))",
         "gold_call": "digest(_oracle_grow_regression_tree(X[:5], np.full(5, 0.37), 3, 3))",
         "tol": 1e-12},
        {"setup": setup,
         "call": "digest(grow_regression_tree(TIED, TIED_Y, 2, 2))",
         "gold_call": "digest(_oracle_grow_regression_tree(TIED, TIED_Y, 2, 2))",
         "tol": 1e-12},
        {"setup": setup,
         "call": "invalid_feature_rank(grow_regression_tree)",
         "gold_call": "invalid_feature_rank(_oracle_grow_regression_tree)"},
    ]
