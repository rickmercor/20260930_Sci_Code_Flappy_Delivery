#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
 
WEIGHT_KEYS = ("cyclone", "cyclone_lag1", "cyclone_lag2", "bleaching", "other")
 
 
def _disturbance_fields(n_sites, n_years):
    """The three intensity fields, laid down by fixed integer maps in exact arithmetic."""
    site = np.arange(n_sites, dtype=np.int64)[:, None]
    year = np.arange(n_years, dtype=np.int64)[None, :]
    cyclone = np.maximum(0, ((7 * site + 11 * year + 3 * site * year) % 23) - 17) / 6.0
    bleaching = np.maximum(0, ((5 * site + 13 * year + site * site) % 19) - 14) / 5.0
    other = np.maximum(0, ((3 * site + 17 * year + year * year) % 29) - 22) / 7.0
    return cyclone, bleaching, other
 
 
def simulate_reef_cover(
    n_sites: int,
    n_years: int,
    growth_rate: float,
    carrying_capacity: float,
    disturbance_weights: dict,
    cover_floor: float,
) -> dict:
    """Reference implementation."""
    WEIGHT_KEYS = ("cyclone", "cyclone_lag1", "cyclone_lag2", "bleaching", "other")
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 1:
        raise ValueError("n_sites wants an integer of one or more")
    if not isinstance(n_years, (int, np.integer)) or int(n_years) < 3:
        raise ValueError("n_years wants an integer of three or more, so that two lags exist")
    growth = float(growth_rate)
    capacity = float(carrying_capacity)
    floor = float(cover_floor)
    if not np.isfinite(growth) or growth <= 0.0:
        raise ValueError("growth_rate wants a finite value above zero")
    if not np.isfinite(capacity) or capacity <= 0.0 or capacity > 1.0:
        raise ValueError("carrying_capacity wants a finite value above zero and at most one")
    absent = [key for key in WEIGHT_KEYS if key not in disturbance_weights]
    if absent:
        raise ValueError("disturbance_weights lacks the entry %s" % absent[0])
    weights = np.array([float(disturbance_weights[key]) for key in WEIGHT_KEYS])
    if not np.isfinite(weights).all() or weights.min() < 0.0:
        raise ValueError("each disturbance weight wants a finite value not below zero")
    if not np.isfinite(floor) or floor < 0.0 or floor >= capacity:
        raise ValueError("cover_floor wants a finite value not below zero and below the carrying capacity")
 
    sites, years = int(n_sites), int(n_years)
    cyclone, bleaching, other = _disturbance_fields(sites, years)
    w_c, w_1, w_2, w_b, w_o = weights
    cover = np.zeros((sites, years))
    cover[:, 0] = 0.25 + 0.01 * (np.arange(sites) % 11)
    for year in range(1, years):
        previous = cover[:, year - 1]
        regrowth = growth * previous * (1.0 - previous / capacity)
        lag_one = cyclone[:, year - 1] if year >= 1 else np.zeros(sites)
        lag_two = cyclone[:, year - 2] if year >= 2 else np.zeros(sites)
        pressure = (w_c * (cyclone[:, year] + w_1 * lag_one + w_2 * lag_two)
                    + w_b * bleaching[:, year] + w_o * other[:, year])
        cover[:, year] = np.clip(previous + regrowth - previous * np.minimum(1.0, pressure),
                                 floor, capacity)
    return {
        "cover": cover,
        "cyclone": cyclone,
        "bleaching": bleaching,
        "other": other,
        "mean_cover": float(cover.mean()),
        "min_cover": float(cover.min()),
        "final_mean_cover": float(cover[:, -1].mean()),
    }

import numpy as np
 
N_COVARIATES = 6
 
 
def assemble_lagged_design(
    cover: np.ndarray,
    cyclone: np.ndarray,
    bleaching: np.ndarray,
    other: np.ndarray,
    eval_stride: int,
    background_stride: int,
) -> dict:
    """Reference implementation."""
    N_COVARIATES = 6
    grid = np.asarray(cover, dtype=float)
    if grid.ndim != 2 or grid.shape[0] < 1 or grid.shape[1] < 3:
        raise ValueError("cover wants shape (n_sites, n_years) with at least one site and three years")
    fields = []
    for field in (cyclone, bleaching, other):
        array = np.asarray(field, dtype=float)
        if array.shape != grid.shape:
            raise ValueError("every intensity array wants the shape of cover")
        fields.append(array)
    if not np.isfinite(grid).all() or not all(np.isfinite(a).all() for a in fields):
        raise ValueError("no entry of cover or of an intensity array may be infinite or undefined")
    if not isinstance(eval_stride, (int, np.integer)) or int(eval_stride) < 2:
        raise ValueError("eval_stride wants an integer of two or more")
    if not isinstance(background_stride, (int, np.integer)) or int(background_stride) < 1:
        raise ValueError("background_stride wants an integer of one or more")
 
    storm, heat, residual = fields
    n_sites, n_years = grid.shape
    span = float(n_years - 3) if n_years > 3 else 1.0
    rows, response, site_of_row, year_of_row = [], [], [], []
    for site in range(n_sites):
        for year in range(2, n_years):
            rows.append([storm[site, year], storm[site, year - 1], storm[site, year - 2],
                         heat[site, year], residual[site, year], (year - 2) / span])
            response.append(grid[site, year])
            site_of_row.append(site)
            year_of_row.append(year)
    features = np.array(rows, dtype=float).reshape(-1, N_COVARIATES)
    target = np.array(response, dtype=float)
    sites = np.array(site_of_row, dtype=np.int64)
    years = np.array(year_of_row, dtype=np.int64)
 
    index = np.arange(features.shape[0])
    held = index % int(eval_stride) == 0
    if held.sum() < 1 or (~held).sum() < 1:
        raise ValueError("eval_stride leaves the evaluation set or the training set empty")
    train_features = features[~held]
    background = train_features[::int(background_stride)]
    if background.shape[0] < 1:
        raise ValueError("background_stride leaves the background set empty")
    return {
        "train_features": train_features,
        "train_response": target[~held],
        "eval_features": features[held],
        "eval_response": target[held],
        "background": background,
        "eval_site": sites[held],
        "eval_year": years[held],
        "n_train": int((~held).sum()),
        "n_eval": int(held.sum()),
        "n_background": int(background.shape[0]),
    }

import numpy as np
 
N_COVARIATES = 6
 
 
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
 
 
def _additive_design(features, n_knots):
    """Intercept column followed by one hat-function block per covariate."""
    knots = np.linspace(0.0, 1.0, n_knots)
    blocks = [np.ones((features.shape[0], 1))]
    for column in range(features.shape[1]):
        blocks.append(_hat_basis(features[:, column], knots))
    return np.hstack(blocks)
 
 
def _second_difference_penalty(n_covariates, n_knots):
    """Sum of the second-difference operators, one per covariate block."""
    width = 1 + n_covariates * n_knots
    penalty = np.zeros((width, width))
    for column in range(n_covariates):
        offset = 1 + column * n_knots
        operator = np.zeros((max(n_knots - 2, 0), width))
        for row in range(n_knots - 2):
            operator[row, offset + row] = 1.0
            operator[row, offset + row + 1] = -2.0
            operator[row, offset + row + 2] = 1.0
        penalty += operator.T @ operator
    return penalty
 
 
def fit_additive_spline_model(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    n_knots: int,
    penalty_weight: float,
    ridge_nugget: float,
) -> dict:
    """Reference implementation."""
    N_COVARIATES = 6
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
    if not isinstance(n_knots, (int, np.integer)) or int(n_knots) < 3:
        raise ValueError("n_knots wants an integer of three or more")
    lam = float(penalty_weight)
    eps = float(ridge_nugget)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("penalty_weight wants a finite value not below zero")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("ridge_nugget wants a finite value above zero")
 
    knots = int(n_knots)
    design = _additive_design(fit_rows, knots)
    penalty = _second_difference_penalty(N_COVARIATES, knots)
    width = design.shape[1]
    normal = design.T @ design + lam * penalty + eps * np.eye(width)
    coefficients = np.linalg.solve(normal, design.T @ target)
    fitted = design @ coefficients
    return {
        "coefficients": coefficients,
        "predictions": _additive_design(query, knots) @ coefficients,
        "train_rss": float(((target - fitted) ** 2).sum()),
        "effective_curvature": float(coefficients @ penalty @ coefficients),
    }

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


def grow_regression_tree(
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
        tree = grow_regression_tree(  # noqa: F821
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
        tree = grow_regression_tree(  # noqa: F821
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
 
 
def evaluate_model_consensus(
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
    additive = fit_additive_spline_model(  # noqa: F821
        fit_rows, target, both, int(model_config["n_knots"]),
        float(model_config["penalty_weight"]), float(model_config["ridge_nugget"]))["predictions"]
    ensembles = predict_tree_ensembles(  # noqa: F821
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
 
 
def exact_shapley_attributions(
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
 
    predicted = evaluate_model_consensus(  # noqa: F821
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

import numpy as np


def explanation_discrepancy(
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

import numpy as np
 
ECOLOGY_KEYS = ("growth_rate", "carrying_capacity", "cover_floor",
                "cyclone", "cyclone_lag1", "cyclone_lag2", "bleaching", "other")
SAMPLING_KEYS = ("eval_stride", "background_stride")
 
 
def flag_consensus_outlier(
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
    record = simulate_reef_cover(  # noqa: F821
        n_sites, n_years, ecology["growth_rate"], ecology["carrying_capacity"],
        weights, ecology["cover_floor"])
    design = assemble_lagged_design(  # noqa: F821
        record["cover"], record["cyclone"], record["bleaching"], record["other"],
        sampling["eval_stride"], sampling["background_stride"])
 
    scored = evaluate_model_consensus(  # noqa: F821
        design["train_features"], design["train_response"], design["eval_features"], model_config)
    held = design["eval_response"]
    spread = float(((held - held.mean()) ** 2).sum())
    residual = ((held[None, :] - scored["query_predictions"]) ** 2).sum(axis=1)
    eval_r_squared = 1.0 - residual / spread if spread > 0.0 else np.zeros(scored["n_models"])
 
    explained = exact_shapley_attributions(  # noqa: F821
        design["train_features"], design["train_response"], design["eval_features"],
        design["background"], model_config)
    discrepancy = explanation_discrepancy(  # noqa: F821
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
SCICODE_GOLD_EOF
