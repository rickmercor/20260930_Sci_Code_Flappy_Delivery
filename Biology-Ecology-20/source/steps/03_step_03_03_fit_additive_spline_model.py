"""
03_fit_additive_spline_model

The first of the four architectures is the statistical one: an additive model in which the response is an intercept plus one smooth function of each covariate, with no interaction terms. Ecologists reach for this class because it admits non-linear responses to environmental drivers while remaining readable, each fitted smooth being a curve that can be plotted and argued about.

The smooths are built from piecewise-linear hat functions. On a grid of $K$ equally spaced knots spanning the unit interval, the hat centred at knot $a$ rises linearly from zero at the neighbouring knot below to one at knot $a$ and falls linearly to zero at the neighbouring knot above; the two end hats are extended flat beyond the ends of the grid so that the basis remains a partition of unity outside the knot range. Every covariate carries its own copy of this basis, and the design matrix is the intercept column followed by the six blocks of $K$ columns.

An unpenalised fit on such a basis is both rough and rank deficient, since each block sums to the intercept column. Both problems are met at once by penalised least squares. Writing $B$ for the design matrix, $y$ for the response, $D_j$ for the second-difference operator acting on the coefficients of covariate $j$, lam for the penalty weight and eps for a small ridge term, the coefficients solve

$$(B^T B + lam \sum_j D_j^T D_j + eps I) beta = B^T y.$$

The second-difference penalty charges curvature in the knot direction, which is the discrete analogue of the integrated squared second derivative used by spline smoothers, so raising lam drives each smooth towards a straight line rather than towards zero. The ridge term is not a modelling choice but a numerical one: it removes the exact rank deficiency between the intercept and the six blocks. Because the null space it fixes lies in the kernel of $B$, it shifts the coefficients without shifting the fitted values, so predictions are unaffected by its value at any small setting.

Once fitted, the model is evaluated at an arbitrary matrix of query rows. That matters downstream, because an attribution method has to interrogate the fitted function at points that are mixtures of an observation and a background row rather than at the training data.

Returns
-------
dict holding the float64 arrays coefficients of shape (1 + 6 n_knots,) and predictions of shape (n_query,), and the native floats train_rss and effective_curvature.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_additive_spline_model(
    train_features: np.ndarray,
    train_response: np.ndarray,
    query_features: np.ndarray,
    n_knots: int,
    penalty_weight: float,
    ridge_nugget: float,
) -> dict:
    """Fit a penalised additive model on piecewise-linear hat bases and evaluate it at the query rows.
 
    Parameters
    ----------
    train_features : np.ndarray
        Training covariates, shape (n_train, 6).
    train_response : np.ndarray
        Training response, shape (n_train,).
    query_features : np.ndarray
        Rows at which the fitted model is evaluated, shape (n_query, 6).
    n_knots : int
        Knots per covariate.
    penalty_weight : float
        Weight on the second-difference penalty.
    ridge_nugget : float
        Weight on the ridge term.
 
    Returns
    -------
    dict
        Under the keys coefficients, predictions, train_rss and effective_curvature. train_rss is the residual sum of squares at the training rows, and effective_curvature is beta^T (sum_j D_j^T D_j) beta, the squared second differences of the fitted coefficients summed over every covariate block, without the penalty weight.
 
    Raises
    ------
    ValueError
        When train_features is not a two-dimensional array of six columns with at least one row, when train_response does not match it in length, when query_features does not carry six columns, when any entry fails to be finite, when n_knots is not an integer of three or more, when penalty_weight is not finite and not below zero, or when ridge_nugget is not finite and above zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
 
 
def _oracle_fit_additive_spline_model(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, extrapolation, and invalid-data cases."""
    setup = """import numpy as np
def grid(n):
    s = np.arange(n, dtype=np.int64)
    return np.stack([((3 * s + 5 * k) % 13) / 12.0 for k in range(6)], axis=1)
TRAIN = grid(120)
Y = 0.4 + 0.3 * TRAIN[:, 0] - 0.5 * TRAIN[:, 3] ** 2 + 0.2 * np.cos(4.0 * TRAIN[:, 5])
QUERY = grid(17) * 0.87 + 0.05
OUTSIDE = np.clip(grid(9) * 2.4 - 0.7, -0.7, 1.7)
def digest(out):
    coefficients = np.asarray(out["coefficients"], dtype=float)
    predictions = np.asarray(out["predictions"], dtype=float)
    return np.concatenate((
        np.asarray([coefficients.size, predictions.size, out["train_rss"],
                    out["effective_curvature"]], dtype=float),
        predictions,
    ))
def invalid_knot_count(fn):
    try:
        fn(TRAIN, Y, QUERY, 2, 2.0, 1e-8)
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(fit_additive_spline_model(TRAIN, Y, QUERY, 5, 2.0, 1e-8))",
         "gold_call": "digest(_oracle_fit_additive_spline_model(TRAIN, Y, QUERY, 5, 2.0, 1e-8))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "digest(fit_additive_spline_model(TRAIN[:40], Y[:40], TRAIN[:1], 3, 0.0, 1e-8))",
         "gold_call": "digest(_oracle_fit_additive_spline_model(TRAIN[:40], Y[:40], TRAIN[:1], 3, 0.0, 1e-8))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "digest(fit_additive_spline_model(TRAIN[:40], Y[:40], OUTSIDE, 5, 20.0, 1e-8))",
         "gold_call": "digest(_oracle_fit_additive_spline_model(TRAIN[:40], Y[:40], OUTSIDE, 5, 20.0, 1e-8))",
         "tol": 1e-8},
        {"setup": setup,
         "call": "invalid_knot_count(fit_additive_spline_model)",
         "gold_call": "invalid_knot_count(_oracle_fit_additive_spline_model)"},
    ]
