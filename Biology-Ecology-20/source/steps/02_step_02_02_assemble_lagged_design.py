"""
02_assemble_lagged_design

A disturbance record and a cover record become a supervised learning problem only after three decisions are made: which covariates carry the disturbance history, which site-years are admissible as rows, and how the rows are divided between fitting and evaluation.

The covariate set follows the mortality structure of the system. Cyclone intensity acts over the current year and the two before it, so it contributes three covariates; thermal stress and the residual class act in the current year only, so they contribute one each. A sixth covariate carries time itself, rescaled to the unit interval so that every covariate occupies the same range and no covariate dominates a distance or a penalty through its units alone. With $t$ running over the admissible years and $T$ the number of years in the record, the time covariate is $(t - 2)/(T - 3)$, which is zero in the first admissible year and one in the last. The covariate order is fixed throughout as cyclone at lag zero, cyclone at lag one, cyclone at lag two, bleaching, other, time.

A site-year is admissible only if both cyclone lags exist, so the first two years of the record are dropped and each site contributes $T - 2$ rows. Rows are laid out site-major: all admissible years of the first site in ascending order, then all admissible years of the second, and so on, which makes the row index a deterministic function of the site and the year and lets a later step recover the identity of any flagged row.

The split is deterministic by construction rather than drawn at random, because a graded quantity that depends on a random partition is not reproducible by anyone who does not share the draw. Rows whose index is divisible by the evaluation stride form the evaluation set; every other row is a training row. Taking every $k$th row rather than a contiguous block leaves gaps in both space and time, which is what a random split is normally used to achieve: the training set then covers the full range of disturbance conditions rather than one era or one region of the domain.

One further subset is needed downstream. An attribution method that measures what a covariate contributes to a prediction has to say what it would mean for that covariate to be absent, and the usual answer is to average the model over the empirical distribution of the absent covariates. That average is taken over a background set, here every $m$th training row, small enough that an exact enumeration over covariate subsets stays cheap and fixed so that the attribution is a deterministic function of the data.

Returns
-------
dict holding the float64 arrays train_features of shape (n_train, 6), train_response of shape (n_train,), eval_features of shape (n_eval, 6), eval_response of shape (n_eval,) and background of shape (n_background, 6); the int64 arrays eval_site and eval_year of shape (n_eval,) carrying the site index and the year index of each evaluation row; and the native integers n_train, n_eval and n_background.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_lagged_design(
    cover: np.ndarray,
    cyclone: np.ndarray,
    bleaching: np.ndarray,
    other: np.ndarray,
    eval_stride: int,
    background_stride: int,
) -> dict:
    """Build the lagged covariate matrix and split it deterministically into training, evaluation and background sets.
 
    Parameters
    ----------
    cover : np.ndarray
        Mean hard coral cover, shape (n_sites, n_years).
    cyclone : np.ndarray
        Cyclone intensity on the same grid.
    bleaching : np.ndarray
        Bleaching intensity on the same grid.
    other : np.ndarray
        Residual disturbance intensity on the same grid.
    eval_stride : int
        Row stride that selects the evaluation set.
    background_stride : int
        Row stride that selects the background set from the training rows.
 
    Returns
    -------
    dict
        Under the keys train_features, train_response, eval_features, eval_response, background, eval_site, eval_year, n_train, n_eval and n_background. When n_years is 3, the time covariate of the single admissible year is 0.
 
    Raises
    ------
    ValueError
        When cover is not a two-dimensional array with at least one site and at least three years, when the three intensity arrays do not share the shape of cover, when any entry fails to be finite, when eval_stride is not an integer of two or more, when background_stride is not an integer of one or more, or when either stride leaves one of the three sets empty.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
N_COVARIATES = 6
 
 
def _oracle_assemble_lagged_design(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-data cases."""
    setup = """import numpy as np
def fields(n_sites, n_years):
    s = np.arange(n_sites, dtype=np.int64)[:, None]
    t = np.arange(n_years, dtype=np.int64)[None, :]
    cyc = np.maximum(0, ((7 * s + 11 * t + 3 * s * t) % 23) - 17) / 6.0
    dhw = np.maximum(0, ((5 * s + 13 * t + s * s) % 19) - 14) / 5.0
    oth = np.maximum(0, ((3 * s + 17 * t + t * t) % 29) - 22) / 7.0
    cover = 0.3 + 0.2 * np.cos(0.4 * s + 0.3 * t) - 0.1 * cyc
    return cover, cyc, dhw, oth
NORMAL = fields(20, 22)
SMALL = fields(1, 5)
WIDE = fields(4, 30)
def digest(out):
    return np.concatenate((
        np.asarray([out["n_train"], out["n_eval"], out["n_background"]], dtype=float),
        np.asarray(out["train_features"], dtype=float).ravel(),
        np.asarray(out["train_response"], dtype=float).ravel(),
        np.asarray(out["eval_features"], dtype=float).ravel(),
        np.asarray(out["eval_response"], dtype=float).ravel(),
        np.asarray(out["background"], dtype=float).ravel(),
        np.asarray(out["eval_site"], dtype=float).ravel(),
        np.asarray(out["eval_year"], dtype=float).ravel(),
    ))
def invalid_short_record(fn):
    cover, cyc, dhw, oth = NORMAL
    try:
        fn(cover[:, :2], cyc[:, :2], dhw[:, :2], oth[:, :2], 4, 3)
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(assemble_lagged_design(*NORMAL, 6, 8))",
         "gold_call": "digest(_oracle_assemble_lagged_design(*NORMAL, 6, 8))",
         "tol": 1e-12},
        {"setup": setup,
         "call": "digest(assemble_lagged_design(*SMALL, 2, 1))",
         "gold_call": "digest(_oracle_assemble_lagged_design(*SMALL, 2, 1))",
         "tol": 1e-12},
        {"setup": setup,
         "call": "digest(assemble_lagged_design(*WIDE, 2, 1))",
         "gold_call": "digest(_oracle_assemble_lagged_design(*WIDE, 2, 1))",
         "tol": 1e-12},
        {"setup": setup,
         "call": "invalid_short_record(assemble_lagged_design)",
         "gold_call": "invalid_short_record(_oracle_assemble_lagged_design)"},
    ]
