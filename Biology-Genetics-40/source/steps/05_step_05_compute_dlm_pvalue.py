"""
Evaluate the analytic discrete local maxima significance of the all-subset maximum for a Gaussian family of subset statistics.

The analytic significance attached to an all-subset scan comes from the discrete local maxima approximation for the excursion probability of a discrete random field. Its logic is a decomposition of the excursion event by the site at which the field attains its maximum. If the field exceeds a high threshold at all, it does so at some site, and at that site it must dominate its own neighbours; summing over sites the probability of the joint event that the field takes a value beyond the threshold there and no neighbour exceeds that value produces an expression that becomes exact as the threshold grows, because the probability of two separated excursions is of smaller order.




Written out for a unit-variance Gaussian family scanned in absolute value, the contribution of one site is an integral over the level attained there. Conditioning the field on the value at that site makes each neighbour a univariate Gaussian whose mean is the correlation with the site times the level and whose variance is one minus the square of that correlation, so the probability that a neighbour stays inside the interval bounded by the level is a difference of two standard normal distribution functions. The neighbours are treated as conditionally independent given the site, which is the working assumption of the approximation, so their probabilities multiply. The integrand is that product weighted by the standard normal density of the level, the factor of two accounts for the two signs of a two-sided scan, and the level runs from the observed threshold upward.




Numerically, the integral is over a semi-infinite interval, but the standard normal density suppresses the integrand so rapidly at a threshold in the tail that truncating a fixed distance above the threshold introduces an error many orders of magnitude below the value being computed. Gauss-Legendre nodes on the truncated interval converge quickly because the integrand is smooth there. The product over neighbours is accumulated in logarithms: each factor is a probability below one, there are as many factors as cell types, and at large thresholds several of them are extremely close to one while others are small, so multiplying them directly loses relative accuracy that the logarithm preserves. Inadmissible neighbours, marked in the neighbour table, contribute no factor at all.




The result is a lower bound in theory, since a local maximum is necessary but not sufficient for the global maximum, but the deficit is negligible in the far tail. What is not negligible is the premise: every conditional probability above assumes the field is Gaussian. If the underlying statistics are sums of a small number of discrete, heavily unbalanced terms, the premise fails in exactly the region where the formula is being used.

Returns
-------
float: the analytic discrete local maxima significance of the all-subset maximum, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_dlm_pvalue(correlation: np.ndarray, neighbours: np.ndarray, threshold: float,
                       n_nodes: int = 400, tail_width: float = 8.0) -> float:
    """Return the discrete local maxima significance of the all-subset maximum.

    The level integral of each subset is truncated at threshold + tail_width
    and evaluated with n_nodes Gauss-Legendre nodes on that interval.
    Correlations are clipped away from plus or minus one by 1e-12 before the
    conditional standard deviation is formed.

    Parameters
    ----------
    correlation : np.ndarray
        Symmetric unit-diagonal correlation matrix of the subset statistics,
        of shape (n_subsets, n_subsets).
    neighbours : np.ndarray
        Integer neighbour table of shape (n_subsets, n_cell_types); entries
        below zero mark inadmissible toggles.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.
    n_nodes : int
        Number of Gauss-Legendre nodes, n_nodes >= 2.
    tail_width : float
        Width of the truncated level interval, tail_width > 0.

    Returns
    -------
    pvalue : float
        Analytic significance as a native Python float.
    """
    return pvalue  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_dlm_pvalue(correlation: np.ndarray, neighbours: np.ndarray, threshold: float,
                               n_nodes: int = 400, tail_width: float = 8.0) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import ndtr

    matrix = np.asarray(correlation, dtype=float)
    table = np.asarray(neighbours)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("correlation must be a square 2D array of order at least 1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation must be finite")
    if table.ndim != 2 or table.shape[0] != matrix.shape[0] or table.shape[1] < 1:
        raise ValueError("neighbours must be 2D with one row per subset")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("neighbours must hold integers")
    if int(table.max(initial=-1)) >= matrix.shape[0]:
        raise ValueError("neighbours refers to a subset outside the correlation matrix")
    if not (isinstance(threshold, (int, float)) and np.isfinite(threshold)
            and float(threshold) > 0.0):
        raise ValueError("threshold must be a finite number > 0")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer >= 2")
    if not (isinstance(tail_width, (int, float)) and np.isfinite(tail_width)
            and float(tail_width) > 0.0):
        raise ValueError("tail_width must be a finite number > 0")

    level_floor = float(threshold)
    half_width = 0.5 * float(tail_width)
    nodes, quad_weights = np.polynomial.legendre.leggauss(int(n_nodes))
    levels = half_width * (nodes + 1.0) + level_floor
    density = np.exp(-0.5 * levels * levels) / np.sqrt(2.0 * np.pi)

    total = 0.0
    for site in range(matrix.shape[0]):
        log_product = np.zeros_like(levels)
        for column in range(table.shape[1]):
            partner = int(table[site, column])
            if partner < 0:
                continue
            rho = float(np.clip(matrix[site, partner], -1.0 + 1e-12, 1.0 - 1e-12))
            conditional_sd = np.sqrt(1.0 - rho * rho)
            inside = (ndtr((1.0 - rho) * levels / conditional_sd)
                      - ndtr(-(1.0 + rho) * levels / conditional_sd))
            log_product += np.log(np.clip(inside, 1e-300, None))
        total += half_width * float(np.sum(quad_weights * 2.0 * np.exp(log_product) * density))
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: five cell types at a tail threshold (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(12)
w = rng.standard_normal((31, 50))
cov = w @ w.T
d = np.sqrt(np.diag(cov))
correlation = np.clip(cov / np.outer(d, d), -1.0, 1.0)
neighbours = np.full((31, 5), -1, dtype=int)
for m in range(1, 32):
    for c in range(5):
        t = m ^ (1 << c)
        neighbours[m - 1, c] = t - 1 if t >= 1 else -1
threshold = 7.0
""",
            "call": "compute_dlm_pvalue(correlation, neighbours, threshold)",
            "gold_call": "_oracle_compute_dlm_pvalue(correlation, neighbours, threshold)",
        },
        # --- Valid: the same field at a moderate threshold ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(12)
w = rng.standard_normal((31, 50))
cov = w @ w.T
d = np.sqrt(np.diag(cov))
correlation = np.clip(cov / np.outer(d, d), -1.0, 1.0)
neighbours = np.full((31, 5), -1, dtype=int)
for m in range(1, 32):
    for c in range(5):
        t = m ^ (1 << c)
        neighbours[m - 1, c] = t - 1 if t >= 1 else -1
threshold = 3.0
""",
            "call": "compute_dlm_pvalue(correlation, neighbours, threshold, 200, 6.0)",
            "gold_call": "_oracle_compute_dlm_pvalue(correlation, neighbours, threshold, 200, 6.0)",
        },
        # --- Boundary: nearly perfectly correlated neighbours ---
        {
            "setup": """import numpy as np
correlation = np.array([[1.0, 1.0 - 1e-13, 0.5],
                        [1.0 - 1e-13, 1.0, 0.5],
                        [0.5, 0.5, 1.0]])
neighbours = np.array([[1, 2], [0, -1], [0, -1]], dtype=int)
threshold = 5.0
""",
            "call": "compute_dlm_pvalue(correlation, neighbours, threshold)",
            "gold_call": "_oracle_compute_dlm_pvalue(correlation, neighbours, threshold)",
        },
        # --- Edge: one subset with no admissible neighbour, a plain two-sided tail ---
        {
            "setup": """import numpy as np
correlation = np.array([[1.0]])
neighbours = np.array([[-1]], dtype=int)
threshold = 4.0
""",
            "call": "compute_dlm_pvalue(correlation, neighbours, threshold)",
            "gold_call": "_oracle_compute_dlm_pvalue(correlation, neighbours, threshold)",
        },
        # --- Invalid: threshold at zero ---
        {
            "setup": """import numpy as np
correlation = np.eye(3)
neighbours = np.array([[1], [0], [-1]], dtype=int)
def run_model():
    try:
        compute_dlm_pvalue(correlation, neighbours, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_dlm_pvalue(correlation, neighbours, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: neighbour index outside the field ---
        {
            "setup": """import numpy as np
correlation = np.eye(2)
neighbours = np.array([[5], [0]], dtype=int)
def run_model():
    try:
        compute_dlm_pvalue(correlation, neighbours, 4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_dlm_pvalue(correlation, neighbours, 4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
