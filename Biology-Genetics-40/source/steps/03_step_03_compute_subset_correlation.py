"""
Build the null correlation matrix of the family of subset meta-analysis statistics from the table of per-subject coefficients.

The analytic multiple-testing correction for an all-subset scan treats the family of subset statistics as a discrete random field and needs its correlation structure. Two subsets that share cell types share information, and the more they overlap the more nearly their statistics coincide, so the field is far from a collection of independent tests and a Bonferroni correction over 2^C - 1 subsets would be badly conservative.




There are two routes to that correlation matrix and they must agree. The first is the Gaussian route: assume the cell-type score vector is multivariate normal with the conditional covariance implied by the expression panel and propagate that covariance through the generalised least squares pooling of each subset. The second is the design route: each subset statistic is a fixed linear form in the centred genotypes, the genotypes are independent across subjects with variance sigma_g^2 = 2 f (1 - f) under Hardy-Weinberg equilibrium, so the covariance of two subset statistics is sigma_g^2 times the inner product of their coefficient rows. Carrying the substitution through shows the two expressions are identical, because the empirical second moment matrix of the panel is exactly the conditional covariance of the score vector. The second route is the useful one: it needs no submatrix inversion, only one symmetric product of the coefficient table with itself.




Two safeguards matter numerically. The diagonal of the resulting matrix is theoretically one, since the pooling was normalized for unit null variance, but rounding leaves it a few units in the last place away, so rescaling by the square roots of the diagonal enforces the property exactly. Off-diagonal entries can likewise leave the interval from minus one to one by a rounding error, and the correction step that consumes them takes a square root of one minus their square, so they are clipped. Nested subsets that differ by one cell type reach correlations very close to one when that cell type carries little independent information, and this near degeneracy is genuine rather than an artifact; it is what makes the effective number of independent tests in the scan far smaller than the number of subsets.

Returns
-------
np.ndarray of shape (n_subsets, n_subsets), float: the symmetric unit-diagonal null correlation matrix of the subset statistics.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_subset_correlation(weights: np.ndarray, maf: float) -> np.ndarray:
    """Return the null correlation matrix of the subset meta-analysis statistics.

    Parameters
    ----------
    weights : np.ndarray
        Table of per-subject coefficients of shape (n_subsets, n_subjects),
        n_subsets >= 1 and n_subjects >= 1.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.

    Returns
    -------
    correlation : np.ndarray
        Symmetric array of shape (n_subsets, n_subsets) with unit diagonal and
        entries in the interval [-1, 1].
    """
    return correlation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_subset_correlation(weights: np.ndarray, maf: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(weights, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("weights must be a non-empty 2D array")
    if not np.all(np.isfinite(table)):
        raise ValueError("weights must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")

    frequency = float(maf)
    genotype_variance = 2.0 * frequency * (1.0 - frequency)
    covariance = genotype_variance * (table @ table.T)
    covariance = 0.5 * (covariance + covariance.T)

    spread = np.sqrt(np.diag(covariance))
    if np.any(spread <= 0.0):
        raise ValueError("a subset statistic has zero null variance")
    correlation = covariance / np.outer(spread, spread)
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: coefficient table of a small correlated panel (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(4)
weights = rng.standard_normal((31, 40)) / np.sqrt(40 * 2 * 0.02 * 0.98)
maf = 0.02
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_correlation(weights, maf))",
            "gold_call": "digest(_oracle_compute_subset_correlation(weights, maf))",
        },
        # --- Valid: two nearly identical subsets, correlation close to one ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(6)
row = rng.standard_normal(25)
weights = np.vstack([row, row + 1e-6 * rng.standard_normal(25), rng.standard_normal(25)])
maf = 0.3
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_correlation(weights, maf))",
            "gold_call": "digest(_oracle_compute_subset_correlation(weights, maf))",
        },
        # --- Boundary: exactly opposite coefficient rows, correlation minus one ---
        {
            "setup": """import numpy as np
row = np.linspace(-1.0, 1.0, 12)
weights = np.vstack([row, -row])
maf = 0.5
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_correlation(weights, maf))",
            "gold_call": "digest(_oracle_compute_subset_correlation(weights, maf))",
        },
        # --- Edge: a single subset over a single subject ---
        {
            "setup": """import numpy as np
weights = np.array([[2.5]])
maf = 0.05
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_correlation(weights, maf))",
            "gold_call": "digest(_oracle_compute_subset_correlation(weights, maf))",
        },
        # --- Invalid: a coefficient row that is identically zero ---
        {
            "setup": """import numpy as np
weights = np.array([[1.0, 0.0], [0.0, 0.0]])
def run_model():
    try:
        compute_subset_correlation(weights, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_subset_correlation(weights, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: allele frequency of zero ---
        {
            "setup": """import numpy as np
weights = np.ones((3, 4))
def run_model():
    try:
        compute_subset_correlation(weights, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_subset_correlation(weights, 0.0)
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
