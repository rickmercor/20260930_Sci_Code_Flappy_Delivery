"""
Generate the deterministic standardized single-cell expression panel that every later step conditions on.

Single-cell expression data enter a cell-type eQTL analysis as a matrix of pseudobulk values, one row per subject and one column per cell type, obtained by aggregating each subject's cells of a given type and transforming the counts to log counts per million. Two features of that matrix survive every transformation and drive everything that follows. The first is dependence across cell types: the same subjects contribute to every column, and subject-level factors such as library size, cell composition, ancestry and batch load onto all cell types at once, so the columns of the panel are positively correlated even after normalization. The second is departure from normality: a gene is silent in a large share of subjects within a cell type, so its distribution is a spike at the detection floor followed by a continuous right-skewed body. Genes such as CDKN1A show exactly this zero-inflated shape, while housekeeping genes are closer to symmetric.




A faithful surrogate for such a panel therefore needs both properties. An equicorrelated Gaussian factor model supplies the first: each subject carries one latent factor shared by all cell types, and each cell in the panel adds independent noise, so that any two columns share a fixed fraction of their variance. Censoring supplies the second: values of the latent variable below a quantile are collapsed onto a single point and the rest are shifted, which is the Tobit mechanism and the standard idealization of a detection floor. The censoring fraction sets the height of the spike, and the surviving continuous part inherits a positive skew because only the upper tail of a symmetric distribution is retained.




Standardizing each column to mean zero and unit variance afterwards is not cosmetic. The score statistic of an eQTL test is a covariance between expression and genotype divided by the standard deviations of both, so working with standardized expression makes each cell-type statistic have unit variance under the null and makes the cross-cell-type covariance of the panel coincide with its correlation matrix. Standardization does not restore normality: it removes only the first two moments, leaving the point mass and the skew, which is precisely the discrepancy that makes a Gaussian tail approximation unreliable for this data type.

Returns
-------
np.ndarray of shape (n_subjects, n_cell_types), float: the standardised expression panel, each column of mean zero and unit variance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_expression_panel(n_subjects: int, n_cell_types: int, zero_fraction: float,
                           factor_correlation: float, seed: int) -> np.ndarray:
    """Build a standardised zero-inflated expression panel with correlated cell types.

    A latent Gaussian value is formed for every subject and cell type as
    sqrt(factor_correlation) times a subject-level factor plus
    sqrt(1 - factor_correlation) times an independent idiosyncratic term. The
    latent values are censored from below at the standard normal quantile of
    zero_fraction, entries at or below it becoming exactly zero and entries
    above it being shifted down by that quantile. Each column is then
    standardised to mean zero and unit variance using the population standard
    deviation.

    Randomness comes from numpy.random.default_rng(seed), which draws the
    subject factor of length n_subjects first and the idiosyncratic array of
    shape (n_subjects, n_cell_types) second.

    Parameters
    ----------
    n_subjects : int
        Number of subjects, n_subjects >= 2.
    n_cell_types : int
        Number of cell types, 1 <= n_cell_types <= 14.
    zero_fraction : float
        Expected fraction of censored (zero) entries, 0 < zero_fraction < 1.
    factor_correlation : float
        Fraction of latent variance carried by the shared subject factor,
        0 <= factor_correlation < 1.
    seed : int
        Seed of the random generator.

    Returns
    -------
    panel : np.ndarray
        Array of shape (n_subjects, n_cell_types), each column having mean
        zero and unit variance.
    """
    return panel  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_expression_panel(n_subjects: int, n_cell_types: int, zero_fraction: float,
                                   factor_correlation: float, seed: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import ndtri

    def _check_int(name, value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not (low <= int(value) <= high):
            raise ValueError(f"{name} must lie between {low} and {high}")
        return int(value)

    n_rows = _check_int("n_subjects", n_subjects, 2, 10 ** 6)
    n_cols = _check_int("n_cell_types", n_cell_types, 1, 14)
    seed_value = _check_int("seed", seed, -(2 ** 63), 2 ** 63 - 1)
    if not (isinstance(zero_fraction, (int, float)) and np.isfinite(zero_fraction)
            and 0.0 < float(zero_fraction) < 1.0):
        raise ValueError("zero_fraction must be a finite number strictly between 0 and 1")
    if not (isinstance(factor_correlation, (int, float)) and np.isfinite(factor_correlation)
            and 0.0 <= float(factor_correlation) < 1.0):
        raise ValueError("factor_correlation must be a finite number in the interval [0, 1)")

    rho = float(factor_correlation)
    rng = np.random.default_rng(seed_value)
    factor = rng.standard_normal(n_rows)
    idiosyncratic = rng.standard_normal((n_rows, n_cols))
    latent = np.sqrt(rho) * factor[:, None] + np.sqrt(1.0 - rho) * idiosyncratic

    # Tobit censoring at the quantile that leaves the requested share of zeros.
    censoring_point = float(ndtri(float(zero_fraction)))
    raw = np.maximum(latent - censoring_point, 0.0)

    spread = raw.std(axis=0)
    if np.any(spread <= 0.0):
        raise ValueError("a cell type is entirely censored; the panel cannot be standardised")
    panel = (raw - raw.mean(axis=0)) / spread
    return np.asarray(panel, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark panel (normal scenario) ---
        {
            "setup": """import numpy as np
n_subjects, n_cell_types = 120, 7
zero_fraction, factor_correlation, seed = 0.6, 0.4, 20260824
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(build_expression_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, seed))",
            "gold_call": "digest(_oracle_build_expression_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, seed))",
        },
        # --- Valid: independent cell types, mild censoring ---
        {
            "setup": """import numpy as np
n_subjects, n_cell_types = 40, 4
zero_fraction, factor_correlation, seed = 0.05, 0.0, 11
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(build_expression_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, seed))",
            "gold_call": "digest(_oracle_build_expression_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, seed))",
        },
        # --- Boundary: heavy censoring and strong shared factor on a single cell type ---
        {
            "setup": """import numpy as np
n_subjects, n_cell_types = 30, 1
zero_fraction, factor_correlation, seed = 0.9, 0.95, 3
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(build_expression_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, seed))",
            "gold_call": "digest(_oracle_build_expression_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, seed))",
        },
        # --- Edge: smallest admissible panel ---
        {
            "setup": """import numpy as np
n_subjects, n_cell_types = 2, 2
zero_fraction, factor_correlation, seed = 0.1, 0.5, 0
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(build_expression_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, seed))",
            "gold_call": "digest(_oracle_build_expression_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, seed))",
        },
        # --- Invalid: censoring fraction outside its open interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_expression_panel(10, 3, 1.0, 0.4, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_expression_panel(10, 3, 1.0, 0.4, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: shared-factor fraction at its excluded upper end ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_expression_panel(10, 3, 0.5, 1.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_expression_panel(10, 3, 0.5, 1.0, 0)
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
