"""
Express every subset meta-analysis statistic as a fixed linear form in the centred genotypes and return the table of per-subject coefficients.

An all-subset meta-analysis of one variant across C cell types forms, for each of the 2^C - 1 nonempty subsets, a pooled statistic and takes the largest absolute value. Conditional on the observed expression panel the only random object under the global null is the genotype vector, so the whole family of subset statistics is a family of linear functionals of that one vector, and the analysis becomes tractable the moment those functionals are written down explicitly.




The cell-type score statistic is the standardised covariance between expression and centred genotype, z_c = sum_i y_ic (g_i - 2f) / (sqrt(N) sigma_g), with allele frequency f and sigma_g^2 = 2 f (1 - f) the genotype variance under Hardy-Weinberg equilibrium. Because all cell types are measured on the same subjects, the score vector is not independent across cell types: conditional on the panel its covariance is exactly the sample cross-cell-type correlation matrix of the standardised expression, so pooling a subset with equal weights would double count the shared information. The efficient fixed-effect pooling of a correlated score vector with a common effect size is the generalised least squares combination, which weights the subvector by the inverse of its own principal submatrix contracted against the vector of ones and normalises so that the pooled statistic has unit variance under the null.




Substituting the definition of the score statistics into that combination collapses the two stages into one: each subset statistic becomes a single weighted sum of centred genotypes, and the weight attached to a subject is determined entirely by the expression panel, the allele frequency and the subset. Assembling those weights once, as a table with one row per subset and one column per subject, is what makes every downstream quantity cheap. The correlation structure of the subset family, the cumulant generating function under a genotype null, the tilted sampling distribution and the evaluation of all subset statistics for a simulated genotype vector are then a matrix product or a row lookup away, and no submatrix ever has to be inverted again.




The subsets must be enumerated in a fixed order for the table to have meaning. The natural choice is the binary one: subset index m - 1 collects the cell types whose bit is set in the integer m, for m from 1 to 2^C - 1, so that adding or removing a cell type is a bit flip and the neighbour structure needed later is immediate.

Returns
-------
np.ndarray of shape (2**n_cell_types - 1, n_subjects), float: the per-subject coefficient of every subset statistic, subsets in binary order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_subset_weights(panel: np.ndarray, maf: float) -> np.ndarray:
    """Return the per-subject coefficients of every subset meta-analysis statistic.

    Subsets are enumerated in binary order: row m - 1 of the returned table
    corresponds to the subset containing cell type c whenever bit c of the
    integer m is set, for m = 1, ..., 2**n_cell_types - 1.

    Parameters
    ----------
    panel : np.ndarray
        Standardised expression panel of shape (n_subjects, n_cell_types),
        n_subjects >= 2 and 1 <= n_cell_types <= 14.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.

    Returns
    -------
    weights : np.ndarray
        Array of shape (2**n_cell_types - 1, n_subjects) whose row A holds the
        coefficient of every subject in the subset statistic of A.
    """
    return weights  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_subset_weights(panel: np.ndarray, maf: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    matrix = np.asarray(panel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("panel must be a 2D array with at least 2 rows and 1 column")
    if matrix.shape[1] > 14:
        raise ValueError("panel must have at most 14 columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("panel must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")

    n_subjects, n_cell_types = matrix.shape
    frequency = float(maf)
    genotype_sd = np.sqrt(2.0 * frequency * (1.0 - frequency))

    # Conditional on the panel the score vector has this covariance.
    cross = matrix.T @ matrix / n_subjects

    weights = np.empty((2 ** n_cell_types - 1, n_subjects), dtype=float)
    for mask in range(1, 2 ** n_cell_types):
        members = np.array([c for c in range(n_cell_types) if (mask >> c) & 1], dtype=int)
        ones = np.ones(members.size, dtype=float)
        block = cross[np.ix_(members, members)]
        try:
            direction = np.linalg.solve(block, ones)
        except np.linalg.LinAlgError as exc:
            raise ValueError("a cell-type submatrix of the panel is singular") from exc
        scale = float(ones @ direction)
        if not (np.isfinite(scale) and scale > 0.0):
            raise ValueError("a cell-type submatrix of the panel is not positive definite")
        weights[mask - 1] = (matrix[:, members] @ direction) / (
            np.sqrt(n_subjects) * genotype_sd * np.sqrt(scale))
    return weights

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark panel and rare variant (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
factor = rng.standard_normal(60)
latent = 0.6 * factor[:, None] + 0.8 * rng.standard_normal((60, 5))
raw = np.maximum(latent - 0.25, 0.0)
panel = (raw - raw.mean(axis=0)) / raw.std(axis=0)
maf = 0.02
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_weights(panel, maf))",
            "gold_call": "digest(_oracle_compute_subset_weights(panel, maf))",
        },
        # --- Valid: same panel at a common allele frequency ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
factor = rng.standard_normal(60)
latent = 0.6 * factor[:, None] + 0.8 * rng.standard_normal((60, 5))
raw = np.maximum(latent - 0.25, 0.0)
panel = (raw - raw.mean(axis=0)) / raw.std(axis=0)
maf = 0.5
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_weights(panel, maf))",
            "gold_call": "digest(_oracle_compute_subset_weights(panel, maf))",
        },
        # --- Boundary: nearly collinear cell types, where pooling weights blow up ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(9)
base = rng.standard_normal(40)
panel = np.column_stack([base, base + 1e-4 * rng.standard_normal(40),
                         rng.standard_normal(40)])
panel = (panel - panel.mean(axis=0)) / panel.std(axis=0)
maf = 0.05
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_weights(panel, maf))",
            "gold_call": "digest(_oracle_compute_subset_weights(panel, maf))",
        },
        # --- Edge: a single cell type, where the pooling is the identity ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1)
column = rng.standard_normal((25, 1))
panel = (column - column.mean(axis=0)) / column.std(axis=0)
maf = 0.1
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_weights(panel, maf))",
            "gold_call": "digest(_oracle_compute_subset_weights(panel, maf))",
        },
        # --- Invalid: allele frequency above one half ---
        {
            "setup": """import numpy as np
panel = np.eye(6, 3)
def run_model():
    try:
        compute_subset_weights(panel, 0.7)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_subset_weights(panel, 0.7)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: one-dimensional panel ---
        {
            "setup": """import numpy as np
panel = np.arange(10.0)
def run_model():
    try:
        compute_subset_weights(panel, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_subset_weights(panel, 0.2)
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
