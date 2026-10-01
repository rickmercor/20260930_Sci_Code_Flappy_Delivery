"""
Reduce two stationary covariance matrices to the ratio of the standard deviations they predict for one weighted sum of the state components.

An aggregate such as a total precursor inventory is a fixed linear combination of the state, so its variance is the corresponding quadratic form of the covariance matrix and picks up every cross-covariance between the components it sums. Comparing two models therefore reduces to the same quadratic form evaluated on each of their covariance matrices.

Returns
-------
float: the dimensionless ratio of the two predicted standard deviations, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_inventory_dispersion_ratio(reference_covariance: np.ndarray,
                                       comparison_covariance: np.ndarray,
                                       weights: np.ndarray) -> float:
    """Return the ratio of the standard deviations two covariances give an aggregate.

    Parameters
    ----------
    reference_covariance : np.ndarray
        Symmetric array of shape (n, n) whose standard deviation is the
        numerator of the ratio.
    comparison_covariance : np.ndarray
        Symmetric array of the same shape whose standard deviation is the
        denominator of the ratio.
    weights : np.ndarray
        One-dimensional array of length n giving the coefficients of the state
        components in the aggregate; at least one entry is non-zero.

    Returns
    -------
    dispersion_ratio : float
        The reference standard deviation of the aggregate divided by the
        comparison standard deviation of the same aggregate, dimensionless, as a
        native Python float.

    Raises
    ------
    ValueError
        If either covariance is not a finite two-dimensional square array, if
        the two do not have the same shape, if either is not symmetric within
        1e-9 of its largest absolute entry, if ``weights`` is not a
        one-dimensional finite array of matching length with at least one
        non-zero entry, if either quadratic form is negative, or if the
        comparison quadratic form is zero so that the ratio is undefined.
    """
    return dispersion_ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_inventory_dispersion_ratio(reference_covariance: np.ndarray,
                                               comparison_covariance: np.ndarray,
                                               weights: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    reference = np.asarray(reference_covariance, dtype=float)
    comparison = np.asarray(comparison_covariance, dtype=float)
    for name, array in (("reference_covariance", reference),
                        ("comparison_covariance", comparison)):
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
            raise ValueError(f"{name} must be a non-empty two-dimensional square array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        scale = float(np.abs(array).max())
        tolerance = 1e-9 * scale if scale > 0.0 else 1e-12
        if float(np.abs(array - array.T).max()) > tolerance:
            raise ValueError(f"{name} must be symmetric")
    if reference.shape != comparison.shape:
        raise ValueError("the two covariance matrices must have the same shape")

    coefficients = np.asarray(weights, dtype=float)
    if coefficients.ndim != 1 or coefficients.size != reference.shape[0]:
        raise ValueError("weights must be one-dimensional of the same length as the covariances")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("weights must contain only finite entries")
    if float(np.abs(coefficients).max()) == 0.0:
        raise ValueError("weights must have at least one non-zero entry")

    # The aggregate is a linear functional of the state, so its variance is the
    # quadratic form of the covariance with the same coefficients; every
    # cross-covariance between the summed components enters here.
    reference_variance = float(coefficients @ reference @ coefficients)
    comparison_variance = float(coefficients @ comparison @ coefficients)
    if reference_variance < 0.0 or comparison_variance < 0.0:
        raise ValueError("a covariance matrix gave a negative variance for the aggregate")
    if comparison_variance == 0.0:
        raise ValueError("the comparison variance vanishes, so the ratio is undefined")

    return float(np.sqrt(reference_variance / comparison_variance))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: an aggregate that sums a correlated block (normal scenario) ---
        {
            "setup": """import numpy as np
reference = np.array([[4.0, 1.2, -0.5], [1.2, 9.0, 2.0], [-0.5, 2.0, 6.0]])
comparison = np.array([[3.0, 0.4, 0.1], [0.4, 5.0, 0.6], [0.1, 0.6, 4.0]])
weights = np.array([0.0, 1.0, 1.0])
""",
            "call": "round(compute_inventory_dispersion_ratio(reference, comparison, weights), 12)",
            "gold_call": "round(_oracle_compute_inventory_dispersion_ratio(reference, comparison, weights), 12)",
        },
        # --- Valid: a single-component aggregate, which reads one diagonal entry
        #     of each matrix and ignores every correlation ---
        {
            "setup": """import numpy as np
reference = np.array([[4.0, 1.2, -0.5], [1.2, 9.0, 2.0], [-0.5, 2.0, 6.0]])
comparison = np.array([[3.0, 0.4, 0.1], [0.4, 5.0, 0.6], [0.1, 0.6, 4.0]])
weights = np.array([1.0, 0.0, 0.0])
""",
            "call": "round(compute_inventory_dispersion_ratio(reference, comparison, weights), 12)",
            "gold_call": "round(_oracle_compute_inventory_dispersion_ratio(reference, comparison, weights), 12)",
        },
        # --- Boundary: identical covariances, which must give exactly one ---
        {
            "setup": """import numpy as np
reference = np.array([[4.0, 1.2, -0.5], [1.2, 9.0, 2.0], [-0.5, 2.0, 6.0]])
comparison = reference.copy()
weights = np.array([1.0, 1.0, 1.0])
""",
            "call": "round(compute_inventory_dispersion_ratio(reference, comparison, weights), 12)",
            "gold_call": "round(_oracle_compute_inventory_dispersion_ratio(reference, comparison, weights), 12)",
        },
        # --- Edge: a signed contrast whose weights cancel most of the aggregate,
        #     so the ratio is dominated by the off-diagonal terms ---
        {
            "setup": """import numpy as np
reference = np.array([[4.0, 1.2, -0.5], [1.2, 9.0, 2.0], [-0.5, 2.0, 6.0]])
comparison = np.array([[3.0, 0.4, 0.1], [0.4, 5.0, 0.6], [0.1, 0.6, 4.0]])
weights = np.array([2.0, -1.0, 0.5])
""",
            "call": "round(compute_inventory_dispersion_ratio(reference, comparison, weights), 12)",
            "gold_call": "round(_oracle_compute_inventory_dispersion_ratio(reference, comparison, weights), 12)",
        },
        # --- Invalid: an aggregate on which the comparison model predicts no
        #     dispersion at all ---
        {
            "setup": """import numpy as np
reference = np.array([[4.0, 0.0], [0.0, 9.0]])
comparison = np.array([[3.0, 0.0], [0.0, 0.0]])
weights = np.array([0.0, 1.0])
def run_model():
    try:
        compute_inventory_dispersion_ratio(reference, comparison, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_inventory_dispersion_ratio(reference, comparison, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: weights of the wrong length ---
        {
            "setup": """import numpy as np
reference = np.array([[4.0, 0.0], [0.0, 9.0]])
comparison = np.array([[3.0, 0.0], [0.0, 5.0]])
weights = np.array([0.0, 1.0, 1.0])
def run_model():
    try:
        compute_inventory_dispersion_ratio(reference, comparison, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_inventory_dispersion_ratio(reference, comparison, weights)
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
