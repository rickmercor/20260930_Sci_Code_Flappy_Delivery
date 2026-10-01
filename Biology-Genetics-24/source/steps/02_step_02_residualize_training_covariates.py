"""
Covariate adjustment must be learned without validation leakage. For every resource-gene column, ordinary least squares fits an intercept and the supplied covariates on training samples only, applies those coefficients to every sample, subtracts the fitted covariate contribution, and adds back the unadjusted training mean so the expression scale is retained.

Inputs

------

predicted_expression: Float array of shape (n_samples, n_resources, n_genes).

covariates: Float array of shape (n_samples, n_covariates), without an intercept column.

training_mask: Binary array of shape (n_samples,).

Returns

-------

adjusted_expression: Float array with the same shape as predicted_expression.

Returns
-------
np.ndarray matching predicted_expression, the training-fitted covariate residuals plus training means
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def residualize_training_covariates(
    predicted_expression: "np.ndarray",
    covariates: "np.ndarray",
    training_mask: "np.ndarray",
) -> "np.ndarray":
    """Residualize predicted expression using training-fitted covariates.

    Parameters
    ----------
    predicted_expression : np.ndarray
        Predicted expression with shape (n_samples, n_resources, n_genes).
    covariates : np.ndarray
        Covariate matrix with shape (n_samples, n_covariates), excluding the
        regression intercept.
    training_mask : np.ndarray
        Binary vector selecting the samples used to fit the regression.

    Raises
    ------
    ValueError
        If dimensions or sample counts disagree, inputs are non-finite,
        `training_mask` is not binary, too few training rows are selected, or
        the training design containing an intercept is rank deficient.

    Returns
    -------
    adjusted_expression : np.ndarray
        Covariate-adjusted expression with the same shape as the input.
    """
    return adjusted_expression  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811

def _oracle_residualize_training_covariates(
    predicted_expression: "np.ndarray",
    covariates: "np.ndarray",
    training_mask: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    expression = np.asarray(predicted_expression, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    mask = np.asarray(training_mask)
    if expression.ndim != 3:
        raise ValueError("predicted_expression must be three dimensional")
    if covariates.ndim != 2:
        raise ValueError("covariates must be two dimensional")
    if covariates.shape[0] != expression.shape[0]:
        raise ValueError("expression and covariates must have the same sample count")
    if mask.shape != (expression.shape[0],):
        raise ValueError("training_mask must have one entry per sample")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("training_mask must be binary")
    if not (np.all(np.isfinite(expression)) and np.all(np.isfinite(covariates))):
        raise ValueError("expression and covariates must be finite")
    mask = mask.astype(bool)
    design = np.column_stack([np.ones(expression.shape[0]), covariates])
    if int(mask.sum()) < design.shape[1]:
        raise ValueError("training_mask selects too few rows for the design")
    if np.linalg.matrix_rank(design[mask]) != design.shape[1]:
        raise ValueError("the training covariate design must have full column rank")

    flat_expression = expression.reshape(expression.shape[0], -1)
    coefficients = np.linalg.lstsq(
        design[mask], flat_expression[mask], rcond=None
    )[0]
    fitted = design @ coefficients
    training_mean = flat_expression[mask].mean(axis=0)
    adjusted = flat_expression - fitted + training_mean
    return adjusted.reshape(expression.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
E = np.array([[[1., 3.]], [[2., 5.]], [[4., 9.]], [[7., 15.]]])
C = np.array([[0.], [1.], [2.], [3.]])
train = np.array([1, 1, 1, 0])
""",
            "call": "residualize_training_covariates(E, C, train).tolist()",
            "gold_call": "_oracle_residualize_training_covariates(E, C, train).tolist()",
        },
        {
            "setup": """import numpy as np
E = np.array([[[2.]], [[4.]]])
C = np.empty((2, 0))
train = np.array([1, 1])
""",
            "call": "residualize_training_covariates(E, C, train).tolist()",
            "gold_call": "_oracle_residualize_training_covariates(E, C, train).tolist()",
        },
        {
            "setup": """import numpy as np
E = np.ones((3, 1, 1))
C = np.array([[1.], [1.], [1.]])
train = np.array([1, 1, 1])
def run_model():
    try:
        residualize_training_covariates(E, C, train)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_residualize_training_covariates(E, C, train)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
