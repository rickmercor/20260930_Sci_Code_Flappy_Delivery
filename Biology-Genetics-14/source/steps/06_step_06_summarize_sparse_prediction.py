"""
Convert retained sparse-effect draws into transferable cis predictions.

If beta^(m) is the sparse SNP-effect vector retained after sweep m, its posterior

mean is beta_mean = M^-1 sum_m beta^(m). Held-out genotype row X_test[i] is

scored as X_test[i]^T beta_mean, or X_test beta_mean for the complete panel. The dense polygenic term

used while fitting the reference samples is not added to this prediction: it is

sample specific and has no held-out value without a separate train-test kernel

construction. The resulting sparse-cis weights are therefore the transferable

quantity used for individual-level prediction and downstream TWAS-style linear

scores.

Inputs

------

beta_samples : retained sparse-effect matrix with one sweep per row

X_test : held-out genotype matrix with matching SNP columns

Returns

-------

beta_mean : componentwise posterior mean SNP weights

predictions : sparse-cis predictions for all held-out rows

first_prediction : prediction for the first held-out individual

Retained sparse-effect draws are summarized into transferable cis-genetic weights for held-out prediction. The fitted sample-specific dense background is not itself a reusable held-out coefficient.

Returns
-------
float64 vector of length p + n_test + 1, concatenating beta_mean, predictions, and first_prediction in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def summarize_sparse_prediction(
    beta_samples: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Average sparse effects and form held-out sparse-cis predictions.

    Parameters
    ----------
    beta_samples : np.ndarray
        Retained sparse-effect samples with shape (m, p).
    X_test : np.ndarray
        Held-out genotype matrix with shape (n_test, p).

    Raises
    ------
    ValueError
        If either input is empty, nonfinite, not two dimensional, or has an
        incompatible SNP dimension.

    Returns
    -------
    result : np.ndarray
        Float64 vector containing the p posterior-mean weights, all n_test
        predictions, and the first prediction as its final entry.
    """
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_summarize_sparse_prediction(
    beta_samples: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    samples = np.asarray(beta_samples, dtype=np.float64)
    test_genotypes = np.asarray(X_test, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("beta_samples must be a nonempty two-dimensional array")
    if test_genotypes.ndim != 2 or test_genotypes.shape[0] == 0:
        raise ValueError("X_test must be a nonempty two-dimensional array")
    if samples.shape[1] != test_genotypes.shape[1]:
        raise ValueError("beta_samples and X_test must have the same SNP dimension")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(test_genotypes)):
        raise ValueError("beta_samples and X_test must contain only finite values")

    beta_mean = np.mean(samples, axis=0)
    predictions = test_genotypes @ beta_mean
    return np.concatenate(
        (beta_mean, predictions, np.array([predictions[0]], dtype=np.float64))
    )

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
samples = np.array([[0.447629995520177, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.40792863962319165, 0.0, 0.0, 0.0]])
X_test = np.array([[0.6, 0.55, -0.2, 0.9, -0.4, 0.82], [-0.3, -0.28, 1.1, -0.5, 0.7, -0.47]])
""",
            "call": "summarize_sparse_prediction(samples, X_test)",
            "gold_call": "_oracle_summarize_sparse_prediction(samples, X_test)",
        },
        {
            "setup": """import numpy as np
samples = np.array([[2.0]])
X_test = np.array([[0.0]])
""",
            "call": "summarize_sparse_prediction(samples, X_test)",
            "gold_call": "_oracle_summarize_sparse_prediction(samples, X_test)",
        },
        {
            "setup": """import numpy as np
samples = np.array([[1.0, 2.0]])
X_test = np.array([[1.0, np.nan]])
def run_model():
    try:
        summarize_sparse_prediction(samples, X_test)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_summarize_sparse_prediction(samples, X_test)
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
