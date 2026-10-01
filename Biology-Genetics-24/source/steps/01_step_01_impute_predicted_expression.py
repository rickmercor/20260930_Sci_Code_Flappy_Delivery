"""
Genetically predicted expression is the fixed-weight linear projection of allelic dosages. For individual i, resource r, and gene g, E_hat[i,r,g] = alpha[r,g] + sum_j X[i,j] * W[r,j,g], where X contains dosage values from zero to two, W contains frozen cis-SNP weights, and alpha is the resource-specific intercept.

Inputs

------

genotypes: Float array of shape (n_samples, n_snps).

weights: Float array of shape (n_resources, n_snps, n_genes).

intercepts: Float array of shape (n_resources, n_genes).

Returns

-------

predicted_expression: Float array of shape (n_samples, n_resources, n_genes).

Returns
-------
np.ndarray of shape (n_samples, n_resources, n_genes), the float64 expression predictions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def impute_predicted_expression(
    genotypes: "np.ndarray",
    weights: "np.ndarray",
    intercepts: "np.ndarray",
) -> "np.ndarray":
    """Project genotype dosages through frozen expression-weight models.

    Parameters
    ----------
    genotypes : np.ndarray
        Dosage matrix with shape (n_samples, n_snps) and entries in [0, 2].
    weights : np.ndarray
        Frozen cis-SNP weights with shape (n_resources, n_snps, n_genes).
    intercepts : np.ndarray
        Resource-gene intercepts with shape (n_resources, n_genes).

    Raises
    ------
    ValueError
        If an input has the wrong dimensionality, the SNP or resource-gene
        dimensions do not agree, an input is non-finite, or a dosage lies
        outside [0, 2].

    Returns
    -------
    predicted_expression : np.ndarray
        Predicted expression with shape (n_samples, n_resources, n_genes).
    """
    return predicted_expression  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811

def _oracle_impute_predicted_expression(
    genotypes: "np.ndarray",
    weights: "np.ndarray",
    intercepts: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    genotypes = np.asarray(genotypes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be two dimensional")
    if weights.ndim != 3:
        raise ValueError("weights must be three dimensional")
    if intercepts.ndim != 2:
        raise ValueError("intercepts must be two dimensional")
    if genotypes.shape[1] != weights.shape[1]:
        raise ValueError("genotypes and weights must have the same SNP dimension")
    if intercepts.shape != (weights.shape[0], weights.shape[2]):
        raise ValueError("intercepts must match the resource and gene dimensions")
    if not (
        np.all(np.isfinite(genotypes))
        and np.all(np.isfinite(weights))
        and np.all(np.isfinite(intercepts))
    ):
        raise ValueError("all inputs must be finite")
    if np.any((genotypes < 0.0) | (genotypes > 2.0)):
        raise ValueError("genotype dosages must lie in [0, 2]")

    return np.einsum("ns,rsg->nrg", genotypes, weights) + intercepts[None, :, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
X = np.array([[0., 1., 2.], [2., 0., 1.]])
W = np.array([[[1., 0.], [0.5, -1.], [0., 2.]], [[-1., 1.], [1., 0.], [0.5, 0.5]]])
alpha = np.array([[0.2, -0.1], [0., 0.3]])
""",
            "call": "impute_predicted_expression(X, W, alpha).tolist()",
            "gold_call": "_oracle_impute_predicted_expression(X, W, alpha).tolist()",
        },
        {
            "setup": """import numpy as np
X = np.array([[2.]])
W = np.array([[[0.25]]])
alpha = np.array([[-0.5]])
""",
            "call": "impute_predicted_expression(X, W, alpha).tolist()",
            "gold_call": "_oracle_impute_predicted_expression(X, W, alpha).tolist()",
        },
        {
            "setup": """import numpy as np
X = np.array([[0., 2.5]])
W = np.ones((1, 2, 1))
alpha = np.zeros((1, 1))
def run_model():
    try:
        impute_predicted_expression(X, W, alpha)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_impute_predicted_expression(X, W, alpha)
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
