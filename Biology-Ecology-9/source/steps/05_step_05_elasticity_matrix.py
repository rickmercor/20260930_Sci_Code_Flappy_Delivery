"""
Given a nonnegative irreducible projection matrix, return its elasticity matrix, the matrix whose entry in row i and column j is v_i m_ij w_j divided by the product of the Perron root and the inner product of v and w, where m_ij is the entry of the matrix and v and w are its reproductive values and stable distribution. Obtain the three objects from the Perron triplet, so that the result is correct for imprimitive matrices. Entries of the matrix that are zero have zero elasticity.

The elasticity of the asymptotic growth rate to an entry of a projection matrix is the proportional change in growth rate per proportional change in that entry. For a simple dominant eigenvalue the sensitivity of the growth rate to entry (i, j) is the product of the i-th reproductive value and the j-th stable-distribution component divided by the inner product of the two vectors, so the elasticity is that sensitivity multiplied by the entry and divided by the growth rate. Elasticities are dimensionless, they are unchanged by any rescaling of the eigenvectors, and they sum to one over the whole matrix, which is why comparative demography uses them to apportion the growth rate among the vital rates.

Two sums of the elasticity matrix are fixed by the eigenvector equations. The sum of row i is the product of the i-th reproductive value and the i-th stable component, and so is the sum of column i, both after scaling the vectors to unit inner product. For a Leslie matrix this ties the elasticities of the oldest classes to their survivorship and fertility and makes the whole matrix a record of where reproductive value flows.

Returns
-------
np.ndarray of shape (N, N), the elasticity matrix, nonnegative and summing to one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def elasticity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Elasticities of the Perron root to every entry of a projection matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Nonnegative irreducible projection matrix, shape (N, N), with a positive Perron root.

    Returns
    -------
    np.ndarray
        The elasticity matrix, shape (N, N).

    Raises
    ------
    ValueError
        When the matrix fails the checks of the Perron triplet or its Perron root is zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_elasticity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    triplet = _oracle_perron_triplet(matrix)  # noqa: F821
    root = triplet["growth_rate"]
    if root <= 0.0:
        raise ValueError("the Perron root must be positive for elasticities to exist")
    a = np.asarray(matrix, dtype=float)
    v = triplet["reproductive_values"]
    w = triplet["stable_distribution"]
    return np.outer(v, w) * a / (root * float(v @ w))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef leslie(fertility, survival):\n    matrix = np.zeros((len(fertility), len(fertility)))\n    matrix[0, :] = fertility\n    for index, probability in enumerate(survival):\n        matrix[index + 1, index] = probability\n    return matrix\n\nF = np.array([0.0, 0.4, 1.0, 1.4, 1.6, 1.8, 1.6])\nP = np.array([0.60, 0.72, 0.78, 0.76, 0.70, 0.50])\n\ndef resolved(fertility, survival, subdivisions):\n    classes = len(fertility)\n    matrix = np.zeros((classes * subdivisions, classes * subdivisions))\n    for age_class in range(1, classes + 1):\n        matrix[0, age_class * subdivisions - 1] = fertility[age_class - 1]\n    for index in range(1, classes * subdivisions):\n        matrix[index, index - 1] = survival[index // subdivisions - 1] if index % subdivisions == 0 else 1.0\n    return matrix\n"
    return [
        {
            "setup": setup + "",
            "call": "project(elasticity_matrix(leslie(F, P)))",
            "gold_call": "project(_oracle_elasticity_matrix(leslie(F, P)))",
        },
        {
            "setup": setup + "",
            "call": "project(elasticity_matrix(np.linalg.matrix_power(resolved(F, P, 3), 7)))",
            "gold_call": "project(_oracle_elasticity_matrix(np.linalg.matrix_power(resolved(F, P, 3), 7)))",
        },
        {
            "setup": setup + "",
            "call": "project(elasticity_matrix(np.array([[2.3]])))",
            "gold_call": "project(_oracle_elasticity_matrix(np.array([[2.3]])))",
        },
        {
            "setup": setup + "BAD = leslie([0.0, 1.0, 0.0], [0.5, 0.5])\n",
            "call": "expect_value_error(elasticity_matrix, BAD)",
            "gold_call": "expect_value_error(_oracle_elasticity_matrix, BAD)",
        },
    ]
