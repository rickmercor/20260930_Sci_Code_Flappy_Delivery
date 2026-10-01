"""
Return the Gauss-Christoffel nodes and weights encoded by a symmetric Jacobi matrix and the zeroth moment, with the nodes in ascending order and the weights in the matching order.

The nodes are the eigenvalues of the Jacobi matrix and the weights follow from the first components of the corresponding eigenvectors scaled by the zeroth moment; together they define the discrete spectral measure of the approximant.

Returns
-------
numpy.ndarray, Array of shape (2, n): row 0 the ascending nodes, row 1 the weights (float64).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quadrature_nodes_weights(jacobi: "np.ndarray", mu0: float) -> "np.ndarray":
    """Return the Gauss-Christoffel nodes and weights encoded by a symmetric Jacobi matrix and the zeroth moment, with the nodes in ascending order and the weights in the matching order.

    Parameters
    ----------
    jacobi : numpy.ndarray
        Finite symmetric square matrix.
    mu0 : float
        Positive zeroth moment.

    Returns
    -------
    nodes_weights : numpy.ndarray
        Array of shape (2, n): row 0 the ascending nodes, row 1 the weights (float64).

    Raises
    ------
    ValueError
        If jacobi is not a finite symmetric square matrix, or mu0 is not positive and finite.
    """
    return nodes_weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_quadrature_nodes_weights(jacobi: "np.ndarray", mu0: float) -> "np.ndarray":
    """Nodes (eigenvalues) and weights mu0 * v[0]^2 of the Jacobi matrix, ascending by node."""
    J = np.asarray(jacobi, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] < 1 or not np.all(np.isfinite(J)):
        raise ValueError("jacobi must be a finite square matrix")
    if not np.allclose(J, J.T, rtol=0.0, atol=1e-12):
        raise ValueError("jacobi must be symmetric")
    if not (np.isfinite(mu0) and mu0 > 0.0):
        raise ValueError("mu0 must be positive and finite")
    nodes, vecs = np.linalg.eigh(J)
    weights = float(mu0) * vecs[0, :] ** 2
    order = np.argsort(nodes)
    return np.vstack([nodes[order], weights[order]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\njacobi = np.array([[0.0, 0.5326, 0.0], [0.5326, 0.0, 0.1732], [0.0, 0.1732, 0.0]])\nmu0 = 1.0\n",
            "call": "quadrature_nodes_weights(jacobi, mu0)",
            "gold_call": "_oracle_quadrature_nodes_weights(jacobi, mu0)",
        },
        {
            "setup": "import numpy as np\njacobi = np.array([[0.7]])\nmu0 = 1.0\n",
            "call": "quadrature_nodes_weights(jacobi, mu0)",
            "gold_call": "_oracle_quadrature_nodes_weights(jacobi, mu0)",
        },
        {
            "setup": "import numpy as np\njacobi = np.array([[0.3, 0.4], [0.4, -0.1]])\nmu0 = 2.0\n",
            "call": "quadrature_nodes_weights(jacobi, mu0)",
            "gold_call": "_oracle_quadrature_nodes_weights(jacobi, mu0)",
        },
        {
            "setup": "import numpy as np\njacobi = np.array([[0.0, 1.0], [0.0, 0.0]])\nmu0 = 1.0\ndef run_model():\n    try:\n        quadrature_nodes_weights(jacobi, mu0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_quadrature_nodes_weights(jacobi, mu0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
