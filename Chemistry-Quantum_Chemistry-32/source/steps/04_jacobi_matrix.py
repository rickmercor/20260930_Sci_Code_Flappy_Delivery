"""
Build the symmetric tridiagonal Jacobi matrix of the rank-point Gauss-Christoffel quadrature rule from the moment sequence, following the source's factorisation route through the Hankel moment matrix.

The Jacobi matrix is the finite Lanczos representation of the Liouvillian on the Krylov space spanned by the moment sequence; its eigen-decomposition yields the quadrature nodes and weights that reproduce the first 2N moments exactly.

Returns
-------
numpy.ndarray, Symmetric array of shape (rank, rank) (float64).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def jacobi_matrix(moments: "np.ndarray", rank: int) -> "np.ndarray":
    """Build the symmetric tridiagonal Jacobi matrix of the rank-point Gauss-Christoffel quadrature rule from the moment sequence, following the source's factorisation route through the Hankel moment matrix.

    Parameters
    ----------
    moments : numpy.ndarray
        Finite 1-D array holding at least 2*rank moments.
    rank : int
        Positive number of quadrature points.

    Returns
    -------
    jacobi : numpy.ndarray
        Symmetric array of shape (rank, rank) (float64).

    Raises
    ------
    ValueError
        If rank is not a positive integer, moments is too short or not finite, or the Hankel matrix at this rank is not positive definite.
    """
    return jacobi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_jacobi_matrix(moments: "np.ndarray", rank: int) -> "np.ndarray":
    """Tridiagonal Jacobi matrix of the rank-point Gauss-Christoffel rule via the LDL^T route.

    With M = L L^T the Cholesky factor of the rank x rank Hankel matrix and M' the shifted
    Hankel [mu_{i+j+1}], the Jacobi matrix is J = L^{-1} M' L^{-T}, symmetrised.
    """
    moments = np.asarray(moments, dtype=np.float64)
    if int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    rank = int(rank)
    if moments.ndim != 1 or moments.size < 2 * rank or not np.all(np.isfinite(moments)):
        raise ValueError("moments must be a finite 1-D array with at least 2*rank entries")
    M = np.array([[moments[i + j] for j in range(rank)] for i in range(rank)])
    Ms = np.array([[moments[i + j + 1] for j in range(rank)] for i in range(rank)])
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        raise ValueError("the Hankel matrix is not positive definite at this rank")
    Li = np.linalg.inv(L)
    J = Li @ Ms @ Li.T
    return 0.5 * (J + J.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nmoments = np.array([1.0, 0.0, 0.2836, 0.0, 0.0889, 0.0])\nrank = 3\n",
            "call": "jacobi_matrix(moments, rank)",
            "gold_call": "_oracle_jacobi_matrix(moments, rank)",
        },
        {
            "setup": "import numpy as np\nmoments = np.array([1.0, 0.0, 0.25, 0.0])\nrank = 2\n",
            "call": "jacobi_matrix(moments, rank)",
            "gold_call": "_oracle_jacobi_matrix(moments, rank)",
        },
        {
            "setup": "import numpy as np\nmoments = np.array([1.0, 0.3, 0.5, 0.6, 1.1, 1.8])\nrank = 3\n",
            "call": "jacobi_matrix(moments, rank)",
            "gold_call": "_oracle_jacobi_matrix(moments, rank)",
        },
        {
            "setup": "import numpy as np\nmoments = np.array([1.0, 0.0, 0.25, 0.0, 0.0625, 0.0])\nrank = 3\ndef run_model():\n    try:\n        jacobi_matrix(moments, rank)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_jacobi_matrix(moments, rank)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
