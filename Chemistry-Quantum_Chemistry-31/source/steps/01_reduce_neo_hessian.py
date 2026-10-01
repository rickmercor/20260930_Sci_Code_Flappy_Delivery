"""
Reduce an extended nuclear-electronic Hessian to classical coordinates.

The input ordering places all classical coordinates before the trailing
quantum-nuclear basis-centre coordinates.  Return the relaxed classical block
in the original energy and length units.  The input must be a finite real
symmetric matrix and the trailing quantum block must be positive definite.
Invalid inputs raise ``ValueError``.

Returns
-------
np.ndarray, symmetric reduced Hessian with shape (n, n), in E_h bohr^-2, where n is the number of classical coordinates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduce_neo_hessian(
    extended_hessian: 'np.ndarray',
    quantum_dim: int = 3,
) -> 'np.ndarray':
    """Return the relaxed classical-coordinate Hessian.

    Parameters
    ----------
    extended_hessian
        Extended Hessian in ``E_h bohr^-2`` with shape
        ``(n + quantum_dim, n + quantum_dim)``.
    quantum_dim
        Number of trailing quantum-coordinate rows and columns.

    Returns
    -------
    np.ndarray
        Symmetric reduced Hessian in ``E_h bohr^-2`` with shape ``(n, n)``.

    Raises
    ------
    ValueError
        If the matrix is invalid, ``quantum_dim`` is invalid, or the quantum
        block is not positive definite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduce_neo_hessian(
    extended_hessian: 'np.ndarray',
    quantum_dim: int = 3,
) -> 'np.ndarray':
    if (
        not isinstance(quantum_dim, (int, np.integer))
        or isinstance(quantum_dim, (bool, np.bool_))
        or int(quantum_dim) < 1
    ):
        raise ValueError("quantum_dim must be a positive integer")
    matrix = np.asarray(extended_hessian)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] <= int(quantum_dim)
        or not np.issubdtype(matrix.dtype, np.number)
        or not np.isrealobj(matrix)
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("extended_hessian must be a finite real square matrix")
    matrix = np.asarray(matrix, dtype=float)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("extended_hessian must be symmetric")
    split = matrix.shape[0] - int(quantum_dim)
    h_cc = matrix[:split, :split]
    h_cq = matrix[:split, split:]
    h_qc = matrix[split:, :split]
    h_qq = matrix[split:, split:]
    try:
        np.linalg.cholesky(h_qq)
        reduced = h_cc - h_cq @ np.linalg.solve(h_qq, h_qc)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the quantum-coordinate block must be positive definite") from exc
    return 0.5 * (reduced + reduced.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nhcc=np.array([[2.0,0.3],[0.3,1.4]])\nhqc=np.array([[0.2,-0.1]])\nhqq=np.array([[0.8]])\nh=np.block([[hcc,hqc.T],[hqc,hqq]])",
            "call": "reduce_neo_hessian(h.copy(),1)",
            "gold_call": "_oracle_reduce_neo_hessian(h.copy(),1)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nh=np.diag([1.2,1.7,0.9]).astype(float)",
            "call": "reduce_neo_hessian(h.copy(),1)",
            "gold_call": "_oracle_reduce_neo_hessian(h.copy(),1)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nhcc=np.array([[1.1]])\nhqc=np.array([[0.08],[0.03]])\nhqq=np.array([[0.7,0.1],[0.1,0.6]])\nh=np.block([[hcc,hqc.T],[hqc,hqq]])",
            "call": "reduce_neo_hessian(h.copy(),2)",
            "gold_call": "_oracle_reduce_neo_hessian(h.copy(),2)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nh=np.array([[1.0,0.2,0.0],[0.1,1.0,0.0],[0.0,0.0,0.5]])\ndef check(fn):\n try: fn(h.copy(),1)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(reduce_neo_hessian)",
            "gold_call": "check(_oracle_reduce_neo_hessian)",
            "tol": 0.0,
        },
    ]
