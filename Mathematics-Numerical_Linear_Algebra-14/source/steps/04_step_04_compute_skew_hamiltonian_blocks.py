"""
Compress the real skew-symmetric matrix into two half-size real blocks in the prescribed orthogonal basis.

A real orthogonal change of basis $Z^TAZ$ leaves $A$ skew-symmetric, so the two half-size blocks this step extracts inherit fixed symmetry classes. Projecting each computed block onto its class removes only roundoff-level defects.

Returns
-------
a finite real array blocks with shape (2, n, n), blocks[0] the symmetric part of Z2.T @ A @ Z1 and blocks[1] the skew-symmetric part of Z1.T @ A @ Z1; ValueError when the documented input contract fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_skew_hamiltonian_blocks(A: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return the stacked pair ``[H, Omega]`` of half-size blocks.

    Raises ``ValueError`` unless every one of the following holds: ``A`` is a
    two-dimensional square array of even order at least two; ``Z`` has the same
    shape as ``A``; every entry of both arrays is finite; ``A`` is skew-symmetric
    to an absolute tolerance of ``1e-10``; and ``Z`` is orthogonal to the same
    tolerance, so a scaled basis such as ``2 * I`` is rejected rather than
    projected.

    Parameters
    ----------
    A : np.ndarray
        Finite real skew-symmetric array of shape ``(2n, 2n)``.
    Z : np.ndarray
        Finite real orthogonal array with the same shape as ``A``.

    Returns
    -------
    np.ndarray
        Real array of shape ``(2, n, n)``.  With ``Z1 = Z[:, :n]`` and
        ``Z2 = Z[:, n:]``, index zero is ``H = Z2.T @ A @ Z1`` replaced by its
        symmetric part and index one is ``Omega = Z1.T @ A @ Z1`` replaced by
        its skew-symmetric part.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_skew_hamiltonian_blocks(
    A: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """Reference block compression."""
    matrix = np.asarray(A, dtype=float)
    basis = np.asarray(Z, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or matrix.shape[0] % 2
    ):
        raise ValueError("A must be an even-order square matrix")
    if basis.shape != matrix.shape:
        raise ValueError("Z must have the same shape as A")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(basis)):
        raise ValueError("A and Z must contain only finite entries")
    if not np.allclose(matrix + matrix.T, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("A must be skew-symmetric")
    if not np.allclose(
        basis.T @ basis,
        np.eye(basis.shape[0]),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("Z must be orthogonal")

    n = matrix.shape[0] // 2
    first = basis[:, :n]
    second = basis[:, n:]
    omega = first.T @ matrix @ first
    hessian = second.T @ matrix @ first
    omega = 0.5 * (omega - omega.T)
    hessian = 0.5 * (hessian + hessian.T)
    return np.stack((hessian, omega))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return canonical, rotated, scaled, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
sigma = np.diag([3.0, 0.5])
A = np.block([[np.zeros((2,2)), -sigma], [sigma, np.zeros((2,2))]])
Z = np.eye(4)
""",
            "call": "compute_skew_hamiltonian_blocks(A, Z)",
            "gold_call": "_oracle_compute_skew_hamiltonian_blocks(A, Z)",
        },
        {
            "setup": """import numpy as np
sigma = np.diag([4.0, 1.0])
S = np.block([[np.zeros((2,2)), -sigma], [sigma, np.zeros((2,2))]])
theta = 0.41
U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
Z = np.block([[U, np.zeros((2,2))], [np.zeros((2,2)), U]])
A = S.copy()
""",
            "call": "compute_skew_hamiltonian_blocks(A, Z)",
            "gold_call": "_oracle_compute_skew_hamiltonian_blocks(A, Z)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1e-6], [1e-6, 0.0]])
Z = np.eye(2)
""",
            "call": "compute_skew_hamiltonian_blocks(A, Z)",
            "gold_call": "_oracle_compute_skew_hamiltonian_blocks(A, Z)",
        },
        {
            "setup": """import numpy as np
A = np.zeros((3, 3))
Z = np.eye(3)
def run_model():
    try:
        compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1.0], [1.0, 0.0]])
A[0, 1] = np.inf
Z = np.eye(2)
def run_model():
    try:
        compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1.0], [1.0, 0.0]])
Z = np.eye(4)
def run_model():
    try:
        compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.eye(4)
Z = np.eye(4)
def run_model():
    try:
        compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1.0], [1.0, 0.0]])
Z = 2.0 * np.eye(2)
def run_model():
    try:
        compute_skew_hamiltonian_blocks(A, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_hamiltonian_blocks(A, Z)
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
