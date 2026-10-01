"""
Convert the complex invariant-subspace basis into the real orthogonal basis used to reduce the original matrix.

The step maps a complex $(2n,n)$ block with orthonormal columns to a real $(2n,2n)$ array, taking one half-width block from the real parts and one from the imaginary parts. The factor $\sqrt{2}$ is what makes the result orthogonal.

Returns
-------
a finite real orthogonal array Z = sqrt(2) * [Re(V_tilde) | -Im(V_tilde)]; ValueError when the documented input contract fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_real_symplectic_basis(V_tilde: np.ndarray) -> np.ndarray:
    """Return ``sqrt(2) [Re(V_tilde) | -Im(V_tilde)]``.

    Raises ``ValueError`` unless every one of the following holds: ``V_tilde`` is
    a two-dimensional array of shape ``(2n, n)`` with ``n`` at least one; every
    real and imaginary part is finite; and the columns of ``V_tilde`` are
    orthonormal to an absolute tolerance of ``1e-10``, so a rank-deficient block
    such as ``numpy.ones((4, 3))`` is rejected rather than realified.

    Parameters
    ----------
    V_tilde : np.ndarray
        Finite complex array of shape ``(2n, n)`` with orthonormal columns.

    Returns
    -------
    np.ndarray
        Real square array ``Z`` of shape ``(2n, 2n)``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_real_symplectic_basis(V_tilde: np.ndarray) -> np.ndarray:
    """Reference complex-to-real basis map."""
    basis = np.asarray(V_tilde, dtype=complex)
    if basis.ndim != 2 or basis.shape[0] != 2 * basis.shape[1] or basis.shape[1] < 1:
        raise ValueError("V_tilde must have shape (2n, n)")
    if not np.all(np.isfinite(basis.real)) or not np.all(np.isfinite(basis.imag)):
        raise ValueError("V_tilde must contain only finite entries")
    identity = np.eye(basis.shape[1])
    if not np.allclose(basis.conj().T @ basis, identity, atol=1e-10, rtol=0.0):
        raise ValueError("V_tilde must have orthonormal columns")
    return np.sqrt(2.0) * np.concatenate((basis.real, -basis.imag), axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return canonical, phase-rotated, mixed, and invalid bases."""
    return [
        {
            "setup": """import numpy as np
V_tilde = np.array([[1.0], [-1.0j]]) / np.sqrt(2.0)
""",
            "call": "build_real_symplectic_basis(V_tilde)",
            "gold_call": "_oracle_build_real_symplectic_basis(V_tilde)",
        },
        {
            "setup": """import numpy as np
phase = np.exp(0.37j)
V_tilde = phase * np.array([[1.0], [-1.0j]]) / np.sqrt(2.0)
""",
            "call": "build_real_symplectic_basis(V_tilde)",
            "gold_call": "_oracle_build_real_symplectic_basis(V_tilde)",
        },
        {
            "setup": """import numpy as np
U = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
V_tilde = np.vstack((U, -1.0j * U)) / np.sqrt(2.0)
""",
            "call": "build_real_symplectic_basis(V_tilde)",
            "gold_call": "_oracle_build_real_symplectic_basis(V_tilde)",
        },
        {
            "setup": """import numpy as np
V_tilde = np.ones((4, 2), dtype=complex)
def run_model():
    try:
        build_real_symplectic_basis(V_tilde)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_real_symplectic_basis(V_tilde)
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
V_tilde = np.ones((3, 2), dtype=complex)
def run_model():
    try:
        build_real_symplectic_basis(V_tilde)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_real_symplectic_basis(V_tilde)
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
V_tilde = np.array([[1.0, 0.0], [0.0, 1.0], [np.inf, 0.0], [0.0, 0.0]], dtype=complex)
def run_model():
    try:
        build_real_symplectic_basis(V_tilde)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_real_symplectic_basis(V_tilde)
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
V_tilde = np.ones((4, 3), dtype=complex)
def run_model():
    try:
        build_real_symplectic_basis(V_tilde)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_real_symplectic_basis(V_tilde)
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
