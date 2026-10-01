"""
Recover the nonnegative spectral values and real orthogonal spectral basis from the half-size eigenvectors.

A real skew-symmetric matrix has a real spectral form whose basis is orthogonal and whose diagonal is nonnegative. This step assembles that basis and that diagonal from the real basis and the half-size eigensystem, carrying the sign of each eigenvalue in the basis rather than in the diagonal.

Returns
-------
one real (2n + 1, 2n) array with the signed-recovery Q in rows :2n and sigma = abs(sigma_tilde) in row 2n, columns :n; ValueError when the documented input contract fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recover_real_spectral_decomposition(
    Z: np.ndarray,
    eigensystem: np.ndarray,
) -> np.ndarray:
    """Return a packed real spectral basis and ordered nonnegative values.

    Raises ``ValueError`` unless every one of the following holds: ``Z`` is a
    two-dimensional square array of even order ``2n`` at least two;
    ``eigensystem`` has shape ``(n + 1, n)``; the packed values in row zero are
    real to an absolute tolerance of ``1e-12``; every entry of ``Z`` and of the
    packed eigensystem is finite; ``Z`` is orthogonal to an absolute tolerance of
    ``1e-10``; and rows ``1:`` form a unitary ``U`` to the same tolerance, so a
    scaled eigenvector matrix is rejected rather than recovered.

    Parameters
    ----------
    Z : np.ndarray
        Finite real orthogonal array of shape ``(2n, 2n)``.
    eigensystem : np.ndarray
        Complex array of shape ``(n + 1, n)`` with signed eigenvalues in row
        zero and a unitary eigenvector matrix in the remaining rows.

    Returns
    -------
    np.ndarray
        Real array of shape ``(2n + 1, 2n)``.  Write ``Z1 = Z[:, :n]``,
        ``Z2 = Z[:, n:]``, ``U_r`` and ``U_i`` for the real and imaginary parts
        of the eigenvectors, and ``s`` for the elementwise sign of the packed
        eigenvalues with ``sign(0) = 1``.  The first ``n`` columns of ``Q`` are
        ``Z1 @ U_i + Z2 @ U_r`` and the last ``n`` are
        ``(-Z1 @ U_r + Z2 @ U_i) * s``.  Rows ``:2n`` store ``Q``; row ``2n``
        stores ``abs(sigma_tilde)`` in its first ``n`` entries and zeros in the
        remaining entries.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_recover_real_spectral_decomposition(
    Z: np.ndarray,
    eigensystem: np.ndarray,
) -> np.ndarray:
    """Reference signed real recovery."""
    basis = np.asarray(Z, dtype=float)
    packed = np.asarray(eigensystem, dtype=complex)
    if (
        basis.ndim != 2
        or basis.shape[0] != basis.shape[1]
        or basis.shape[0] < 2
        or basis.shape[0] % 2
    ):
        raise ValueError("Z must be an even-order square matrix")
    n = basis.shape[0] // 2
    if packed.shape != (n + 1, n):
        raise ValueError("eigensystem must have shape (n + 1, n)")
    if not np.allclose(packed[0].imag, 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("the packed eigenvalues must be real")
    values = packed[0].real
    vectors = packed[1:]
    if (
        not np.all(np.isfinite(basis))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(vectors.real))
        or not np.all(np.isfinite(vectors.imag))
    ):
        raise ValueError("all inputs must contain only finite entries")
    if not np.allclose(basis.T @ basis, np.eye(2 * n), atol=1e-10, rtol=0.0):
        raise ValueError("Z must be orthogonal")
    if not np.allclose(vectors.conj().T @ vectors, np.eye(n), atol=1e-10, rtol=0.0):
        raise ValueError("U must be unitary")

    first = basis[:, :n]
    second = basis[:, n:]
    real = vectors.real
    imaginary = vectors.imag
    signs = np.where(values < 0.0, -1.0, 1.0)
    q_first = first @ imaginary + second @ real
    q_second = (-first @ real + second @ imaginary) * signs
    spectral_basis = np.concatenate((q_first, q_second), axis=1)
    result = np.zeros((2 * n + 1, 2 * n), dtype=float)
    result[: 2 * n] = spectral_basis
    result[2 * n, :n] = np.abs(values)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return positive, negative, mixed-unitary, signed-with-zero, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
Z = np.eye(4)
eigensystem = np.vstack((np.array([3.0, 0.5], dtype=complex), np.eye(2, dtype=complex)))
""",
            "call": "recover_real_spectral_decomposition(Z, eigensystem)",
            "gold_call": "_oracle_recover_real_spectral_decomposition(Z, eigensystem)",
        },
        {
            "setup": """import numpy as np
Z = np.eye(2)
eigensystem = np.array([[-0.2], [1.0]], dtype=complex)
""",
            "call": "recover_real_spectral_decomposition(Z, eigensystem)",
            "gold_call": "_oracle_recover_real_spectral_decomposition(Z, eigensystem)",
        },
        {
            "setup": """import numpy as np
Z = np.eye(4)
U = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
eigensystem = np.vstack((np.array([2.0, -0.4], dtype=complex), U))
""",
            "call": "recover_real_spectral_decomposition(Z, eigensystem)",
            "gold_call": "_oracle_recover_real_spectral_decomposition(Z, eigensystem)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(20260830)
Z = np.linalg.qr(rng.standard_normal((6, 6)))[0]
U = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
eigensystem = np.vstack((np.array([2.0, -0.5, 0.0], dtype=complex), U))
""",
            "call": "recover_real_spectral_decomposition(Z, eigensystem)",
            "gold_call": "_oracle_recover_real_spectral_decomposition(Z, eigensystem)",
        },
        {
            "setup": """import numpy as np
Z = np.eye(3)
eigensystem = np.vstack((np.array([3.0, 0.5], dtype=complex), np.eye(2, dtype=complex)))
def run_model():
    try:
        recover_real_spectral_decomposition(Z, eigensystem)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_recover_real_spectral_decomposition(Z, eigensystem)
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
Z = 2.0 * np.eye(4)
eigensystem = np.vstack((np.array([3.0, 0.5], dtype=complex), np.eye(2, dtype=complex)))
def run_model():
    try:
        recover_real_spectral_decomposition(Z, eigensystem)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_recover_real_spectral_decomposition(Z, eigensystem)
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
Z = np.eye(4)
Z[0, 0] = np.nan
eigensystem = np.vstack((np.array([3.0, 0.5], dtype=complex), np.eye(2, dtype=complex)))
def run_model():
    try:
        recover_real_spectral_decomposition(Z, eigensystem)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_recover_real_spectral_decomposition(Z, eigensystem)
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
Z = np.eye(4)
eigensystem = np.vstack((np.array([3.0 + 1e-11j, 0.5], dtype=complex), np.eye(2, dtype=complex)))
def run_model():
    try:
        recover_real_spectral_decomposition(Z, eigensystem)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_recover_real_spectral_decomposition(Z, eigensystem)
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
Z = np.eye(4)
eigensystem = np.vstack((np.array([3.0 + 1e-13j, 0.5], dtype=complex), np.eye(2, dtype=complex)))
""",
            "call": "recover_real_spectral_decomposition(Z, eigensystem)",
            "gold_call": "_oracle_recover_real_spectral_decomposition(Z, eigensystem)",
        },
        {
            "setup": """import numpy as np
Z = np.eye(4)
eigensystem = np.vstack((np.array([3.0, 0.5, 0.25], dtype=complex), np.eye(3, dtype=complex)[:, :3]))
def run_model():
    try:
        recover_real_spectral_decomposition(Z, eigensystem)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_recover_real_spectral_decomposition(Z, eigensystem)
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
Z = np.eye(4)
U = 2.0 * np.eye(2, dtype=complex)
eigensystem = np.vstack((np.array([2.0, 0.4], dtype=complex), U))
def run_model():
    try:
        recover_real_spectral_decomposition(Z, eigensystem)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_recover_real_spectral_decomposition(Z, eigensystem)
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
