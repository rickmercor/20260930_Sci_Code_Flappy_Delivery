"""
Extract and order the finite spectrum of a generalized matrix pencil.



The homogeneous generalized-eigenvalue representation `$alpha / beta$` is

used so infinite modes can be identified before division.  This matters for

singular right-hand matrices and avoids treating overflowed quotients as

ordinary finite eigenvalues.

Returns
-------
one nonempty finite nondecreasing float array containing all finite mode magnitudes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_finite_spectral_magnitudes(
    a_matrix: np.ndarray, q_matrix: np.ndarray
) -> np.ndarray:
    """Return sorted magnitudes of the finite generalized eigenvalues.

    Both arguments must be matching, nonempty, finite square matrices.  Modes
    whose homogeneous denominator is exactly zero are excluded.  A
    ``ValueError`` is raised if no finite eigenvalue remains.

    Parameters
    ----------
    a_matrix : np.ndarray
        Left matrix of ``A y = lambda Q y``, shape ``(m, m)``.
    q_matrix : np.ndarray
        Right matrix of ``A y = lambda Q y``, shape ``(m, m)``.

    Returns
    -------
    np.ndarray
        Nondecreasing finite eigenvalue magnitudes.
    """
    return magnitudes  # noqa: F821 - model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import linalg

def _oracle_compute_finite_spectral_magnitudes(
    a_matrix: np.ndarray, q_matrix: np.ndarray
) -> np.ndarray:
    """Reference homogeneous generalized-eigenvalue filtering."""
    a = np.asarray(a_matrix, dtype=float)
    q = np.asarray(q_matrix, dtype=float)
    if (
        a.ndim != 2
        or q.ndim != 2
        or a.shape != q.shape
        or a.shape[0] == 0
        or a.shape[0] != a.shape[1]
    ):
        raise ValueError("pencil matrices must be matching nonempty squares")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(q)):
        raise ValueError("pencil matrices must be finite")

    homogeneous = linalg.eigvals(a, q, homogeneous_eigvals=True, check_finite=True)
    numerators = homogeneous[0]
    denominators = homogeneous[1]
    mask = np.isfinite(numerators) & np.isfinite(denominators) & (denominators != 0.0)
    eigenvalues = numerators[mask] / denominators[mask]
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    if eigenvalues.size == 0:
        raise ValueError("the pencil has no finite generalized eigenvalues")
    magnitudes = np.sort(np.abs(eigenvalues).astype(float, copy=False))
    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("finite generalized eigenvalues could not be resolved")
    return magnitudes

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return diagonal, coupled, singular-right, and invalid-shape cases."""
    return [
        {
            "setup": """import numpy as np
a_matrix = np.diag(np.array([-6.0, 1.0, 8.0]))
q_matrix = np.diag(np.array([2.0, 0.25, 4.0]))
""",
            "call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(compute_finite_spectral_magnitudes(a_matrix, q_matrix))",
            "gold_call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(_oracle_compute_finite_spectral_magnitudes(a_matrix, q_matrix))",
        },
        {
            "setup": """import numpy as np
a_matrix = np.array([[3.0, -7.0, 0.5], [2.0, 1.0, 4.0], [0.0, -2.0, 5.0]])
q_matrix = np.array([[2.0, 0.0, 0.25], [0.0, 1.5, 0.0], [0.0, 0.0, 0.75]])
""",
            "call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(compute_finite_spectral_magnitudes(a_matrix, q_matrix))",
            "gold_call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(_oracle_compute_finite_spectral_magnitudes(a_matrix, q_matrix))",
        },
        {
            "setup": """import numpy as np
a_matrix = np.diag(np.array([2.0, -9.0, 5.0]))
q_matrix = np.diag(np.array([1.0, 0.0, 0.5]))
""",
            "call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(compute_finite_spectral_magnitudes(a_matrix, q_matrix))",
            "gold_call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(_oracle_compute_finite_spectral_magnitudes(a_matrix, q_matrix))",
        },
        {
            "setup": """import numpy as np
a_matrix = np.eye(2)
q_matrix = np.eye(3)
def run_model():
    try:
        compute_finite_spectral_magnitudes(a_matrix, q_matrix)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_finite_spectral_magnitudes(a_matrix, q_matrix)
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
