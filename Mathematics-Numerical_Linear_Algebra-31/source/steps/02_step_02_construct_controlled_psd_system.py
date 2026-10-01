"""
Construct the deterministic positive-definite spectral system used by the benchmark.

A cosine eigenbasis and geometric spectrum produce a reproducible symmetric positive-definite matrix with a prescribed condition number. The right-hand side has nonzero components across the spectrum so the finite Krylov process exercises both large and small eigenmodes.

Returns
-------
np.ndarray, an augmented array [A | b] with shape (n, n+1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_controlled_psd_system(
    n: int,
    condition_number: float,
) -> np.ndarray:
    """Return the augmented array ``[A | b]`` for a controlled spectral system.

    Let ``U`` be the orthogonal cosine basis with

    ``U[i, 0] = 1 / sqrt(n)`` and
    ``U[i, j] = sqrt(2/n) cos(pi (i + 1/2) j / n)`` for ``j > 0``.

    Define ``lambda[j] = condition_number**(-j/(n-1))`` and
    ``c[j] = 1 + 0.25 cos(pi (j+1)/(n+1))``. Form
    ``A = U diag(lambda) U.T`` and ``b = U c / ||c||_2``.

    Parameters
    ----------
    n : int
        System dimension, at least 4. Boolean values are invalid.
    condition_number : float
        Finite target condition number strictly greater than 1.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, n+1)`` whose first ``n`` columns are the
        symmetric positive-definite matrix ``A`` and whose last column is
        the unit-norm vector ``b``.

    Raises
    ------
    ValueError
        If ``n`` is not an integer at least 4 (booleans are invalid), or if
        ``condition_number`` is not finite and strictly greater than 1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_construct_controlled_psd_system(
    n: int,
    condition_number: float,
) -> np.ndarray:
    """Return the deterministic augmented spectral system."""
    np = __import__("numpy")

    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("n must be an integer at least 4")
    condition_number = float(condition_number)
    if not np.isfinite(condition_number) or condition_number <= 1.0:
        raise ValueError("condition_number must be finite and greater than 1")

    rows = np.arange(n, dtype=float)[:, None]
    cols = np.arange(n, dtype=float)[None, :]
    U = np.sqrt(2.0 / n) * np.cos(np.pi * (rows + 0.5) * cols / n)
    U[:, 0] = 1.0 / np.sqrt(n)
    indices = np.arange(n, dtype=float)
    eigenvalues = condition_number ** (-indices / (n - 1))
    coefficients = 1.0 + 0.25 * np.cos(np.pi * (indices + 1.0) / (n + 1.0))
    A = (U * eigenvalues) @ U.T
    A = 0.5 * (A + A.T)
    b = U @ coefficients
    b = b / np.linalg.norm(b)
    return np.column_stack((A, b))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return small, ill-conditioned, larger, and invalid cases."""
    return [
        {
            "setup": "n = 4\ncondition_number = 10.0",
            "call": "construct_controlled_psd_system(n, condition_number)",
            "gold_call": "_oracle_construct_controlled_psd_system(n, condition_number)",
        },
        {
            "setup": "n = 7\ncondition_number = 1.0e8",
            "call": "construct_controlled_psd_system(n, condition_number)",
            "gold_call": "_oracle_construct_controlled_psd_system(n, condition_number)",
        },
        {
            "setup": "n = 12\ncondition_number = 257.0",
            "call": "construct_controlled_psd_system(n, condition_number)",
            "gold_call": "_oracle_construct_controlled_psd_system(n, condition_number)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        construct_controlled_psd_system(3, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_controlled_psd_system(3, 100.0)
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
