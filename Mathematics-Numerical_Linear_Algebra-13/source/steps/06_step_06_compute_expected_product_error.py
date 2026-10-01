"""
Evaluate the exact expected squared error of a product with both factors noisy.



For mutually independent zero-mean entry errors with variance fields `$v_A$`

and `$v_B$`, the result is the sum of the left propagated contribution, the

right propagated contribution, and the simultaneous-error contribution:



``sum_ik v_A[ik] ||B[k,:]||_2^2``;

``sum_kj v_B[kj] ||A[:,k]||_2^2``; and

`$sum_k (sum_i v_A[ik]) (sum_j v_B[kj])$`.

Returns
-------
finite float np.ndarray of shape (4,), [left, right, simultaneous, total]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_expected_product_error(
    A_tilde: np.ndarray,
    B_tilde: np.ndarray,
    variance_a: np.ndarray,
    variance_b: np.ndarray,
) -> np.ndarray:
    """Return the three exact error contributions and their total.

    Raises ``ValueError`` unless the factor arrays are finite, nonempty, and
    two-dimensional with a common contracted dimension; the two variance
    arrays have exactly the corresponding factor shapes; and all variances are
    finite and nonnegative.

    Parameters
    ----------
    A_tilde : np.ndarray
        Transformed left factor of shape ``(m, K)``.
    B_tilde : np.ndarray
        Transformed right factor of shape ``(K, n)``.
    variance_a : np.ndarray
        Left entrywise variance field of shape ``(m, K)``.
    variance_b : np.ndarray
        Right entrywise variance field of shape ``(K, n)``.

    Returns
    -------
    np.ndarray
        Finite float array ``[left, right, simultaneous, total]`` of shape
        ``(4,)``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_expected_product_error(
    A_tilde: np.ndarray,
    B_tilde: np.ndarray,
    variance_a: np.ndarray,
    variance_b: np.ndarray,
) -> np.ndarray:
    """Reference evaluation of the finite-dimensional identity."""
    left = np.asarray(A_tilde, dtype=float)
    right = np.asarray(B_tilde, dtype=float)
    field_a = np.asarray(variance_a, dtype=float)
    field_b = np.asarray(variance_b, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] < 1:
        raise ValueError("A_tilde must be a nonempty 2D array")
    if right.ndim != 2 or right.shape[0] < 1 or right.shape[1] < 1:
        raise ValueError("B_tilde must be a nonempty 2D array")
    if left.shape[1] != right.shape[0]:
        raise ValueError("contracted dimensions must agree")
    if field_a.shape != left.shape or field_b.shape != right.shape:
        raise ValueError("variance fields must match their factor shapes")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("factor entries must be finite")
    if not np.all(np.isfinite(field_a)) or np.any(field_a < 0.0):
        raise ValueError("variance_a must be finite and nonnegative")
    if not np.all(np.isfinite(field_b)) or np.any(field_b < 0.0):
        raise ValueError("variance_b must be finite and nonnegative")

    row_energies = np.sum(right**2, axis=1)
    column_energies = np.sum(left**2, axis=0)
    left_contribution = float(np.sum(field_a * row_energies[None, :]))
    right_contribution = float(np.sum(field_b * column_energies[:, None]))
    simultaneous = float(np.sum(np.sum(field_a, axis=0) * np.sum(field_b, axis=1)))
    total = left_contribution + right_contribution + simultaneous
    result = np.array(
        [left_contribution, right_contribution, simultaneous, total], dtype=float
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("expected error must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return task, general contraction, sparse, scaled, zero, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A_tilde = np.array([[12.0,0.5],[7.0,-0.4],[0.6,9.0],[0.8,6.0],[1.2,0.7]])
B_tilde = np.array([[0.2,0.5,-0.3,0.4],[6.0,-4.0,5.0,3.0]])
ranges_a = np.max(np.abs(A_tilde), axis=1)
ranges_b = np.max(np.abs(B_tilde), axis=0)
variance_a = np.repeat((ranges_a**2 / 12.0)[:, None], 2, axis=1)
variance_b = np.repeat((ranges_b**2 / 12.0)[None, :], 2, axis=0)
""",
            "call": "compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "gold_call": "_oracle_compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A_tilde = np.array([[2.0], [-1.0]])
B_tilde = np.array([[3.0, 4.0]])
variance_a = np.array([[0.25], [0.5]])
variance_b = np.array([[0.1, 0.2]])
""",
            "call": "compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "gold_call": "_oracle_compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A_tilde = np.array([[0.0, 2.0]])
B_tilde = np.array([[3.0], [0.0]])
variance_a = np.zeros((1, 2))
variance_b = np.zeros((2, 1))
""",
            "call": "compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "gold_call": "_oracle_compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A_tilde = np.array([[2.0,-1.0,0.5],[-3.0,4.0,1.5]])
B_tilde = np.array([[1.0,-2.0],[0.25,3.0],[-4.0,0.5]])
variance_a = np.array([[0.2,0.4,0.1],[0.3,0.05,0.6]])
variance_b = np.array([[0.7,0.2],[0.1,0.9],[0.4,0.3]])
""",
            "call": "compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "gold_call": "_oracle_compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
A_tilde = np.array([[1e6,1e-6],[-2e5,3e-5]])
B_tilde = np.array([[2e-6,-4e-6],[3e5,1e5]])
variance_a = np.array([[1e-8,2e-8],[3e-9,4e-9]])
variance_b = np.array([[5e-7,6e-7],[7e-8,8e-8]])
""",
            "call": "compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "gold_call": "_oracle_compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)",
            "tol": 1e-8,
        },
        {
            "setup": """import numpy as np
A_tilde = np.ones((2, 2))
B_tilde = np.ones((2, 3))
variance_a = np.ones((2, 1))
variance_b = np.ones((2, 3))
def run_model():
    try:
        compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_expected_product_error(A_tilde, B_tilde, variance_a, variance_b)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 0.0,
        },
    ]
