"""
Normalize a sparse Pauli expansion by its unique identity coefficient.

Nonunitary propagation can amplify all coefficients by a common factor.  Since only the identity Pauli string contributes to the trace, dividing by its unique nonzero coefficient fixes the trace scale before the next elementary generator without changing coefficient ratios.

Returns
-------
tuple[np.ndarray, np.ndarray] containing a copy of the input code array and normalized finite coefficients whose identity coefficient is exactly one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import numpy as np


def normalize_pauli_trace(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    identity_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale a unique Pauli expansion so its identity coefficient is one.

    If R = sum_P c_P P on n qubits, then the corresponding trace-one
    state is R / 2**n once c_I = 1. The row order is unchanged.

    Parameters
    ----------
    pauli_codes : np.ndarray
        Unique integer Pauli rows of shape (m, n).
    coefficients : np.ndarray
        Finite real coefficients of shape (m,).
    identity_tol : float, optional
        Finite nonnegative lower bound for the absolute identity coefficient.

    Returns
    -------
    normalized_codes : np.ndarray
        Copy of the input Pauli rows.
    normalized_coefficients : np.ndarray
        Float coefficients divided by the unique identity coefficient.

    Raises
    ------
    ValueError
        If the arrays are invalid, rows are not unique, there is not exactly
        one identity row, or its coefficient is not finite and larger than
        identity_tol in magnitude.
    """
    return (
        np.empty_like(pauli_codes, dtype=np.int8),
        np.empty_like(coefficients, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_normalize_pauli_trace(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    identity_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    from numbers import Real

    import numpy as np

    codes = np.asarray(pauli_codes)
    if codes.ndim != 2 or codes.shape[0] < 1 or codes.shape[1] < 1:
        raise ValueError("pauli_codes must have shape (m,n) with m,n >= 1")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("pauli_codes must be an integer array")
    if np.any((codes < 0) | (codes > 3)):
        raise ValueError("pauli_codes entries must lie in {0,1,2,3}")
    codes = codes.astype(np.int8, copy=False)

    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 1 or coeffs.shape[0] != codes.shape[0] or np.iscomplexobj(coeffs):
        raise ValueError("coefficients must be a real array with shape (m,)")
    coeffs = coeffs.astype(float, copy=False)
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must be finite")

    keys = [tuple(int(value) for value in row) for row in codes]
    if len(set(keys)) != len(keys):
        raise ValueError("pauli_codes rows must be unique")
    if isinstance(identity_tol, bool) or not isinstance(identity_tol, Real):
        raise ValueError("identity_tol must be a finite nonnegative real scalar")
    tolerance = float(identity_tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("identity_tol must be finite and nonnegative")

    identity_indices = np.flatnonzero(np.all(codes == 0, axis=1))
    if identity_indices.size != 1:
        raise ValueError("the expansion must contain exactly one identity row")
    identity_coefficient = float(coeffs[int(identity_indices[0])])
    if abs(identity_coefficient) <= tolerance:
        raise ValueError("the identity coefficient is too small to normalize")

    normalized = coeffs / identity_coefficient
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalization produced non-finite coefficients")
    return codes.copy(), normalized.astype(float, copy=False)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    return [
        # Case 1: identity-first normalization.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0,0], [3,3], [1,0]], dtype=np.int8)\n"
                "coefficients = np.array([2.0, -0.5, 0.25], dtype=float)\n"
                "identity_tol = 1e-14\n"
                "def _pack_pair(result):\n"
                "    codes, coeffs = result\n"
                "    codes = np.asarray(codes, dtype=float)\n"
                "    coeffs = np.asarray(coeffs, dtype=float)\n"
                "    header = np.asarray(\n"
                "        [codes.ndim, *codes.shape, coeffs.ndim, *coeffs.shape],\n"
                "        dtype=float,\n"
                "    )\n"
                "    return np.concatenate((header, codes.ravel(), coeffs.ravel()))"
            ),
            "call": "_pack_pair(normalize_pauli_trace(pauli_codes, coefficients, identity_tol))",
            "gold_call": "_pack_pair(_oracle_normalize_pauli_trace(pauli_codes, coefficients, identity_tol))",
        },
        # Case 2: identity row in the middle.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[1,0], [0,0], [0,3]], dtype=np.int8)\n"
                "coefficients = np.array([0.3, 0.75, -0.2], dtype=float)\n"
                "identity_tol = 0.0\n"
                "def _pack_pair(result):\n"
                "    codes, coeffs = result\n"
                "    codes = np.asarray(codes, dtype=float)\n"
                "    coeffs = np.asarray(coeffs, dtype=float)\n"
                "    header = np.asarray(\n"
                "        [codes.ndim, *codes.shape, coeffs.ndim, *coeffs.shape],\n"
                "        dtype=float,\n"
                "    )\n"
                "    return np.concatenate((header, codes.ravel(), coeffs.ravel()))"
            ),
            "call": "_pack_pair(normalize_pauli_trace(pauli_codes, coefficients, identity_tol))",
            "gold_call": "_pack_pair(_oracle_normalize_pauli_trace(pauli_codes, coefficients, identity_tol))",
        },
        # Case 3: negative identity coefficient.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0], [3]], dtype=np.int8)\n"
                "coefficients = np.array([-2.0, 0.5], dtype=float)\n"
                "identity_tol = 1e-15\n"
                "def _pack_pair(result):\n"
                "    codes, coeffs = result\n"
                "    codes = np.asarray(codes, dtype=float)\n"
                "    coeffs = np.asarray(coeffs, dtype=float)\n"
                "    header = np.asarray(\n"
                "        [codes.ndim, *codes.shape, coeffs.ndim, *coeffs.shape],\n"
                "        dtype=float,\n"
                "    )\n"
                "    return np.concatenate((header, codes.ravel(), coeffs.ravel()))"
            ),
            "call": "_pack_pair(normalize_pauli_trace(pauli_codes, coefficients, identity_tol))",
            "gold_call": "_pack_pair(_oracle_normalize_pauli_trace(pauli_codes, coefficients, identity_tol))",
        },
        # Case 4: missing identity row.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[1,0], [3,3]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, 0.5], dtype=float)\n"
                "identity_tol = 1e-14\n"
                "def run_model():\n"
                "    try:\n"
                "        normalize_pauli_trace(pauli_codes, coefficients, identity_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_normalize_pauli_trace(pauli_codes, coefficients, identity_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 5: identity coefficient below tolerance.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0], [1]], dtype=np.int8)\n"
                "coefficients = np.array([1e-16, 0.2], dtype=float)\n"
                "identity_tol = 1e-14\n"
                "def run_model():\n"
                "    try:\n"
                "        normalize_pauli_trace(pauli_codes, coefficients, identity_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_normalize_pauli_trace(pauli_codes, coefficients, identity_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
