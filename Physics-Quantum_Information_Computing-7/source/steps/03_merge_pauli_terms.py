"""
Merge duplicate Pauli strings with deterministic ordering and stable real summation.

An elementary propagation step can create repeated Pauli strings whose coefficients interfere constructively or destructively.  Canonicalizing the sparse expansion therefore requires cancellation-stable accumulation, a strict zero threshold, and preservation of first occurrence so later tie rules remain deterministic.

Returns
-------
tuple[np.ndarray, np.ndarray] containing unique integer Pauli rows of shape (r, n) in first-occurrence order and aligned float coefficients of shape (r,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import math
import numpy as np


def merge_pauli_terms(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    zero_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate strings in first-occurrence order.

    Coefficients belonging to identical Pauli rows are summed with a
    cancellation-stable real summation. A merged row is retained only when the
    strict condition abs(sum) > zero_tol holds.

    Parameters
    ----------
    pauli_codes : np.ndarray
        Integer array of shape (m, n) with entries in {0, 1, 2, 3}.
    coefficients : np.ndarray
        Finite real array of shape (m,).
    zero_tol : float, optional
        Finite nonnegative cancellation threshold.

    Returns
    -------
    merged_codes : np.ndarray
        Unique Pauli rows in first-occurrence order, with shape (r, n).
    merged_coefficients : np.ndarray
        Float array of shape (r,) containing the merged coefficients.

    Raises
    ------
    ValueError
        If shapes, Pauli codes, coefficients, or zero_tol are invalid.
    """
    return (
        np.empty((0, pauli_codes.shape[1]), dtype=np.int8),
        np.empty(0, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_merge_pauli_terms(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    zero_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    codes = np.asarray(pauli_codes)
    if codes.ndim != 2 or codes.shape[1] < 1 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("pauli_codes must be an integer array with shape (m,n), n>=1")
    if np.any((codes < 0) | (codes > 3)):
        raise ValueError("pauli_codes entries must lie in {0,1,2,3}")
    codes = codes.astype(np.int8, copy=False)

    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 1 or coeffs.shape[0] != codes.shape[0] or np.iscomplexobj(coeffs):
        raise ValueError("coefficients must be a real array with shape (m,)")
    coeffs = coeffs.astype(float, copy=False)
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must be finite")

    if isinstance(zero_tol, bool) or not isinstance(zero_tol, Real):
        raise ValueError("zero_tol must be a finite nonnegative real scalar")
    tolerance = float(zero_tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tol must be finite and nonnegative")

    first_order: list[tuple[int, ...]] = []
    grouped: dict[tuple[int, ...], list[float]] = {}
    for code, coefficient in zip(codes, coeffs):
        key = tuple(int(value) for value in code)
        if key not in grouped:
            first_order.append(key)
            grouped[key] = []
        grouped[key].append(float(coefficient))

    kept_keys: list[tuple[int, ...]] = []
    kept_values: list[float] = []
    for key in first_order:
        merged = math.fsum(grouped[key])
        if abs(merged) > tolerance:
            kept_keys.append(key)
            kept_values.append(float(merged))

    if kept_keys:
        merged_codes = np.asarray(kept_keys, dtype=np.int8)
    else:
        merged_codes = np.empty((0, codes.shape[1]), dtype=np.int8)
    return merged_codes, np.asarray(kept_values, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    return [
        # Case 1: duplicate merging.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0,0], [3,0], [0,0], [3,0], [1,1]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, 0.4, -0.25, 0.1, 0.2], dtype=float)\n"
                "zero_tol = 1e-14\n"
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
            "call": "_pack_pair(merge_pauli_terms(pauli_codes, coefficients, zero_tol))",
            "gold_call": "_pack_pair(_oracle_merge_pauli_terms(pauli_codes, coefficients, zero_tol))",
        },
        # Case 2: already-unique boundary.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0], [1], [2], [3]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, -2.0, 3.0, -4.0], dtype=float)\n"
                "zero_tol = 0.0\n"
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
            "call": "_pack_pair(merge_pauli_terms(pauli_codes, coefficients, zero_tol))",
            "gold_call": "_pack_pair(_oracle_merge_pauli_terms(pauli_codes, coefficients, zero_tol))",
        },
        # Case 3: cancellation-stability edge.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[1,0], [1,0], [1,0], [3,3], [3,3]], dtype=np.int8)\n"
                "coefficients = np.array([1e16, 1.0, -1e16, 0.25, -0.25], dtype=float)\n"
                "zero_tol = 0.5\n"
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
            "call": "_pack_pair(merge_pauli_terms(pauli_codes, coefficients, zero_tol))",
            "gold_call": "_pack_pair(_oracle_merge_pauli_terms(pauli_codes, coefficients, zero_tol))",
        },
        # Case 4: coefficient-shape mismatch.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0,0], [1,0]], dtype=np.int8)\n"
                "coefficients = np.array([1.0], dtype=float)\n"
                "zero_tol = 1e-14\n"
                "def run_model():\n"
                "    try:\n"
                "        merge_pauli_terms(pauli_codes, coefficients, zero_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_merge_pauli_terms(pauli_codes, coefficients, zero_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 5: invalid negative tolerance.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0]], dtype=np.int8)\n"
                "coefficients = np.array([1.0], dtype=float)\n"
                "zero_tol = -1.0\n"
                "def run_model():\n"
                "    try:\n"
                "        merge_pauli_terms(pauli_codes, coefficients, zero_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_merge_pauli_terms(pauli_codes, coefficients, zero_tol)\n"
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
