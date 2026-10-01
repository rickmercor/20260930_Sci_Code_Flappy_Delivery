"""
Apply deterministic fixed-cardinality truncation to a sparse Pauli expansion.

Pauli support grows under repeated nonunitary updates.  A fixed-rank cap controls this growth by retaining the largest coefficient magnitudes, while a stable tie convention and restoration of original row order make the selected sparse state reproducible.

Returns
-------
tuple[np.ndarray, np.ndarray] containing at most max_terms integer Pauli rows and their aligned coefficients, returned in original row order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral

import numpy as np


def truncate_pauli_terms(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    max_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain at most the largest max_terms coefficient magnitudes.

    The input rows must already be unique. Ranking is stable: equal magnitudes
    are resolved by the original row index. After selecting the survivors,
    return them in their original input order.

    Parameters
    ----------
    pauli_codes : np.ndarray
        Unique integer Pauli rows of shape (m, n).
    coefficients : np.ndarray
        Finite real coefficients of shape (m,).
    max_terms : int
        Positive maximum number of retained terms; bool is not accepted.

    Returns
    -------
    truncated_codes : np.ndarray
        Selected Pauli rows, with shape (min(m, max_terms), n).
    truncated_coefficients : np.ndarray
        Selected float coefficients in original row order.

    Raises
    ------
    ValueError
        If the arrays are invalid, rows are not unique, or max_terms is not
        a positive integer.
    """
    return (
        np.empty((0, pauli_codes.shape[1]), dtype=np.int8),
        np.empty(0, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_truncate_pauli_terms(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    max_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    from numbers import Integral

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

    keys = [tuple(int(value) for value in row) for row in codes]
    if len(set(keys)) != len(keys):
        raise ValueError("pauli_codes rows must be unique before truncation")
    if isinstance(max_terms, bool) or not isinstance(max_terms, Integral):
        raise ValueError("max_terms must be a positive integer")
    limit = int(max_terms)
    if limit < 1:
        raise ValueError("max_terms must be a positive integer")

    if codes.shape[0] <= limit:
        return codes.copy(), coeffs.copy()

    ranked = sorted(range(codes.shape[0]), key=lambda index: (-abs(coeffs[index]), index))
    retained = sorted(ranked[:limit])
    return codes[retained].copy(), coeffs[retained].copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    return [
        # Case 1: stable magnitude ranking.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0,0], [1,0], [2,0], [3,0], [0,1]], dtype=np.int8)\n"
                "coefficients = np.array([0.2, -0.7, 0.7, 0.5, -0.7], dtype=float)\n"
                "max_terms = 2\n"
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
            "call": "_pack_pair(truncate_pauli_terms(pauli_codes, coefficients, max_terms))",
            "gold_call": "_pack_pair(_oracle_truncate_pauli_terms(pauli_codes, coefficients, max_terms))",
        },
        # Case 2: cap larger than expansion.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0], [1], [3]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, 0.2, -0.3], dtype=float)\n"
                "max_terms = 8\n"
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
            "call": "_pack_pair(truncate_pauli_terms(pauli_codes, coefficients, max_terms))",
            "gold_call": "_pack_pair(_oracle_truncate_pauli_terms(pauli_codes, coefficients, max_terms))",
        },
        # Case 3: equal-magnitude tie boundary.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0,0], [3,3], [1,0]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, -1.0, 0.2], dtype=float)\n"
                "max_terms = 1\n"
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
            "call": "_pack_pair(truncate_pauli_terms(pauli_codes, coefficients, max_terms))",
            "gold_call": "_pack_pair(_oracle_truncate_pauli_terms(pauli_codes, coefficients, max_terms))",
        },
        # Case 4: invalid zero cap.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0]], dtype=np.int8)\n"
                "coefficients = np.array([1.0], dtype=float)\n"
                "max_terms = 0\n"
                "def run_model():\n"
                "    try:\n"
                "        truncate_pauli_terms(pauli_codes, coefficients, max_terms)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_truncate_pauli_terms(pauli_codes, coefficients, max_terms)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 5: duplicate-input rejection.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0,0], [0,0]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, 0.5], dtype=float)\n"
                "max_terms = 1\n"
                "def run_model():\n"
                "    try:\n"
                "        truncate_pauli_terms(pauli_codes, coefficients, max_terms)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_truncate_pauli_terms(pauli_codes, coefficients, max_terms)\n"
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
