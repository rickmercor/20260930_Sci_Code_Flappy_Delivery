"""
Evaluate a cancellation-stable normalized Hilbert--Schmidt overlap of two sparse Pauli expansions.

Tensor-product Pauli strings are orthogonal under the normalized trace inner product, so only equal support rows contribute. The sparse support join avoids dense matrices, but real and imaginary contributions can cancel across widely different scales. Accumulate both components in a cancellation-resistant way so a representable residual is not lost merely because of support order.

Returns
-------
complex, the normalized Hilbert--Schmidt overlap as a native Python complex scalar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
import numpy as np


def pauli_coefficient_overlap(
    left_codes: np.ndarray,
    left_coefficients: np.ndarray,
    right_codes: np.ndarray,
    right_coefficients: np.ndarray,
) -> complex:
    """Compute the normalized Hilbert--Schmidt overlap from sparse coefficients.

    Both expansions must use unique Pauli rows with the same qubit width. Match
    equal rows by their encoded support rather than by array position. Accumulate
    real and imaginary contributions separately with cancellation-resistant
    summation so that a finite representable residual is retained.

    Parameters
    ----------
    left_codes : np.ndarray
        Unique integer Pauli rows for A, with shape (m, n).
    left_coefficients : np.ndarray
        Finite real or complex coefficients of shape (m,).
    right_codes : np.ndarray
        Unique integer Pauli rows for B, with shape (r, n).
    right_coefficients : np.ndarray
        Finite real or complex coefficients of shape (r,).

    Returns
    -------
    overlap : complex
        Native Python complex value of sum_P conj(a_P) * b_P.

    Raises
    ------
    ValueError
        If array shapes, code values, uniqueness, widths, or coefficient values
        are invalid.
    """
    return 0j

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pauli_coefficient_overlap(
    left_codes: np.ndarray,
    left_coefficients: np.ndarray,
    right_codes: np.ndarray,
    right_coefficients: np.ndarray,
) -> complex:
    """Reference implementation."""
    import math
    import numpy as np

    def validate(
        name: str,
        codes_value: np.ndarray,
        coefficients_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        codes = np.asarray(codes_value)
        if codes.ndim != 2 or codes.shape[1] < 1 or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError(f"{name}_codes must be an integer array with shape (m,n), n>=1")
        if np.any((codes < 0) | (codes > 3)):
            raise ValueError(f"{name}_codes entries must lie in {{0,1,2,3}}")
        codes = codes.astype(np.int8, copy=False)
        keys = [tuple(int(value) for value in row) for row in codes]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{name}_codes rows must be unique")

        coefficients = np.asarray(coefficients_value)
        if coefficients.ndim != 1 or coefficients.shape[0] != codes.shape[0]:
            raise ValueError(f"{name}_coefficients must have shape (m,)")
        coefficients = coefficients.astype(np.complex128, copy=False)
        if not np.all(np.isfinite(coefficients.real)) or not np.all(np.isfinite(coefficients.imag)):
            raise ValueError(f"{name}_coefficients must be finite")
        return codes, coefficients

    left, left_values = validate("left", left_codes, left_coefficients)
    right, right_values = validate("right", right_codes, right_coefficients)
    if left.shape[1] != right.shape[1]:
        raise ValueError("left and right Pauli widths must match")

    right_map = {
        tuple(int(value) for value in row): coefficient
        for row, coefficient in zip(right, right_values)
    }
    real_parts: list[float] = []
    imag_parts: list[float] = []
    for row, coefficient in zip(left, left_values):
        other = right_map.get(tuple(int(value) for value in row), 0.0 + 0.0j)
        contribution = np.conjugate(coefficient) * other
        real_parts.append(float(contribution.real))
        imag_parts.append(float(contribution.imag))
    return complex(math.fsum(real_parts), math.fsum(imag_parts))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    return [
        # Case 1: partially matching real expansions.
        {
            "setup": (
                "import numpy as np\n"
                "left_codes = np.array([[0,0], [3,3], [1,0]], dtype=np.int8)\n"
                "left_coefficients = np.array([1.0, 0.5, -0.25], dtype=float)\n"
                "right_codes = left_codes.copy()\n"
                "right_coefficients = np.array([1.0, -0.125, 0.25], dtype=float)"
            ),
            "call": "pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, right_coefficients)",
            "gold_call": (
                "_oracle_pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, "
                "right_coefficients)"
            ),
        },
        # Case 2: disjoint supports.
        {
            "setup": (
                "import numpy as np\n"
                "left_codes = np.array([[1], [2]], dtype=np.int8)\n"
                "left_coefficients = np.array([2.0, 3.0], dtype=float)\n"
                "right_codes = np.array([[0], [3]], dtype=np.int8)\n"
                "right_coefficients = np.array([5.0, 7.0], dtype=float)"
            ),
            "call": "pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, right_coefficients)",
            "gold_call": (
                "_oracle_pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, "
                "right_coefficients)"
            ),
        },
        # Case 3: complex-coefficient overlap.
        {
            "setup": (
                "import numpy as np\n"
                "left_codes = np.array([[0], [1]], dtype=np.int8)\n"
                "left_coefficients = np.array([1.0+2.0j, -0.5j], dtype=complex)\n"
                "right_codes = np.array([[1], [0]], dtype=np.int8)\n"
                "right_coefficients = np.array([3.0-1.0j, 0.25+0.75j], dtype=complex)"
            ),
            "call": "pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, right_coefficients)",
            "gold_call": (
                "_oracle_pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, "
                "right_coefficients)"
            ),
        },
        # Case 4: empty-left boundary.
        {
            "setup": (
                "import numpy as np\n"
                "left_codes = np.empty((0, 2), dtype=np.int8)\n"
                "left_coefficients = np.empty(0, dtype=float)\n"
                "right_codes = np.array([[0,0]], dtype=np.int8)\n"
                "right_coefficients = np.array([1.0], dtype=float)"
            ),
            "call": "pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, right_coefficients)",
            "gold_call": (
                "_oracle_pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, "
                "right_coefficients)"
            ),
        },
        # Case 5: mismatched Pauli widths.
        {
            "setup": (
                "import numpy as np\n"
                "left_codes = np.array([[0,0]], dtype=np.int8)\n"
                "left_coefficients = np.array([1.0], dtype=float)\n"
                "right_codes = np.array([[0]], dtype=np.int8)\n"
                "right_coefficients = np.array([1.0], dtype=float)\n"
                "def run_model():\n"
                "    try:\n"
                "        pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, right_coefficients)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, right_coefficients)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },

        # Case 6: real and imaginary residuals survive catastrophic cancellation.
        {
            "setup": (
                "import numpy as np\n"
                "left_codes = np.array([[0,0], [1,2], [3,1]], dtype=np.int8)\n"
                "left_coefficients = np.ones(3, dtype=complex)\n"
                "right_codes = left_codes.copy()\n"
                "right_coefficients = np.array([complex(2**54, 2**54), 1.0+1.0j, complex(-(2**54), -(2**54))], dtype=complex)\n"
                "def _pack_complex(value):\n"
                "    value = complex(value)\n"
                "    return np.array([value.real, value.imag], dtype=float)"
            ),
            "call": "_pack_complex(pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, right_coefficients))",
            "gold_call": "_pack_complex(_oracle_pauli_coefficient_overlap(left_codes, left_coefficients, right_codes, right_coefficients))",
        },
    ]
