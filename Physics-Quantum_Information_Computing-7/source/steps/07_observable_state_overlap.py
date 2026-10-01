"""
Evaluate the phase-aware real overlap entering a squared-state observable numerator.

The squared-state numerator can be contracted sparsely by multiplying each observable string into each state string and matching the product back to the stored state support. The full tensor-product Pauli phase is essential, and imaginary contributions that cancel in the Hermitian total must not be discarded term by term. Sum both components cancellation-resistently before applying the relative reality check.

Returns
-------
float, the real normalized overlap 2**(-n) * Tr(O R**2) as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import math
import numpy as np


def observable_state_overlap(
    observable_codes: np.ndarray,
    observable_coefficients: np.ndarray,
    state_codes: np.ndarray,
    state_coefficients: np.ndarray,
    imag_tol: float = 1e-12,
) -> float:
    """Compute the sparse normalized trace overlap entering Tr(O R**2).

    Both O and R use unique Pauli rows and finite real coefficients. Form every
    product in observable-left order with its complete tensor-product Pauli
    phase, match the product against the support of R, and accumulate all real
    and imaginary contributions cancellation-resistently. Apply imag_tol only to
    the final summed imaginary component, since nonzero termwise phases may
    cancel in the Hermitian contraction.

    Parameters
    ----------
    observable_codes : np.ndarray
        Unique integer Pauli rows for O, with shape (m, n).
    observable_coefficients : np.ndarray
        Finite real coefficients of shape (m,).
    state_codes : np.ndarray
        Unique integer Pauli rows for R, with shape (r, n).
    state_coefficients : np.ndarray
        Finite real coefficients of shape (r,).
    imag_tol : float, optional
        Finite nonnegative relative tolerance for the final imaginary residual.

    Returns
    -------
    numerator : float
        Native Python float equal to the normalized trace overlap.

    Raises
    ------
    ValueError
        If arrays, Pauli codes, uniqueness, widths, coefficients, imag_tol, or
        the final reality check are invalid.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_observable_state_overlap(
    observable_codes: np.ndarray,
    observable_coefficients: np.ndarray,
    state_codes: np.ndarray,
    state_coefficients: np.ndarray,
    imag_tol: float = 1e-12,
) -> float:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def validate(
        name: str,
        codes_value: np.ndarray,
        coefficients_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        codes = np.asarray(codes_value)
        if (
            codes.ndim != 2
            or codes.shape[0] < 1
            or codes.shape[1] < 1
            or not np.issubdtype(codes.dtype, np.integer)
        ):
            raise ValueError(f"{name}_codes must be an integer array with shape (m,n), m,n>=1")
        if np.any((codes < 0) | (codes > 3)):
            raise ValueError(f"{name}_codes entries must lie in {{0,1,2,3}}")
        codes = codes.astype(np.int8, copy=False)
        keys = [tuple(int(value) for value in row) for row in codes]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{name}_codes rows must be unique")

        coefficients = np.asarray(coefficients_value)
        if (
            coefficients.ndim != 1
            or coefficients.shape[0] != codes.shape[0]
            or np.iscomplexobj(coefficients)
        ):
            raise ValueError(f"{name}_coefficients must be a real array with shape (m,)")
        coefficients = coefficients.astype(float, copy=False)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError(f"{name}_coefficients must be finite")
        return codes, coefficients

    code_table = (
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 1, 0),
    )
    phase_table = (
        (1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j),
        (1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j),
        (1.0 + 0.0j, 0.0 - 1.0j, 1.0 + 0.0j, 0.0 + 1.0j),
        (1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j, 1.0 + 0.0j),
    )

    def multiply(left: np.ndarray, right: np.ndarray) -> tuple[tuple[int, ...], complex]:
        product: list[int] = []
        phase = 1.0 + 0.0j
        for left_code, right_code in zip(left, right):
            a = int(left_code)
            b = int(right_code)
            product.append(code_table[a][b])
            phase *= phase_table[a][b]
        return tuple(product), phase

    observable, observable_values = validate(
        "observable", observable_codes, observable_coefficients
    )
    state, state_values = validate("state", state_codes, state_coefficients)
    if observable.shape[1] != state.shape[1]:
        raise ValueError("observable and state Pauli widths must match")
    if isinstance(imag_tol, bool) or not isinstance(imag_tol, Real):
        raise ValueError("imag_tol must be a finite nonnegative real scalar")
    tolerance = float(imag_tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("imag_tol must be finite and nonnegative")

    state_map = {
        tuple(int(value) for value in row): float(coefficient)
        for row, coefficient in zip(state, state_values)
    }
    real_parts: list[float] = []
    imag_parts: list[float] = []
    for observable_row, observable_coefficient in zip(observable, observable_values):
        for state_row, state_coefficient in zip(state, state_values):
            product_key, phase = multiply(observable_row, state_row)
            matching_state = state_map.get(product_key)
            if matching_state is None:
                continue
            contribution = (
                float(observable_coefficient)
                * float(state_coefficient)
                * float(matching_state)
                * phase
            )
            real_parts.append(float(contribution.real))
            imag_parts.append(float(contribution.imag))

    real_total = math.fsum(real_parts)
    imag_total = math.fsum(imag_parts)
    if abs(imag_total) > tolerance * max(1.0, abs(real_total)):
        raise ValueError("observable-state overlap is not real within imag_tol")
    if not math.isfinite(real_total):
        raise ValueError("observable-state overlap is not finite")
    return float(real_total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    return [
        # Case 1: simple nonzero numerator.
        {
            "setup": (
                "import numpy as np\n"
                "observable_codes = np.array([[1]], dtype=np.int8)\n"
                "observable_coefficients = np.array([1.0], dtype=float)\n"
                "state_codes = np.array([[0], [1]], dtype=np.int8)\n"
                "state_coefficients = np.array([1.0, 0.3], dtype=float)\n"
                "imag_tol = 1e-12"
            ),
            "call": (
                "observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
            "gold_call": (
                "_oracle_observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
        },
        # Case 2: zero-overlap boundary.
        {
            "setup": (
                "import numpy as np\n"
                "observable_codes = np.array([[3]], dtype=np.int8)\n"
                "observable_coefficients = np.array([2.0], dtype=float)\n"
                "state_codes = np.array([[0], [1]], dtype=np.int8)\n"
                "state_coefficients = np.array([1.0, -0.4], dtype=float)\n"
                "imag_tol = 0.0"
            ),
            "call": (
                "observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
            "gold_call": (
                "_oracle_observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
        },
        # Case 3: phase-aware multi-term case.
        {
            "setup": (
                "import numpy as np\n"
                "observable_codes = np.array([[1]], dtype=np.int8)\n"
                "observable_coefficients = np.array([1.0], dtype=float)\n"
                "state_codes = np.array([[0], [1], [2], [3]], dtype=np.int8)\n"
                "state_coefficients = np.array([1.0, 0.25, -0.4, 0.6], dtype=float)\n"
                "imag_tol = 1e-14"
            ),
            "call": (
                "observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
            "gold_call": (
                "_oracle_observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
        },
        # Case 4: complex observable rejection.
        {
            "setup": (
                "import numpy as np\n"
                "observable_codes = np.array([[1]], dtype=np.int8)\n"
                "observable_coefficients = np.array([1.0 + 0.1j], dtype=complex)\n"
                "state_codes = np.array([[0], [1]], dtype=np.int8)\n"
                "state_coefficients = np.array([1.0, 0.2], dtype=float)\n"
                "imag_tol = 1e-12\n"
                "def run_model():\n"
                "    try:\n"
                "        observable_state_overlap(observable_codes, observable_coefficients, state_codes, state_coefficients, imag_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_observable_state_overlap(observable_codes, observable_coefficients, state_codes, state_coefficients, imag_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 5: mismatched Pauli widths.
        {
            "setup": (
                "import numpy as np\n"
                "observable_codes = np.array([[1,0]], dtype=np.int8)\n"
                "observable_coefficients = np.array([1.0], dtype=float)\n"
                "state_codes = np.array([[0]], dtype=np.int8)\n"
                "state_coefficients = np.array([1.0], dtype=float)\n"
                "imag_tol = 1e-12\n"
                "def run_model():\n"
                "    try:\n"
                "        observable_state_overlap(observable_codes, observable_coefficients, state_codes, state_coefficients, imag_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_observable_state_overlap(observable_codes, observable_coefficients, state_codes, state_coefficients, imag_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },

        # Case 6: two-site Pauli phases change the sign of the real contraction.
        {
            "setup": (
                "import numpy as np\n"
                "observable_codes = np.array([[2,2]], dtype=np.int8)\n"
                "observable_coefficients = np.array([1.0], dtype=float)\n"
                "state_codes = np.array([[1,1], [3,3]], dtype=np.int8)\n"
                "state_coefficients = np.array([0.7, -0.6], dtype=float)\n"
                "imag_tol = 1e-14"
            ),
            "call": (
                "observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
            "gold_call": (
                "_oracle_observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
        },
        # Case 7: a unit residual must survive cancellation of 2**54 terms.
        {
            "setup": (
                "import numpy as np\n"
                "observable_codes = np.array([[0], [1]], dtype=np.int8)\n"
                "observable_coefficients = np.array([1.0, -1.0], dtype=float)\n"
                "state_codes = np.array([[0], [1], [3]], dtype=np.int8)\n"
                "state_coefficients = np.array([2.0**27, 2.0**27, 1.0], dtype=float)\n"
                "imag_tol = 1e-14"
            ),
            "call": (
                "observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
            "gold_call": (
                "_oracle_observable_state_overlap(observable_codes, observable_coefficients, state_codes, "
                "state_coefficients, imag_tol)"
            ),
        },
    ]
