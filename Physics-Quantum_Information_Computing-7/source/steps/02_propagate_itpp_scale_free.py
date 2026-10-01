"""
Apply one globally scale-free imaginary-time Pauli-propagation update.

For a Pauli involution, the exact two-sided imaginary-time update reduces to separate commuting and anticommuting branches. Derive the branch weights from the involution algebra, remove the common positive factor cosh(theta) from the complete output so that each commuting parent's coefficient is unchanged, and evaluate the remaining dimensionless factors without overflow or avoidable underflow. Preserve input-row order and place any generated generator-times-row term immediately after its parent.

Returns
-------
tuple[np.ndarray, np.ndarray] containing the propagated integer Pauli codes with shape (r, n) and aligned finite float coefficients with shape (r,), where m <= r <= 2*m.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import math
import numpy as np


def propagate_itpp_scale_free(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    generator: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one globally rescaled two-sided imaginary-time Pauli update.

    For each stored Pauli row P, determine the branch from its commutation
    relation with the involutory generator Q. Derive the scale-free branch
    weights from exp(-theta * Q / 2) P exp(-theta * Q / 2) after removing
    the common positive factor cosh(theta) from the entire output. The
    coefficient of each commuting parent row is therefore unchanged; do
    not normalize rows or branches independently. Form generated rows as
    QP, preserve parent order,
    and keep every representable scale-free coefficient finite for finite
    inputs, including large absolute theta.

    Parameters
    ----------
    pauli_codes : np.ndarray
        Integer array of shape (m, n) with entries in {0, 1, 2, 3}.
    coefficients : np.ndarray
        Finite real array of shape (m,).
    generator : np.ndarray
        Non-identity integer Pauli code of shape (n,).
    theta : float
        Finite real parameter in exp(-theta * Q / 2).

    Returns
    -------
    propagated_codes : np.ndarray
        Integer array with m to 2*m rows and n columns.
    propagated_coefficients : np.ndarray
        Finite float array aligned with propagated_codes.

    Raises
    ------
    ValueError
        If shapes, Pauli codes, coefficient values, generator, or theta are
        invalid.
    """
    return (
        np.empty((0, pauli_codes.shape[1]), dtype=np.int8),
        np.empty(0, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_propagate_itpp_scale_free(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    generator: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def validate_codes(name: str, values: np.ndarray, ndim: int) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be an integer array with ndim={ndim}")
        if np.any((array < 0) | (array > 3)):
            raise ValueError(f"{name} entries must lie in {{0,1,2,3}}")
        return array.astype(np.int8, copy=False)

    def stable_sech(value: float) -> float:
        magnitude = abs(float(value))
        exp_neg = math.exp(-magnitude)
        return 2.0 * exp_neg / (1.0 + exp_neg * exp_neg)

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

    def multiply(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, complex]:
        product = np.empty(left.shape[0], dtype=np.int8)
        phase = 1.0 + 0.0j
        for index in range(left.shape[0]):
            a = int(left[index])
            b = int(right[index])
            product[index] = code_table[a][b]
            phase *= phase_table[a][b]
        return product, phase

    def commutes(left: np.ndarray, right: np.ndarray) -> bool:
        anti_sites = np.count_nonzero((left != 0) & (right != 0) & (left != right))
        return bool(anti_sites % 2 == 0)

    codes = validate_codes("pauli_codes", pauli_codes, 2)
    if codes.shape[0] < 1 or codes.shape[1] < 1:
        raise ValueError("pauli_codes must have shape (m,n) with m,n >= 1")
    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 1 or coeffs.shape[0] != codes.shape[0]:
        raise ValueError("coefficients must have shape (m,)")
    if np.iscomplexobj(coeffs):
        raise ValueError("coefficients must be real")
    coeffs = coeffs.astype(float, copy=False)
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must be finite")

    gen = validate_codes("generator", generator, 1)
    if gen.shape[0] != codes.shape[1]:
        raise ValueError("generator width must match pauli_codes")
    if np.all(gen == 0):
        raise ValueError("generator must not be the identity string")
    if isinstance(theta, bool) or not isinstance(theta, Real):
        raise ValueError("theta must be a finite real scalar")
    angle = float(theta)
    if not np.isfinite(angle):
        raise ValueError("theta must be finite")

    tanh_value = math.tanh(angle)
    sech_value = stable_sech(angle)
    output_codes: list[np.ndarray] = []
    output_coefficients: list[float] = []

    for code, coefficient in zip(codes, coeffs):
        if commutes(code, gen):
            product, phase = multiply(gen, code)
            if abs(phase.imag) > 1e-12:
                raise ValueError("a commuting generator product must have real phase")
            output_codes.append(code.copy())
            output_coefficients.append(float(coefficient))
            output_codes.append(product)
            output_coefficients.append(float(-coefficient * tanh_value * phase.real))
        else:
            output_codes.append(code.copy())
            output_coefficients.append(float(coefficient * sech_value))

    return np.asarray(output_codes, dtype=np.int8), np.asarray(output_coefficients, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    return [
        # Case 1: mixed commuting/anticommuting branches.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0, 0], [1, 0], [3, 3]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, -0.25, 0.4], dtype=float)\n"
                "generator = np.array([3, 0], dtype=np.int8)\n"
                "theta = -0.16\n"
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
            "call": "_pack_pair(propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
            "gold_call": "_pack_pair(_oracle_propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
        },
        # Case 2: zero-angle boundary.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0], [3]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, 0.3], dtype=float)\n"
                "generator = np.array([3], dtype=np.int8)\n"
                "theta = 0.0\n"
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
            "call": "_pack_pair(propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
            "gold_call": "_pack_pair(_oracle_propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
        },
        # Case 3: large-angle stability edge.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0], [1]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, 2.0], dtype=float)\n"
                "generator = np.array([3], dtype=np.int8)\n"
                "theta = 800.0\n"
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
            "call": "_pack_pair(propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
            "gold_call": "_pack_pair(_oracle_propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
        },
        # Case 4: invalid code rank.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([0, 1], dtype=np.int8)\n"
                "coefficients = np.array([1.0, 0.2], dtype=float)\n"
                "generator = np.array([3, 0], dtype=np.int8)\n"
                "theta = 0.1\n"
                "def run_model():\n"
                "    try:\n"
                "        propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 5: identity-generator rejection.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0, 0]], dtype=np.int8)\n"
                "coefficients = np.array([1.0], dtype=float)\n"
                "generator = np.array([0, 0], dtype=np.int8)\n"
                "theta = 0.1\n"
                "def run_model():\n"
                "    try:\n"
                "        propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },

        # Case 6: multi-qubit phase and mixed-branch ordering.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0,0,0], [1,1,0], [1,0,0], [2,3,0]], dtype=np.int8)\n"
                "coefficients = np.array([1.0, 0.75, -0.5, 0.2], dtype=float)\n"
                "generator = np.array([2,2,0], dtype=np.int8)\n"
                "theta = 0.37\n"
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
            "call": "_pack_pair(propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
            "gold_call": "_pack_pair(_oracle_propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
        },
        # Case 7: large-angle scale-free coefficient remains representable.
        {
            "setup": (
                "import numpy as np\n"
                "pauli_codes = np.array([[0], [1]], dtype=np.int8)\n"
                "coefficients = np.array([0.5, 1e308], dtype=float)\n"
                "generator = np.array([3], dtype=np.int8)\n"
                "theta = 711.0\n"
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
            "call": "_pack_pair(propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
            "gold_call": "_pack_pair(_oracle_propagate_itpp_scale_free(pauli_codes, coefficients, generator, theta))",
        },
    ]
