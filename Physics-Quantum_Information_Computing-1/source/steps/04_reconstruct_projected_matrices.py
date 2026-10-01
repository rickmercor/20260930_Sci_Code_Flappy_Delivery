"""
Use the centered finite-difference rows and the symmetric-shift propagator tensor to reconstruct the overlap matrix and approximate projected powers of the shifted Hamiltonian. Let $\delta_t=\texttt{delta\_t}$. Obtain the overlap matrix from the zero-shift slice. For derivative order $q$, multiply the corresponding weighted propagator sum by $i^q/\delta_t^q$. Replace the overlap matrix and every reconstructed power matrix $A$ by its Hermitian part $(A+A^\dagger)/2$.

Differentiating a unitary propagator at zero time produces powers of its Hermitian generator. The same measured shifted propagators therefore yield the projected Hamiltonian and all projected powers through the highest available derivative order by classical post-processing. Centered weights carry alternating parity, and Hermitian projection preserves the observable matrix while suppressing floating-point asymmetry.

Returns
-------
tuple (overlap, power_matrices), where overlap has shape (n, n) and power_matrices has shape (number_of_derivative_rows, n, n), both complex128
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import numpy as np

def reconstruct_projected_matrices(
    propagators: np.ndarray,
    coefficients: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct overlap and projected shifted-Hamiltonian powers.

    Parameters
    ----------
    propagators : np.ndarray
        Complex array of shape ``(2 * degree + 1, n, n)`` ordered by symmetric
        shift labels.
    coefficients : np.ndarray
        Real array of shape ``(q_max, 2 * degree + 1)`` with derivative rows
        ordered from 1 through ``q_max``.
    delta_t : float
        Positive finite time shift.

    Returns
    -------
    overlap : np.ndarray
        Hermitian complex128 overlap matrix of shape ``(n, n)``.
    power_matrices : np.ndarray
        Hermitian complex128 array of shape ``(q_max, n, n)`` whose row
        ``q - 1`` approximates the projected ``q``-th power.

    Raises
    ------
    ValueError
        If shapes are inconsistent, entries are non-finite, ``delta_t`` is not
        positive and finite, or a finite complex128 result cannot be formed.
    """
    return (
        np.empty((propagators.shape[1], propagators.shape[1]), dtype=np.complex128),
        np.empty(
            (coefficients.shape[0], propagators.shape[1], propagators.shape[1]),
            dtype=np.complex128,
        ),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reconstruct_projected_matrices(
    propagators: np.ndarray,
    coefficients: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    import math
    from numbers import Real

    import numpy as np

    if isinstance(delta_t, bool) or not isinstance(delta_t, Real):
        raise ValueError("delta_t must be a positive finite real scalar")
    delta = float(delta_t)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_t must be positive and finite")

    try:
        u = np.asarray(propagators, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("propagators must be a numeric complex array") from exc
    raw_coefficients = np.asarray(coefficients)
    if np.iscomplexobj(raw_coefficients) and np.any(np.abs(np.imag(raw_coefficients)) > 0.0):
        raise ValueError("coefficients must be real")
    try:
        weights = np.asarray(raw_coefficients, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("coefficients must be a real numeric array") from exc

    if u.ndim != 3 or u.shape[1] != u.shape[2] or u.shape[1] < 1 or u.shape[0] % 2 != 1:
        raise ValueError("propagators must have shape (odd, n, n) with n >= 1")
    if weights.ndim != 2 or weights.shape[1] != u.shape[0] or weights.shape[0] < 1:
        raise ValueError("coefficient columns must match the propagator shift axis")
    degree = (u.shape[0] - 1) // 2
    if weights.shape[0] > 2 * degree:
        raise ValueError("no more than 2 * degree derivative rows are supported")
    if not np.all(np.isfinite(u.real)) or not np.all(np.isfinite(u.imag)) or not np.all(np.isfinite(weights)):
        raise ValueError("propagators and coefficients must be finite")

    center = degree
    overlap = 0.5 * (u[center] + u[center].conjugate().T)
    if not np.all(np.isfinite(overlap.real)) or not np.all(np.isfinite(overlap.imag)):
        raise ValueError("the reconstructed overlap is not finite complex128")

    powers = np.empty((weights.shape[0], u.shape[1], u.shape[2]), dtype=np.complex128)
    log_max = math.log(float(np.finfo(np.float64).max))
    log_min_subnormal = math.log(float(np.nextafter(0.0, 1.0)))
    for derivative_order in range(1, weights.shape[0] + 1):
        accumulator = np.zeros((u.shape[1], u.shape[2]), dtype=np.complex128)
        with np.errstate(over="ignore", invalid="ignore"):
            for shift_position in range(u.shape[0]):
                accumulator += float(weights[derivative_order - 1, shift_position]) * u[shift_position]
        if not np.all(np.isfinite(accumulator.real)) or not np.all(np.isfinite(accumulator.imag)):
            raise ValueError("a weighted propagator sum is not finite complex128")

        power_log = derivative_order * math.log(delta)
        log_inverse_power = -power_log
        matrix = None
        if math.isfinite(power_log) and abs(power_log) <= log_max:
            try:
                denominator = delta ** derivative_order
                scale = np.complex128((1j ** derivative_order) / denominator)
                with np.errstate(over="ignore", invalid="ignore"):
                    matrix = np.asarray(scale * accumulator, dtype=np.complex128)
            except (OverflowError, ZeroDivisionError):
                matrix = None

        if matrix is None:
            # Form the product entry-by-entry in log magnitude whenever either
            # delta**q or its inverse is outside the directly representable range.
            # Exact zeros remain zero; any nonzero result outside complex128 range
            # raises the declared ValueError rather than leaking an arithmetic error.
            phase = 1j ** derivative_order
            matrix = np.zeros_like(accumulator, dtype=np.complex128)
            for index in np.ndindex(accumulator.shape):
                value = complex(accumulator[index])
                magnitude = abs(value)
                if magnitude == 0.0:
                    continue
                log_magnitude = math.log(magnitude) + log_inverse_power
                if not math.isfinite(log_magnitude) or log_magnitude > log_max:
                    raise ValueError("a reconstructed power matrix is not finite complex128")
                scaled_magnitude = 0.0 if log_magnitude < log_min_subnormal else math.exp(log_magnitude)
                matrix[index] = np.complex128(phase * (value / magnitude) * scaled_magnitude)

        matrix = 0.5 * (matrix + matrix.conjugate().T)
        if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
            raise ValueError("a reconstructed power matrix is not finite complex128")
        powers[derivative_order - 1] = matrix

    return overlap.astype(np.complex128), powers

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
propagators = np.array([
    [[0.8+0.2j]],
    [[1.0+0.0j]],
    [[0.8-0.2j]],
], dtype=complex)
coefficients = np.array([[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]], dtype=float)

def pack_result(result):
    overlap, powers = result
    shape = np.asarray([overlap.shape[0], overlap.shape[1], *powers.shape], dtype=float)
    return np.concatenate([
        shape, overlap.real.ravel(), overlap.imag.ravel(),
        powers.real.ravel(), powers.imag.ravel(),
    ])
""",
            "call": "pack_result(reconstruct_projected_matrices(propagators, coefficients, 0.1))",
            "gold_call": "pack_result(_oracle_reconstruct_projected_matrices(propagators, coefficients, 0.1))",
        },
        {
            "setup": """import numpy as np
propagators = np.zeros((3, 2, 2), dtype=complex)
propagators[1] = np.array([[1.0, 0.3-0.1j], [0.3+0.1j, 1.0]])
propagators[0] = np.array([[0.9+0.2j, 0.1], [0.2, 0.7+0.1j]])
propagators[2] = propagators[0].conjugate().T
coefficients = np.array([[-0.5, 0.0, 0.5]], dtype=float)

def pack_result(result):
    overlap, powers = result
    shape = np.asarray([overlap.shape[0], overlap.shape[1], *powers.shape], dtype=float)
    return np.concatenate([
        shape, overlap.real.ravel(), overlap.imag.ravel(),
        powers.real.ravel(), powers.imag.ravel(),
    ])
""",
            "call": "pack_result(reconstruct_projected_matrices(propagators, coefficients, 0.25))",
            "gold_call": "pack_result(_oracle_reconstruct_projected_matrices(propagators, coefficients, 0.25))",
        },
        {
            "setup": """import numpy as np
propagators = np.ones((5, 1, 1), dtype=complex)
coefficients = np.array([[1/12, -2/3, 0.0, 2/3, -1/12]], dtype=float)

def pack_result(result):
    overlap, powers = result
    shape = np.asarray([overlap.shape[0], overlap.shape[1], *powers.shape], dtype=float)
    return np.concatenate([
        shape, overlap.real.ravel(), overlap.imag.ravel(),
        powers.real.ravel(), powers.imag.ravel(),
    ])
""",
            "call": "pack_result(reconstruct_projected_matrices(propagators, coefficients, 1.0e-3))",
            "gold_call": "pack_result(_oracle_reconstruct_projected_matrices(propagators, coefficients, 1.0e-3))",
        },
        {
            "setup": """import numpy as np
propagators = np.ones((3, 2, 2), dtype=complex)
coefficients = np.ones((1, 5), dtype=float)
def run_model():
    try:
        reconstruct_projected_matrices(propagators, coefficients, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_reconstruct_projected_matrices(propagators, coefficients, 0.1)
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
propagators = np.ones((3, 1, 1), dtype=complex)
coefficients = np.array([[-0.5, 0.0, 0.5]], dtype=float)
def run_model():
    try:
        reconstruct_projected_matrices(propagators, coefficients, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_reconstruct_projected_matrices(propagators, coefficients, 0.0)
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
propagators = np.array([
    [[0.8+0.2j]],
    [[1.0+0.0j]],
    [[0.8-0.2j]],
], dtype=complex)
coefficients = np.array([[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]], dtype=float)

def pack_result(result):
    overlap, powers = result
    shape = np.asarray([overlap.shape[0], overlap.shape[1], *powers.shape], dtype=float)
    return np.concatenate([
        shape, overlap.real.ravel(), overlap.imag.ravel(),
        powers.real.ravel(), powers.imag.ravel(),
    ])
""",
            "call": "pack_result(reconstruct_projected_matrices(propagators, coefficients, 1.0e300))",
            "gold_call": "pack_result(_oracle_reconstruct_projected_matrices(propagators, coefficients, 1.0e300))",
        },
        {
            "setup": """import numpy as np
propagators = np.array([
    [[0.8+0.2j]],
    [[1.0+0.0j]],
    [[0.8-0.2j]],
], dtype=complex)
coefficients = np.array([[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]], dtype=float)

def run_model():
    try:
        reconstruct_projected_matrices(propagators, coefficients, 1.0e-300)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_reconstruct_projected_matrices(propagators, coefficients, 1.0e-300)
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
