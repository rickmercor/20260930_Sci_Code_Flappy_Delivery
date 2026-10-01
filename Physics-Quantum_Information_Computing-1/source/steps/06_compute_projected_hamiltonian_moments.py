"""
Evaluate shifted-Hamiltonian moments from the projected power matrices, the overlap matrix, and the approximate Krylov ground-state coefficients in the original nonorthogonal basis. Set the zeroth moment to one and use the physical quotient $\mu_q=v^\dagger M^{(q)}v/(v^\dagger S v)$ for every supplied power matrix, returning only real values after checking roundoff-scale imaginary residues.

The shifted-propagator data can be reused to estimate projected powers rather than only the first power. In nonorthogonal Krylov coordinates, the overlap matrix is the metric: Euclidean contraction is valid only after transforming both the vector and every projected operator to one common orthonormal basis. The quotient is invariant under any nonzero complex rescaling of the coefficient vector.

Returns
-------
np.ndarray of shape (number_of_power_matrices + 1,), with moments[0] = 1.0 and all entries stored as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_projected_hamiltonian_moments(
    power_matrices: np.ndarray,
    overlap: np.ndarray,
    state_coefficients: np.ndarray,
) -> np.ndarray:
    """Compute physical moments in a nonorthogonal projected basis.

    Parameters
    ----------
    power_matrices : np.ndarray
        Hermitian complex array of shape ``(q_max, n, n)`` ordered by powers
        1 through ``q_max`` in the original Krylov coordinates.
    overlap : np.ndarray
        Hermitian complex overlap matrix of shape ``(n, n)`` defining the
        physical metric in those coordinates.
    state_coefficients : np.ndarray
        Nonzero complex coefficient vector of length ``n`` in the same
        original Krylov coordinates.

    Returns
    -------
    moments : np.ndarray
        Float64 vector of length ``q_max + 1`` beginning with ``mu_0 = 1``.

    Raises
    ------
    ValueError
        If shapes, finiteness, Hermiticity, positive overlap norm, or
        real-valued moment consistency is invalid.
    """
    return np.empty(power_matrices.shape[0] + 1, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_projected_hamiltonian_moments(
    power_matrices: np.ndarray,
    overlap: np.ndarray,
    state_coefficients: np.ndarray,
) -> np.ndarray:
    import math

    import numpy as np

    try:
        matrices = np.asarray(power_matrices, dtype=np.complex128)
        metric = np.asarray(overlap, dtype=np.complex128)
        vector = np.asarray(state_coefficients, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("inputs must be numeric complex arrays") from exc
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or matrices.shape[0] < 1:
        raise ValueError("power_matrices must have shape (q_max, n, n) with q_max,n >= 1")
    dimension = matrices.shape[1]
    if metric.ndim != 2 or metric.shape != (dimension, dimension):
        raise ValueError("overlap must be a square matrix matching the power matrices")
    if vector.ndim != 1 or vector.size != dimension:
        raise ValueError("state_coefficients must be a vector matching the matrix dimension")
    if not np.all(np.isfinite(matrices.real)) or not np.all(np.isfinite(matrices.imag)):
        raise ValueError("power_matrices must be finite")
    if not np.all(np.isfinite(metric.real)) or not np.all(np.isfinite(metric.imag)):
        raise ValueError("overlap must be finite")
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise ValueError("state_coefficients must be finite")
    if not np.allclose(metric, metric.conjugate().T, rtol=0.0, atol=1.0e-10):
        raise ValueError("overlap must be Hermitian within atol=1e-10")
    for matrix in matrices:
        if not np.allclose(matrix, matrix.conjugate().T, rtol=0.0, atol=1.0e-10):
            raise ValueError("each projected power matrix must be Hermitian within atol=1e-10")

    metric = 0.5 * (metric + metric.conjugate().T)
    denominator = np.vdot(vector, metric @ vector)
    denominator_real = float(np.real(denominator))
    denominator_tolerance = 1.0e-9 * max(1.0, abs(denominator_real))
    if (
        not np.isfinite(denominator.real)
        or not np.isfinite(denominator.imag)
        or abs(float(np.imag(denominator))) > denominator_tolerance
        or not math.isfinite(denominator_real)
        or denominator_real <= 0.0
    ):
        raise ValueError("state_coefficients must have positive finite overlap norm")

    moments = np.empty(matrices.shape[0] + 1, dtype=np.float64)
    moments[0] = 1.0
    for index, matrix in enumerate(matrices, start=1):
        matrix = 0.5 * (matrix + matrix.conjugate().T)
        value = np.vdot(vector, matrix @ vector) / denominator_real
        tolerance = 1.0e-9 * max(1.0, abs(float(np.real(value))))
        if not np.isfinite(value.real) or not np.isfinite(value.imag) or abs(float(np.imag(value))) > tolerance:
            raise ValueError("a Hamiltonian moment is not finite and real within tolerance")
        moments[index] = float(np.real(value))
    return moments

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
powers = np.array([
    np.diag([-1.0, 2.0]),
    np.diag([1.0, 4.0]),
    np.diag([-1.0, 8.0]),
], dtype=complex)
S = np.diag([2.0, 0.5]).astype(complex)
v = np.array([1.0, 2.0j], dtype=complex)
""",
            "call": "compute_projected_hamiltonian_moments(powers, S, v)",
            "gold_call": "_oracle_compute_projected_hamiltonian_moments(powers, S, v)",
        },
        {
            "setup": """import numpy as np
powers = np.array([[[0.5+0j]], [[0.25+0j]], [[0.125+0j]]], dtype=complex)
S = np.array([[3.0+0j]], dtype=complex)
v = np.array([3.0-4.0j], dtype=complex)
""",
            "call": "compute_projected_hamiltonian_moments(powers, S, v)",
            "gold_call": "_oracle_compute_projected_hamiltonian_moments(powers, S, v)",
        },
        {
            "setup": """import numpy as np
powers = np.array([
    [[0.2, 0.1j], [-0.1j, 0.7]],
    [[0.5, 0.03-0.02j], [0.03+0.02j, 1.1]],
], dtype=complex)
S = np.array([[1.0, 0.2+0.1j], [0.2-0.1j, 0.8]], dtype=complex)
v = (7.0-3.0j) * np.array([0.6, 0.8j], dtype=complex)
""",
            "call": "compute_projected_hamiltonian_moments(powers, S, v)",
            "gold_call": "_oracle_compute_projected_hamiltonian_moments(powers, S, v)",
        },
        {
            "setup": """import numpy as np
powers = np.array([np.eye(2)], dtype=complex)
S = np.eye(2, dtype=complex)
v = np.zeros(2, dtype=complex)
def run_model():
    try:
        compute_projected_hamiltonian_moments(powers, S, v)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_projected_hamiltonian_moments(powers, S, v)
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
powers = np.array([np.eye(2)], dtype=complex)
S = np.array([[1.0, 0.2], [0.1, 1.0]], dtype=complex)
v = np.ones(2, dtype=complex)
def run_model():
    try:
        compute_projected_hamiltonian_moments(powers, S, v)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_projected_hamiltonian_moments(powers, S, v)
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
powers = np.array([[[1.0, 1.0], [0.0, 1.0]]], dtype=complex)
S = np.eye(2, dtype=complex)
v = np.ones(2, dtype=complex)
def run_model():
    try:
        compute_projected_hamiltonian_moments(powers, S, v)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_projected_hamiltonian_moments(powers, S, v)
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
