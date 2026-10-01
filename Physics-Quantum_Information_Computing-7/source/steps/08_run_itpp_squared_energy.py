"""
Run the complete sequential sparse imaginary-time Pauli-propagation pipeline and return the squared-state energy.

The final workflow builds the ordered open-chain Hamiltonian, repeatedly composes the seven preceding public operations for propagation, merging, truncation, normalization, and overlap evaluation, and returns Tr(H R**2) / Tr(R**2).

Returns
-------
float, the finite squared-state total-energy estimate Tr(H R**2) / Tr(R**2) as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real

import math
import numpy as np


def run_itpp_squared_energy(
    n_qubits: int,
    coupling: float,
    field: float,
    delta_tau: float,
    final_tau: float,
    max_terms: int,
    zero_tol: float = 1e-14,
) -> float:
    """Run sequential sparse imaginary-time propagation and a squared-state estimator.

    Build the ordered open-chain Hamiltonian, start from the identity expansion,
    and use ceil(final_tau / delta_tau) full first-order Trotter layers. Call
    build_open_tfim_pauli_terms once, then for every elementary generator
    call propagate_itpp_scale_free, merge_pauli_terms,
    truncate_pauli_terms, and normalize_pauli_trace in that order.
    Finally call pauli_coefficient_overlap and
    observable_state_overlap to return Tr(H R**2) / Tr(R**2) in
    normalized Pauli units.

    Parameters
    ----------
    n_qubits : int
        Positive number of qubits.
    coupling : float
        Finite real nearest-neighbour coupling.
    field : float
        Finite real transverse-field strength.
    delta_tau : float
        Finite strictly positive Trotter step size.
    final_tau : float
        Finite nonnegative target imaginary time.
    max_terms : int
        Positive fixed-rank Pauli truncation cap.
    zero_tol : float, optional
        Finite nonnegative threshold used for merged cancellation and identity
        normalization.

    Returns
    -------
    energy : float
        Native Python float equal to Tr(H R**2) / Tr(R**2).

    Raises
    ------
    ValueError
        If scalar controls are invalid, an intermediate expansion cannot be
        trace-normalized, or the final overlap ratio is non-finite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_itpp_squared_energy(
    n_qubits: int,
    coupling: float,
    field: float,
    delta_tau: float,
    final_tau: float,
    max_terms: int,
    zero_tol: float = 1e-14,
) -> float:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    if isinstance(delta_tau, bool) or not isinstance(delta_tau, Real):
        raise ValueError("delta_tau must be a finite positive real scalar")
    if isinstance(final_tau, bool) or not isinstance(final_tau, Real):
        raise ValueError("final_tau must be a finite nonnegative real scalar")
    if isinstance(max_terms, bool) or not isinstance(max_terms, Integral):
        raise ValueError("max_terms must be a positive integer")
    if isinstance(zero_tol, bool) or not isinstance(zero_tol, Real):
        raise ValueError("zero_tol must be a finite nonnegative real scalar")
    step_size = float(delta_tau)
    target_time = float(final_tau)
    limit = int(max_terms)
    tolerance = float(zero_tol)
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("delta_tau must be finite and positive")
    if not math.isfinite(target_time) or target_time < 0.0:
        raise ValueError("final_tau must be finite and nonnegative")
    if limit < 1:
        raise ValueError("max_terms must be a positive integer")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tol must be finite and nonnegative")

    hamiltonian_codes, hamiltonian_coefficients = (
        _oracle_build_open_tfim_pauli_terms(n_qubits, coupling, field)
    )
    state_codes = np.zeros((1, int(n_qubits)), dtype=np.int8)
    state_coefficients = np.array([1.0], dtype=float)
    n_steps = int(math.ceil(target_time / step_size))

    for _ in range(n_steps):
        for generator, strength in zip(hamiltonian_codes, hamiltonian_coefficients):
            raw_codes, raw_coefficients = _oracle_propagate_itpp_scale_free(
                state_codes,
                state_coefficients,
                generator,
                float(strength) * step_size,
            )
            merged_codes, merged_coefficients = _oracle_merge_pauli_terms(
                raw_codes, raw_coefficients, tolerance
            )
            truncated_codes, truncated_coefficients = _oracle_truncate_pauli_terms(
                merged_codes, merged_coefficients, limit
            )
            state_codes, state_coefficients = _oracle_normalize_pauli_trace(
                truncated_codes, truncated_coefficients, tolerance
            )

    denominator_value = _oracle_pauli_coefficient_overlap(
        state_codes, state_coefficients, state_codes, state_coefficients
    )
    if abs(denominator_value.imag) > 1e-12 * max(1.0, abs(denominator_value.real)):
        raise ValueError("the squared-state denominator is not real")
    denominator = float(denominator_value.real)
    numerator = _oracle_observable_state_overlap(
        hamiltonian_codes,
        hamiltonian_coefficients,
        state_codes,
        state_coefficients,
        1e-12,
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("the squared-state denominator must be finite and positive")
    energy = numerator / denominator
    if not math.isfinite(energy):
        raise ValueError("the final energy must be finite")
    return float(energy)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    return [
        # Case 1: normal two-qubit pipeline.
        {
            "setup": (
                "n_qubits = 2\n"
                "coupling = 1.1\n"
                "field = 0.4\n"
                "delta_tau = 0.13\n"
                "final_tau = 0.39\n"
                "max_terms = 4\n"
                "zero_tol = 1e-14"
            ),
            "call": (
                "run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
            "gold_call": (
                "_oracle_run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
        },
        # Case 2: zero-time boundary.
        {
            "setup": (
                "n_qubits = 1\n"
                "coupling = 1.0\n"
                "field = 0.7\n"
                "delta_tau = 0.2\n"
                "final_tau = 0.0\n"
                "max_terms = 1\n"
                "zero_tol = 1e-14"
            ),
            "call": (
                "run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
            "gold_call": (
                "_oracle_run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
        },
        # Case 3: large-angle stability edge.
        {
            "setup": (
                "n_qubits = 1\n"
                "coupling = 0.0\n"
                "field = 1.0\n"
                "delta_tau = 800.0\n"
                "final_tau = 800.0\n"
                "max_terms = 2\n"
                "zero_tol = 0.0"
            ),
            "call": (
                "run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
            "gold_call": (
                "_oracle_run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
        },
        # Case 4: invalid zero step size.
        {
            "setup": (
                "n_qubits = 2\n"
                "coupling = 1.0\n"
                "field = 0.5\n"
                "delta_tau = 0.0\n"
                "final_tau = 0.4\n"
                "max_terms = 4\n"
                "zero_tol = 1e-14\n"
                "def run_model():\n"
                "    try:\n"
                "        run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, zero_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, zero_tol)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 5: three-qubit pipeline perturbation.
        {
            "setup": (
                "n_qubits = 3\n"
                "coupling = 0.8\n"
                "field = 0.55\n"
                "delta_tau = 0.11\n"
                "final_tau = 0.33\n"
                "max_terms = 5\n"
                "zero_tol = 1e-14"
            ),
            "call": (
                "run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
            "gold_call": (
                "_oracle_run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
        },

        # Case 6: nonintegral layer count with order- and mutation-sensitive truncation.
        {
            "setup": (
                "n_qubits = 4\n"
                "coupling = 1.2\n"
                "field = 0.4\n"
                "delta_tau = 0.17\n"
                "final_tau = 0.41\n"
                "max_terms = 9\n"
                "zero_tol = 1e-14"
            ),
            "call": (
                "run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
            "gold_call": (
                "_oracle_run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
        },
        # Case 7: low-rank four-qubit case stresses sequential truncation.
        {
            "setup": (
                "n_qubits = 4\n"
                "coupling = 0.95\n"
                "field = 0.36\n"
                "delta_tau = 0.17\n"
                "final_tau = 0.53\n"
                "max_terms = 6\n"
                "zero_tol = 1e-14"
            ),
            "call": (
                "run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
            "gold_call": (
                "_oracle_run_itpp_squared_energy(n_qubits, coupling, field, delta_tau, final_tau, max_terms, "
                "zero_tol)"
            ),
        },
    ]
