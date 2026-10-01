#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def build_open_tfim_pauli_terms(
    n_qubits: int,
    coupling: float,
    field: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    from numbers import Integral, Real

    import numpy as np

    if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral):
        raise ValueError("n_qubits must be a positive integer")
    n = int(n_qubits)
    if n < 1:
        raise ValueError("n_qubits must be a positive integer")
    if isinstance(coupling, bool) or not isinstance(coupling, Real):
        raise ValueError("coupling must be a finite real scalar")
    if isinstance(field, bool) or not isinstance(field, Real):
        raise ValueError("field must be a finite real scalar")
    j = float(coupling)
    h = float(field)
    if not np.isfinite(j) or not np.isfinite(h):
        raise ValueError("coupling and field must be finite")

    codes = np.zeros((2 * n - 1, n), dtype=np.int8)
    coeffs = np.empty(2 * n - 1, dtype=float)

    row = 0
    for site in range(n - 1):
        codes[row, site] = 3
        codes[row, site + 1] = 3
        coeffs[row] = -j
        row += 1
    for site in range(n):
        codes[row, site] = 1
        coeffs[row] = -h
        row += 1

    return codes, coeffs

import numpy as np


def propagate_itpp_scale_free(
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

import numpy as np


def merge_pauli_terms(
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

import numpy as np


def truncate_pauli_terms(
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

import numpy as np


def normalize_pauli_trace(
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

import numpy as np


def pauli_coefficient_overlap(
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

import numpy as np


def observable_state_overlap(
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

def run_itpp_squared_energy(
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
        build_open_tfim_pauli_terms(n_qubits, coupling, field)
    )
    state_codes = np.zeros((1, int(n_qubits)), dtype=np.int8)
    state_coefficients = np.array([1.0], dtype=float)
    n_steps = int(math.ceil(target_time / step_size))

    for _ in range(n_steps):
        for generator, strength in zip(hamiltonian_codes, hamiltonian_coefficients):
            raw_codes, raw_coefficients = propagate_itpp_scale_free(
                state_codes,
                state_coefficients,
                generator,
                float(strength) * step_size,
            )
            merged_codes, merged_coefficients = merge_pauli_terms(
                raw_codes, raw_coefficients, tolerance
            )
            truncated_codes, truncated_coefficients = truncate_pauli_terms(
                merged_codes, merged_coefficients, limit
            )
            state_codes, state_coefficients = normalize_pauli_trace(
                truncated_codes, truncated_coefficients, tolerance
            )

    denominator_value = pauli_coefficient_overlap(
        state_codes, state_coefficients, state_codes, state_coefficients
    )
    if abs(denominator_value.imag) > 1e-12 * max(1.0, abs(denominator_value.real)):
        raise ValueError("the squared-state denominator is not real")
    denominator = float(denominator_value.real)
    numerator = observable_state_overlap(
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
SCICODE_GOLD_EOF
