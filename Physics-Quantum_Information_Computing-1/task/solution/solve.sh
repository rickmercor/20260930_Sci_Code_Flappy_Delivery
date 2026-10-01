#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_central_difference_coefficients(
    degree: int,
    max_order: int,
) -> np.ndarray:
    import math
    from fractions import Fraction
    from numbers import Integral

    import numpy as np

    if isinstance(degree, bool) or not isinstance(degree, Integral) or int(degree) < 1:
        raise ValueError("degree must be an integer >= 1")
    if (
        isinstance(max_order, bool)
        or not isinstance(max_order, Integral)
        or int(max_order) < 1
        or int(max_order) > 2 * int(degree)
    ):
        raise ValueError("max_order must be an integer in [1, 2 * degree]")

    degree = int(degree)
    max_order = int(max_order)
    nodes = list(range(-degree, degree + 1))
    rows = np.zeros((max_order, 2 * degree + 1), dtype=np.float64)

    # Build each Lagrange cardinal polynomial exactly in ascending powers.
    for column, node in enumerate(nodes):
        polynomial = [Fraction(1, 1)]
        for other in nodes:
            if other == node:
                continue
            denominator = node - other
            updated = [Fraction(0, 1)] * (len(polynomial) + 1)
            for power, coefficient in enumerate(polynomial):
                updated[power] += -Fraction(other, denominator) * coefficient
                updated[power + 1] += Fraction(1, denominator) * coefficient
            polynomial = updated

        for derivative_order in range(1, max_order + 1):
            rows[derivative_order - 1, column] = float(
                math.factorial(derivative_order) * polynomial[derivative_order]
            )

    return rows

def compute_msd_preprocessing(
    first_derivative_coefficients: np.ndarray,
    krylov_dimension: int,
    degree: int,
    total_shots: float,
    sector_minimum: float,
    sector_maximum: float,
) -> np.ndarray:
    import math
    from numbers import Integral, Real

    import numpy as np

    if isinstance(krylov_dimension, bool) or not isinstance(krylov_dimension, Integral) or int(krylov_dimension) < 1:
        raise ValueError("krylov_dimension must be an integer >= 1")
    if isinstance(degree, bool) or not isinstance(degree, Integral) or int(degree) < 1:
        raise ValueError("degree must be an integer >= 1")
    if isinstance(total_shots, bool) or not isinstance(total_shots, Real):
        raise ValueError("total_shots must be a positive finite real scalar")
    if isinstance(sector_minimum, bool) or not isinstance(sector_minimum, Real):
        raise ValueError("sector_minimum must be finite")
    if isinstance(sector_maximum, bool) or not isinstance(sector_maximum, Real):
        raise ValueError("sector_maximum must be finite")

    degree = int(degree)
    n = int(krylov_dimension)
    shots = float(total_shots)
    e_min = float(sector_minimum)
    e_max = float(sector_maximum)
    if not math.isfinite(shots) or shots <= 0.0:
        raise ValueError("total_shots must be > 0 and finite")
    if not math.isfinite(e_min) or not math.isfinite(e_max) or not e_min < e_max:
        raise ValueError("sector bounds must be finite and strictly ordered")

    raw = np.asarray(first_derivative_coefficients)
    if raw.ndim != 1 or raw.size != 2 * degree + 1:
        raise ValueError("coefficient length must equal 2 * degree + 1")
    if np.iscomplexobj(raw) and np.any(np.abs(np.imag(raw)) > 0.0):
        raise ValueError("first-derivative coefficients must be real")
    try:
        coefficients = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("first-derivative coefficients must be real numeric values") from exc
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("first-derivative coefficients must be finite")

    half_min = 0.5 * e_min
    half_max = 0.5 * e_max
    energy_shift = half_min + half_max
    shifted_norm = half_max - half_min
    if not math.isfinite(energy_shift) or not math.isfinite(shifted_norm) or shifted_norm <= 0.0:
        raise ValueError("the centered sector quantities must be finite")

    nodes = np.arange(-degree, degree + 1, dtype=np.float64)
    coefficient_norm = float(np.sum(np.abs(coefficients)))
    alpha_nj = 2.0 * n * math.sqrt(2.0 * math.log(2.0 * n)) * coefficient_norm
    factorial = float(math.factorial(2 * degree + 1))
    beta_nj = n * float(np.sum(np.abs(coefficients * nodes ** (2 * degree + 1)))) / factorial
    if not math.isfinite(alpha_nj) or not math.isfinite(beta_nj) or alpha_nj <= 0.0 or beta_nj <= 0.0:
        raise ValueError("the supplied coefficients do not define positive finite balance constants")

    exponent = 2 * degree + 1
    log_delta = (
        math.log(alpha_nj)
        - math.log(2.0 * degree)
        - math.log(beta_nj)
        - exponent * math.log(shifted_norm)
        - 0.5 * math.log(shots)
    ) / exponent
    try:
        delta_t = math.exp(log_delta)
    except OverflowError as exc:
        raise ValueError("optimized_delta_t is not representable") from exc
    if not math.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("optimized_delta_t must be positive and finite")

    return np.array([energy_shift, shifted_norm, delta_t, alpha_nj, beta_nj], dtype=np.float64)

def build_shifted_propagators(
    g_nonnegative: np.ndarray,
    krylov_dimension: int,
    degree: int,
    stride: int,
) -> np.ndarray:
    from numbers import Integral

    import numpy as np

    for value, name in ((krylov_dimension, "krylov_dimension"), (degree, "degree"), (stride, "stride")):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    n = int(krylov_dimension)
    degree = int(degree)
    stride = int(stride)

    try:
        samples = np.asarray(g_nonnegative, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("g_nonnegative must be a one-dimensional numeric array") from exc
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("g_nonnegative must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(samples.real)) or not np.all(np.isfinite(samples.imag)):
        raise ValueError("g_nonnegative must contain finite complex values")

    largest_index = stride * (n - 1) + degree
    if samples.size <= largest_index:
        raise ValueError("g_nonnegative is too short for the requested matrix indices")

    result = np.empty((2 * degree + 1, n, n), dtype=np.complex128)
    for shift_position, shift in enumerate(range(-degree, degree + 1)):
        for row in range(n):
            for column in range(n):
                signed_index = stride * (column - row) + shift
                if signed_index >= 0:
                    value = samples[signed_index]
                else:
                    value = np.conjugate(samples[-signed_index])
                result[shift_position, row, column] = value
    return result

def reconstruct_projected_matrices(
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

def solve_resolved_krylov_ground_state(
    overlap: np.ndarray,
    shifted_hamiltonian: np.ndarray,
    relative_cutoff: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    import math
    from numbers import Real

    import numpy as np

    if isinstance(relative_cutoff, bool) or not isinstance(relative_cutoff, Real):
        raise ValueError("relative_cutoff must be a finite real scalar in (0, 1)")
    cutoff = float(relative_cutoff)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < 1.0:
        raise ValueError("relative_cutoff must lie strictly between 0 and 1")

    try:
        s = np.asarray(overlap, dtype=np.complex128)
        h = np.asarray(shifted_hamiltonian, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("overlap and shifted_hamiltonian must be numeric matrices") from exc
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 1 or h.shape != s.shape:
        raise ValueError("both matrices must have the same nonempty square shape")
    if not np.all(np.isfinite(s.real)) or not np.all(np.isfinite(s.imag)):
        raise ValueError("overlap must be finite")
    if not np.all(np.isfinite(h.real)) or not np.all(np.isfinite(h.imag)):
        raise ValueError("shifted_hamiltonian must be finite")
    if not np.allclose(s, s.conjugate().T, rtol=0.0, atol=1.0e-10):
        raise ValueError("overlap must be Hermitian within atol=1e-10")
    if not np.allclose(h, h.conjugate().T, rtol=0.0, atol=1.0e-10):
        raise ValueError("shifted_hamiltonian must be Hermitian within atol=1e-10")

    s = 0.5 * (s + s.conjugate().T)
    h = 0.5 * (h + h.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    s_max = float(eigenvalues[-1])
    if not math.isfinite(s_max) or s_max <= 0.0:
        raise ValueError("overlap must possess a positive resolved eigenvalue")
    retained = np.flatnonzero(eigenvalues > cutoff * s_max)
    if retained.size == 0 or np.any(eigenvalues[retained] <= 0.0):
        raise ValueError("the strict cutoff leaves no positive resolved subspace")

    transform = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])[None, :]
    reduced_hamiltonian = transform.conjugate().T @ h @ transform
    reduced_hamiltonian = 0.5 * (reduced_hamiltonian + reduced_hamiltonian.conjugate().T)
    energies, vectors = np.linalg.eigh(reduced_hamiltonian)
    state = transform @ vectors[:, 0]
    metric_norm_squared = np.vdot(state, s @ state)
    metric_norm_real = float(np.real(metric_norm_squared))
    metric_norm_tolerance = 1.0e-10 * max(1.0, abs(metric_norm_real))
    if (
        not np.isfinite(metric_norm_squared.real)
        or not np.isfinite(metric_norm_squared.imag)
        or abs(float(np.imag(metric_norm_squared))) > metric_norm_tolerance
        or metric_norm_real <= 0.0
    ):
        raise ValueError("the ground-state coefficient vector has invalid overlap norm")
    state = state / math.sqrt(metric_norm_real)
    pivot = int(np.argmax(np.abs(state)))
    if abs(state[pivot]) > 0.0:
        state = state * np.exp(-1j * np.angle(state[pivot]))
    state = np.asarray(state, dtype=np.complex128)

    return float(energies[0]), state, eigenvalues.astype(np.float64), retained.astype(np.int64)

def compute_projected_hamiltonian_moments(
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

def run_moment_lanczos(
    moments: np.ndarray,
    max_order: int,
    energy_shift: float,
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray, int]:
    import math
    from numbers import Integral, Real

    import numpy as np

    if isinstance(max_order, bool) or not isinstance(max_order, Integral) or int(max_order) < 1:
        raise ValueError("max_order must be an integer >= 1")
    if isinstance(energy_shift, bool) or not isinstance(energy_shift, Real):
        raise ValueError("energy_shift must be finite")
    shift = float(energy_shift)
    if not math.isfinite(shift):
        raise ValueError("energy_shift must be finite")

    raw = np.asarray(moments)
    if raw.ndim != 1:
        raise ValueError("moments must be one-dimensional")
    if np.iscomplexobj(raw) and np.any(np.abs(np.imag(raw)) > 0.0):
        raise ValueError("moments must be real")
    try:
        mu = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("moments must be real numeric values") from exc
    order_limit = int(max_order)
    if mu.size < 2 * order_limit + 1:
        raise ValueError("moments must include mu_0 through mu_(2 * max_order)")
    if not np.all(np.isfinite(mu)):
        raise ValueError("moments must be finite")
    if abs(float(mu[0]) - 1.0) > 1.0e-12:
        raise ValueError("mu_0 must equal 1 within 1e-12")

    l_cache = {-1: 1.0, 0: 1.0}
    m_cache = {-1: 0.0, 0: float(mu[1])}

    def l_determinant(index: int) -> float:
        if index in l_cache:
            return l_cache[index]
        matrix = np.array(
            [[mu[row + column] for column in range(index + 1)] for row in range(index + 1)],
            dtype=np.float64,
        )
        value = float(np.linalg.det(matrix))
        l_cache[index] = value
        return value

    def m_determinant(index: int) -> float:
        if index in m_cache:
            return m_cache[index]
        exponents = list(range(index)) + [index + 1]
        matrix = np.array(
            [[mu[row + exponent] for exponent in exponents] for row in range(index + 1)],
            dtype=np.float64,
        )
        value = float(np.linalg.det(matrix))
        m_cache[index] = value
        return value

    alpha_values = []
    beta_values = []
    beta_squared = []
    trial_energies = []
    accepted_order = 0
    previous_shifted_energy = None

    for order in range(1, order_limit + 1):
        l_previous = l_determinant(order - 1)
        l_before_previous = l_determinant(order - 2)
        if l_previous == 0.0 or l_before_previous == 0.0:
            raise ValueError("a determinant denominator is zero before a defined stop")
        alpha = m_determinant(order - 1) / l_previous - m_determinant(order - 2) / l_before_previous
        if not math.isfinite(alpha):
            raise ValueError("a Lanczos diagonal coefficient is not finite")
        alpha_values.append(float(alpha))

        tridiagonal = np.diag(np.asarray(alpha_values, dtype=np.float64))
        for index, beta in enumerate(beta_values):
            tridiagonal[index, index + 1] = beta
            tridiagonal[index + 1, index] = beta
        shifted_energy = float(np.linalg.eigvalsh(tridiagonal)[0])
        trial_energy = shifted_energy + shift
        trial_energies.append(trial_energy)

        if previous_shifted_energy is not None and shifted_energy > previous_shifted_energy:
            return (
                float(previous_shifted_energy + shift),
                int(accepted_order),
                np.asarray(trial_energies, dtype=np.float64),
                np.asarray(beta_squared, dtype=np.float64),
                np.asarray(alpha_values, dtype=np.float64),
                2,
            )

        previous_shifted_energy = shifted_energy
        accepted_order = order
        if order < order_limit:
            denominator = l_previous * l_previous
            if denominator == 0.0 or not math.isfinite(denominator):
                raise ValueError("a squared determinant denominator is singular before a defined stop")
            beta_square = l_determinant(order) * l_before_previous / denominator
            if not math.isfinite(beta_square):
                raise ValueError("a squared Lanczos off-diagonal coefficient is not finite")
            beta_squared.append(float(beta_square))
            if beta_square < 0.0:
                return (
                    float(shifted_energy + shift),
                    int(accepted_order),
                    np.asarray(trial_energies, dtype=np.float64),
                    np.asarray(beta_squared, dtype=np.float64),
                    np.asarray(alpha_values, dtype=np.float64),
                    1,
                )
            if beta_square == 0.0:
                return (
                    float(shifted_energy + shift),
                    int(accepted_order),
                    np.asarray(trial_energies, dtype=np.float64),
                    np.asarray(beta_squared, dtype=np.float64),
                    np.asarray(alpha_values, dtype=np.float64),
                    3,
                )
            beta_values.append(math.sqrt(beta_square))

    return (
        float(previous_shifted_energy + shift),
        int(accepted_order),
        np.asarray(trial_energies, dtype=np.float64),
        np.asarray(beta_squared, dtype=np.float64),
        np.asarray(alpha_values, dtype=np.float64),
        0,
    )

def run_msd_energy_pipeline(
    g_nonnegative: np.ndarray,
    krylov_dimension: int,
    degree: int,
    stride: int,
    total_shots: float,
    sector_minimum: float,
    sector_maximum: float,
    relative_cutoff: float,
) -> float:
    import numpy as np

    coefficients = compute_central_difference_coefficients(degree, 2 * degree)
    preprocessing = compute_msd_preprocessing(
        coefficients[0],
        krylov_dimension,
        degree,
        total_shots,
        sector_minimum,
        sector_maximum,
    )
    propagators = build_shifted_propagators(
        g_nonnegative,
        krylov_dimension,
        degree,
        stride,
    )
    overlap, power_matrices = reconstruct_projected_matrices(
        propagators,
        coefficients,
        preprocessing[2],
    )
    _, state_coefficients, _, _ = solve_resolved_krylov_ground_state(
        overlap,
        power_matrices[0],
        relative_cutoff,
    )
    moments = compute_projected_hamiltonian_moments(
        power_matrices,
        overlap,
        state_coefficients,
    )
    final_energy, _, _, _, _, _ = run_moment_lanczos(
        moments,
        degree,
        preprocessing[0],
    )
    if not np.isfinite(final_energy):
        raise ValueError("the final mitigated energy must be finite")
    return float(final_energy)
SCICODE_GOLD_EOF
