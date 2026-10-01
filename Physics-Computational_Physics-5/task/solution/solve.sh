#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def evaluate_kinetic_response(zeta: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _faddeeva(argument, terms=48):
        """Weideman rational evaluation of the Faddeeva function."""
        half = 2 * terms
        full = 2 * half
        index = np.arange(-half + 1, half)
        scale = np.sqrt(terms / np.sqrt(2.0))
        node = scale * np.tan(index * np.pi / full)
        weight = np.exp(-node ** 2) * (scale ** 2 + node ** 2)
        coefficients = np.real(np.fft.fft(np.fft.fftshift(
            np.append(0.0, weight)))) / full
        coefficients = np.flipud(coefficients[1:terms + 1])
        # The rational form converges only above the real axis; below it the
        # reflection w(z) + w(-z) = 2 exp(-z**2) supplies the continuation.
        upper = argument.imag >= 0.0
        folded = np.where(upper, argument, -argument)
        mapped = (scale + 1j * folded) / (scale - 1j * folded)
        series = np.polyval(coefficients, mapped)
        value = (2.0 * series / (scale - 1j * folded) ** 2
                 + (1.0 / np.sqrt(np.pi)) / (scale - 1j * folded))
        return np.where(upper, value, 2.0 * np.exp(-folded ** 2) - value)

    speeds = np.asarray(zeta, dtype=complex)
    if speeds.size < 1:
        raise ValueError("zeta must hold at least one phase speed")
    if not np.all(np.isfinite(speeds)):
        raise ValueError("zeta must contain only finite entries")

    # The plasma dispersion function is the Faddeeva function up to the factor
    # i*sqrt(pi); the density response is one plus the phase speed times it.
    dispersion = 1j * np.sqrt(np.pi) * _faddeeva(speeds.ravel())
    response = 1.0 + speeds.ravel() * dispersion
    return response.reshape(speeds.shape)

def solve_kinetic_root(wavenumber: float) -> complex:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _faddeeva(argument, terms=48):
        """Weideman rational evaluation of the Faddeeva function."""
        half = 2 * terms
        full = 2 * half
        index = np.arange(-half + 1, half)
        scale = np.sqrt(terms / np.sqrt(2.0))
        node = scale * np.tan(index * np.pi / full)
        weight = np.exp(-node ** 2) * (scale ** 2 + node ** 2)
        coefficients = np.real(np.fft.fft(np.fft.fftshift(
            np.append(0.0, weight)))) / full
        coefficients = np.flipud(coefficients[1:terms + 1])
        upper = argument.imag >= 0.0
        folded = np.where(upper, argument, -argument)
        mapped = (scale + 1j * folded) / (scale - 1j * folded)
        series = np.polyval(coefficients, mapped)
        value = (2.0 * series / (scale - 1j * folded) ** 2
                 + (1.0 / np.sqrt(np.pi)) / (scale - 1j * folded))
        return np.where(upper, value, 2.0 * np.exp(-folded ** 2) - value)

    # Below a tenth of the Debye wave number the least-damped root is damped by
    # an exponentially small amount that double precision cannot resolve, and
    # the tracking returns a spurious root, so the domain is closed there.
    if not (isinstance(wavenumber, (int, float, np.floating, np.integer))
            and not isinstance(wavenumber, bool) and np.isfinite(wavenumber)
            and float(wavenumber) >= 0.1):
        raise ValueError("wavenumber must be a finite number of at least 0.1")
    wavenumber = float(wavenumber)

    def _response(point):
        speed = np.asarray([point], dtype=complex)
        dispersion = 1j * np.sqrt(np.pi) * _faddeeva(speed)
        return complex(1.0 + speed[0] * dispersion[0]), complex(dispersion[0])

    # The long-wavelength limit is where the least-damped branch is easiest to
    # identify, so the root is tracked there from the Bohm-Gross estimate and
    # then continued in the wave number up to the requested one.
    start = min(wavenumber, 0.05)
    grid = np.linspace(start, wavenumber,
                       max(2, int(np.ceil(wavenumber / 0.02)) + 1))
    point = (np.sqrt(1.0 + 3.0 * start ** 2) / (np.sqrt(2.0) * start)) - 0.001j
    for step_wavenumber in grid:
        offset = step_wavenumber ** 2
        for _ in range(200):
            response, dispersion = _response(point)
            # d/dzeta of the response, using Z'(zeta) = -2 (1 + zeta Z(zeta)).
            slope = dispersion - 2.0 * point * response
            if slope == 0.0:
                break
            correction = (response + offset) / slope
            point = point - correction
            if abs(correction) <= 1.0e-15 * max(1.0, abs(point)):
                break
    return complex(point)

def solve_matched_pade_coefficients(phase_speeds: np.ndarray,
                                            response_values: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    speeds = np.asarray(phase_speeds, dtype=complex)
    values = np.asarray(response_values, dtype=complex)
    for name, array in (("phase_speeds", speeds), ("response_values", values)):
        if array.shape != (2,):
            raise ValueError(f"{name} must be an array of shape (2,)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
    if speeds[0] == speeds[1]:
        raise ValueError("phase_speeds must hold two distinct phase speeds")

    # The approximant is (1 + a z) / (1 + b z + c z**2 - 2 a z**3) with c = -2.
    # Clearing the denominator against the target value R at the anchor z makes
    # the condition linear in a and b:
    #     a (z + 2 R z**3) - b (R z) = -(1 - R - c R z**2).
    quadratic = -2.0 + 0.0j
    matrix = np.empty((2, 2), dtype=complex)
    matrix[:, 0] = speeds + 2.0 * values * speeds ** 3
    matrix[:, 1] = -values * speeds
    rhs = -(1.0 - values - quadratic * values * speeds ** 2)

    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    scale = np.max(np.abs(matrix))
    if not np.isfinite(determinant) or abs(determinant) <= 1.0e-13 * max(scale ** 2, 1.0):
        raise ValueError("the anchoring conditions do not determine the coefficients")

    numerator, linear = np.linalg.solve(matrix, rhs)
    return np.array([numerator, linear, quadratic], dtype=complex)

def build_asymptotic_pade_coefficients(label: str) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not isinstance(label, str):
        raise ValueError("label must be one of 'R30', 'R31' or 'HP'")

    root_pi = np.sqrt(np.pi)
    if label == "R30":
        # Three adiabatic orders matched; the quadratic coefficient is free and
        # is not minus two, so this member alone keeps an electrostatic term in
        # the heat flux.
        numerator = -1j * root_pi * (np.pi - 3.0) / (4.0 - np.pi)
        linear = -1j * root_pi / (4.0 - np.pi)
        quadratic = -(3.0 * np.pi - 8.0) / (4.0 - np.pi) + 0.0j
    elif label == "R31":
        # One adiabatic order traded for one fluid order, which forces the
        # quadratic coefficient to minus two.
        numerator = -1j * (4.0 - np.pi) / root_pi
        linear = -4.0j / root_pi
        quadratic = -2.0 + 0.0j
    elif label == "HP":
        # A second adiabatic order traded for a second fluid order; this is the
        # Hammett-Perkins member.
        numerator = -1j * root_pi / 2.0
        linear = -3.0j * root_pi / 2.0
        quadratic = -2.0 + 0.0j
    else:
        raise ValueError("label must be one of 'R30', 'R31' or 'HP'")

    return np.array([numerator, linear, quadratic], dtype=complex)

def evaluate_closure_parameters(pade_coefficients: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coefficients = np.asarray(pade_coefficients, dtype=complex)
    if coefficients.shape != (3,):
        raise ValueError("pade_coefficients must be an array of shape (3,)")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("pade_coefficients must contain only finite entries")

    numerator, linear, quadratic = (complex(value) for value in coefficients)
    if numerator == 0.0:
        raise ValueError("the numerator coefficient must not vanish")

    # Eliminating the heat flux between the linearised pressure equation and the
    # approximated dispersion relation leaves these three combinations.
    velocity = (linear - 3.0 * numerator) / numerator
    potential = (1.0 + quadratic / 2.0) / (1j * numerator)
    temperature = 1.0 / (1j * numerator)

    parameters = np.array([velocity, potential, temperature], dtype=complex)
    tolerance = 1.0e-6 * np.maximum(1.0, np.abs(parameters))
    if np.any(np.abs(parameters.imag) > tolerance):
        raise ValueError("the closure parameters must be real to within 1e-6")
    return parameters.real.astype(float)

def build_moment_evolution_matrix(wavenumber: float,
                                          closure_parameters: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not (isinstance(wavenumber, (int, float, np.floating, np.integer))
            and not isinstance(wavenumber, bool) and np.isfinite(wavenumber)
            and float(wavenumber) > 0.0):
        raise ValueError("wavenumber must be a finite number greater than zero")
    wavenumber = float(wavenumber)

    parameters = np.asarray(closure_parameters)
    if np.iscomplexobj(parameters):
        raise ValueError("closure_parameters must be a real array")
    parameters = parameters.astype(float)
    if parameters.shape != (3,):
        raise ValueError("closure_parameters must be an array of shape (3,)")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("closure_parameters must contain only finite entries")

    velocity, potential, temperature = (float(value) for value in parameters)
    root_two = np.sqrt(2.0)
    matrix = np.zeros((3, 3), dtype=complex)

    # Continuity: the density is driven by the compression of the velocity.
    matrix[0, 1] = -1j * root_two * wavenumber
    # Momentum: the pressure gradient and, through the Poisson equation, the
    # space-charge restoring force, which is the term that carries the inverse
    # wave number.
    matrix[1, 0] = -1j / (root_two * wavenumber)
    matrix[1, 2] = -1j * wavenumber / root_two
    # Pressure: adiabatic compression plus the divergence of the closed heat
    # flux. The potential is eliminated in favour of the density there too, so
    # the potential-corrected pressure coefficient also reaches the density
    # column, divided by the wave number.
    matrix[2, 0] = root_two * (potential / wavenumber + wavenumber * temperature)
    matrix[2, 1] = -1j * root_two * wavenumber * (3.0 + velocity)
    matrix[2, 2] = root_two * wavenumber * (potential - temperature)
    return matrix

def integrate_moment_history(evolution_matrix: np.ndarray,
                                     initial_state: np.ndarray,
                                     time_step: float,
                                     step_count: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    matrix = np.asarray(evolution_matrix, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("evolution_matrix must be an array of shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("evolution_matrix must contain only finite entries")

    state = np.asarray(initial_state, dtype=complex)
    if state.shape != (3,):
        raise ValueError("initial_state must be an array of shape (3,)")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite entries")

    if not (isinstance(time_step, (int, float, np.floating, np.integer))
            and not isinstance(time_step, bool) and np.isfinite(time_step)
            and float(time_step) > 0.0):
        raise ValueError("time_step must be a finite number greater than zero")
    time_step = float(time_step)

    if (isinstance(step_count, bool)
            or not isinstance(step_count, (int, np.integer))
            or int(step_count) < 1):
        raise ValueError("step_count must be an integer greater than zero")
    step_count = int(step_count)

    history = np.empty((step_count + 1, 3), dtype=complex)
    history[0] = state
    for level in range(step_count):
        trial = matrix @ state
        state = state + time_step * (matrix @ (state + 0.5 * time_step * trial))
        history[level + 1] = state
    return history

def compute_field_amplitude_history(moment_history: np.ndarray,
                                            wavenumber: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    history = np.asarray(moment_history, dtype=complex)
    if history.ndim != 2 or history.shape[1] != 3 or history.shape[0] < 1:
        raise ValueError("moment_history must be two-dimensional with three columns")
    if not np.all(np.isfinite(history)):
        raise ValueError("moment_history must contain only finite entries")

    if not (isinstance(wavenumber, (int, float, np.floating, np.integer))
            and not isinstance(wavenumber, bool) and np.isfinite(wavenumber)
            and float(wavenumber) > 0.0):
        raise ValueError("wavenumber must be a finite number greater than zero")
    wavenumber = float(wavenumber)

    # Poisson ties the normalised potential to minus the density divided by the
    # squared wave number; the field is minus the gradient of the potential, so
    # in Fourier space it is the density divided by the wave number, rotated by
    # a quarter turn.
    return 1j * history[:, 0] / wavenumber

def compute_relative_field_deviation(reference_history: np.ndarray,
                                             test_history: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    reference = np.asarray(reference_history, dtype=complex)
    test = np.asarray(test_history, dtype=complex)
    for name, array in (("reference_history", reference), ("test_history", test)):
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
    if reference.shape != test.shape:
        raise ValueError("the two histories must have the same shape")

    # The level count cancels between the two mean squares, so the measure is
    # the ratio of the two accumulated squared magnitudes.
    reference_power = float(np.sum(np.abs(reference) ** 2))
    if reference_power <= 0.0:
        raise ValueError("reference_history must not vanish at every level")
    difference_power = float(np.sum(np.abs(test - reference) ** 2))
    return float(np.sqrt(difference_power / reference_power))

def run_closure_fidelity_pipeline(wavenumbers: np.ndarray = (0.2, 0.3, 0.4, 0.5, 0.6),
                                          amplitude: float = 0.02,
                                          time_step: float = 0.005,
                                          step_count: int = 8000,
                                          benchmark: str = "HP") -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    # -- Validate the orchestrator inputs.
    band = np.asarray(wavenumbers, dtype=float)
    if band.ndim != 1 or band.size < 1:
        raise ValueError("wavenumbers must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(band)) or np.any(band < 0.1):
        raise ValueError("wavenumbers entries must be finite and at least 0.1")
    for name, value in (("amplitude", amplitude), ("time_step", time_step)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)
                and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number greater than zero")
    if (isinstance(step_count, bool)
            or not isinstance(step_count, (int, np.integer))
            or int(step_count) < 1):
        raise ValueError("step_count must be an integer greater than zero")
    if benchmark not in ("HP", "R31", "R30"):
        raise ValueError("benchmark must be one of 'HP', 'R31' or 'R30'")

    amplitude = float(amplitude)
    time_step = float(time_step)
    step_count = int(step_count)

    # -- The reference pipeline calls the  twin of every earlier step, so
    #    that it stays independent of the submitted public functions. Chaining
    #    through the public names instead would put the same defect on both sides
    #    of the comparison and let an incorrect submission pass. The public chain
    #    is graded separately by integration test 3, which composes the public
    #    steps 01-09 by hand and checks them against this reference.

    # -- Sub-problem 04: the conventional closure is wave-number independent, so
    #    its parameters are fixed once for the whole band.
    asymptotic = build_asymptotic_pade_coefficients(benchmark)
    # -- Sub-problem 05.
    benchmark_parameters = evaluate_closure_parameters(asymptotic)

    # -- The cosine density ripple of a Maxwellian carries an equal pressure
    #    ripple and no flow.
    initial_state = np.array([amplitude, 0.0, amplitude], dtype=complex)

    deviations = []
    for wavenumber in band:
        wavenumber = float(wavenumber)

        # -- Sub-problem 02: the least-damped kinetic root, and its mirror image
        #    across the imaginary axis, which the Maxwellian parity supplies.
        root = solve_kinetic_root(wavenumber)
        phase_speeds = np.array([root, -np.conj(root)], dtype=complex)

        # -- Sub-problem 01: the exact response at those two phase speeds, which
        #    is what the approximant is anchored on. On a genuine root it must
        #    equal minus the squared wave number.
        response_values = evaluate_kinetic_response(phase_speeds)
        residual = float(np.max(np.abs(np.asarray(response_values, dtype=complex)
                                       + wavenumber ** 2)))
        if not np.isfinite(residual) or residual > 1.0e-8:
            raise ValueError("the located phase speeds do not satisfy the "
                             "dispersion relation to within 1e-8")

        # -- Sub-problems 03 and 05: the anchored closure at this wave number.
        matched = solve_matched_pade_coefficients(phase_speeds, response_values)
        matched_parameters = evaluate_closure_parameters(matched)

        # -- Sub-problems 06, 07 and 08 for each closure in turn.
        reference_field = compute_field_amplitude_history(
            integrate_moment_history(
                build_moment_evolution_matrix(wavenumber, matched_parameters),
                initial_state, time_step, step_count),
            wavenumber)
        test_field = compute_field_amplitude_history(
            integrate_moment_history(
                build_moment_evolution_matrix(wavenumber, benchmark_parameters),
                initial_state, time_step, step_count),
            wavenumber)

        # -- Sub-problem 09.
        deviations.append(float(compute_relative_field_deviation(reference_field,
                                                                         test_field)))

    return float(np.sqrt(np.mean(np.asarray(deviations, dtype=float) ** 2)))
SCICODE_GOLD_EOF
