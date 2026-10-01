#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def grating_channels(incident_sine, reciprocal_ratio, order_count):
    import numpy as np

    parameters = []
    for value in (incident_sine, reciprocal_ratio):
        raw = np.asarray(value)
        if raw.ndim != 0 or raw.dtype.kind not in "iuf" or not np.isfinite(raw):
            raise ValueError("Channel parameters must be finite real scalars.")
        parameters.append(float(raw))
    incident_sine, reciprocal_ratio = parameters
    if not -1 < incident_sine < 1 or reciprocal_ratio <= 0:
        raise ValueError("The incident channel and reciprocal ratio are invalid.")
    if isinstance(order_count, (bool, np.bool_)) or not isinstance(order_count, (int, np.integer)) or order_count < 1:
        raise ValueError("order_count must be a positive integer.")
    sines = incident_sine + reciprocal_ratio * np.arange(order_count)
    radicands = 1 - sines**2
    if not np.all(np.isfinite(radicands)) or np.any(np.abs(radicands) <= 1e-12):
        raise ValueError("Nonfinite or grazing diffraction channels are unsupported.")
    return sines, np.sqrt(radicands.astype(complex))

def reciprocal_fourier(coefficients, order_count):
    import numpy as np

    raw = np.asarray(coefficients)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iufc" or not np.all(np.isfinite(raw)):
        raise ValueError("Expected a nonempty finite numeric coefficient vector.")
    values = raw.astype(complex)
    if abs(values[0]) <= np.sum(np.abs(values[1:])):
        raise ValueError("The constant coefficient must strictly dominate the tail.")
    if isinstance(order_count, (bool, np.bool_)) or not isinstance(order_count, (int, np.integer)) or order_count < 1:
        raise ValueError("order_count must be a positive integer.")
    inverse = np.zeros(order_count, dtype=complex)
    inverse[0] = 1 / values[0]
    for degree in range(1, order_count):
        inverse[degree] = -sum(
            values[harmonic] * inverse[degree - harmonic]
            for harmonic in range(1, min(degree + 1, len(values)))
        ) / values[0]
    if not np.all(np.isfinite(inverse)):
        raise ValueError("The reciprocal coefficients are not finite.")
    return inverse

def bergmann_generator(sines, alpha_coefficients, beta_coefficients):
    import numpy as np

    sines = np.asarray(sines)
    if (
        sines.ndim != 1
        or sines.size == 0
        or sines.dtype.kind not in "iuf"
        or not np.all(np.isfinite(sines))
    ):
        raise ValueError("sines must be a nonempty finite real vector.")
    count = len(sines)
    alpha_inverse = reciprocal_fourier(alpha_coefficients, count)

    def convolution(values):
        values = np.asarray(values)
        if (
            values.ndim != 1
            or values.size == 0
            or values.dtype.kind not in "iufc"
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "Material coefficients must be finite numeric vectors."
            )
        matrix = np.zeros((count, count), dtype=complex)
        for harmonic, value in enumerate(values[:count]):
            columns = np.arange(count - harmonic)
            matrix[columns + harmonic, columns] = value
        return matrix

    alpha = convolution(alpha_coefficients)
    beta = convolution(beta_coefficients)
    inverse = convolution(alpha_inverse)
    lower = beta - sines[:, None] * inverse * sines[None, :]
    zero = np.zeros_like(alpha)
    result = np.block([[zero, alpha], [lower, zero]])
    if not np.all(np.isfinite(result)):
        raise ValueError("The generator is not finite.")
    return result

def ordered_transfer_series(generators, fractions):
    import numpy as np

    generators = np.asarray(generators)
    fractions = np.asarray(fractions)
    if (
        generators.ndim != 3
        or generators.shape[0] == 0
        or generators.shape[1] == 0
        or generators.shape[1] != generators.shape[2]
    ):
        raise ValueError("Expected a nonempty stack of square generators.")
    if generators.dtype.kind not in "iufc" or not np.all(
        np.isfinite(generators)
    ):
        raise ValueError("Generators must be finite numeric matrices.")
    if (
        fractions.shape != (generators.shape[0],)
        or fractions.dtype.kind not in "iuf"
        or not np.all(np.isfinite(fractions))
    ):
        raise ValueError(
            "Fractions must be a finite real vector, one per layer."
        )
    if np.any(fractions < 0) or abs(float(np.sum(fractions)) - 1) > 1e-12:
        raise ValueError("Nonnegative fractions must sum to one.")
    dimension = generators.shape[1]
    coefficients = np.zeros((3, dimension, dimension), dtype=complex)
    coefficients[0] = np.eye(dimension)
    for generator, fraction in zip(generators, fractions):
        first = 1j * fraction * generator
        coefficients[2] += first @ coefficients[1] + 0.5 * first @ first
        coefficients[1] += first
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("State coefficients are not finite.")
    return coefficients

def global_transfer_series(state_series, cosines):
    import numpy as np

    cosines = np.asarray(cosines)
    if (
        cosines.ndim != 1
        or cosines.size == 0
        or cosines.dtype.kind not in "iufc"
        or not np.all(np.isfinite(cosines))
    ):
        raise ValueError("Expected a finite numeric cosine vector.")
    cosines = cosines.astype(complex)
    branches = ((cosines.real > 0) & (cosines.imag == 0)) | (
        (cosines.real == 0) & (cosines.imag > 0)
    )
    if not np.all(branches) or np.any(np.abs(cosines) <= 1e-12):
        raise ValueError("Cosines must use nongrazing outgoing branches.")
    count = len(cosines)
    state_series = np.asarray(state_series)
    if (
        state_series.shape != (3, 2 * count, 2 * count)
        or state_series.dtype.kind not in "iufc"
        or not np.all(np.isfinite(state_series))
    ):
        raise ValueError("State coefficients have an invalid shape or values.")
    if not np.allclose(state_series[0], np.eye(2 * count), rtol=0, atol=1e-12):
        raise ValueError(
            "The constant state-transfer coefficient must be identity."
        )
    identity = np.eye(count)
    longitudinal = np.diag(cosines)
    basis = np.block([[identity, identity], [longitudinal, -longitudinal]])
    face_first = np.linalg.solve(basis, state_series[1] @ basis)
    face_second = np.linalg.solve(basis, state_series[2] @ basis)
    phase_first = np.diag(np.concatenate((-1j * cosines, 1j * cosines)))
    result = np.array(
        [
            np.eye(2 * count),
            phase_first + face_first,
            0.5 * phase_first @ phase_first
            + phase_first @ face_first
            + face_second,
        ]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Global coefficients are not finite.")
    return result

def scattering_series(global_series):
    import numpy as np

    global_series = np.asarray(global_series)
    if (
        global_series.ndim != 3
        or global_series.shape[0] != 3
        or global_series.shape[1] == 0
        or global_series.shape[1] != global_series.shape[2]
        or global_series.shape[1] % 2
    ):
        raise ValueError(
            "Expected three even-dimensional square coefficients."
        )
    if global_series.dtype.kind not in "iufc" or not np.all(
        np.isfinite(global_series)
    ):
        raise ValueError(
            "Transfer coefficients must be finite numeric values."
        )
    count = global_series.shape[1] // 2
    if not np.allclose(
        global_series[0], np.eye(2 * count), rtol=0, atol=1e-12
    ):
        raise ValueError("The constant transfer coefficient must be identity.")
    global_series = global_series.astype(np.complex128)
    first = global_series[1]
    second = global_series[2]
    incident = np.eye(count)[:, 0]
    result = np.zeros((2, 3, count), dtype=complex)
    result[1, 0] = incident
    result[0, 1] = -first[count:, :count] @ incident
    result[0, 2] = (
        first[count:, count:] @ first[count:, :count] - second[count:, :count]
    ) @ incident
    result[1, 1] = first[:count, :count] @ incident
    result[1, 2] = (
        second[:count, :count] - first[:count, count:] @ first[count:, :count]
    ) @ incident
    if not np.all(np.isfinite(result)):
        raise ValueError("Amplitude coefficients are not finite.")
    return result

def diffraction_efficiencies(amplitude_series, cosines, thickness):
    import numpy as np

    raw_thickness = np.asarray(thickness)
    if (
        raw_thickness.ndim != 0
        or raw_thickness.dtype.kind not in "iuf"
        or not np.isfinite(raw_thickness)
        or raw_thickness < 0
    ):
        raise ValueError("thickness must be a finite nonnegative real scalar.")
    thickness = float(raw_thickness)
    cosines = np.asarray(cosines)
    if (
        cosines.ndim != 1
        or cosines.size == 0
        or cosines.dtype.kind not in "iufc"
        or not np.all(np.isfinite(cosines))
    ):
        raise ValueError("Expected a finite numeric cosine vector.")
    cosines = cosines.astype(complex)
    branches = ((cosines.real > 0) & (cosines.imag == 0)) | (
        (cosines.real == 0) & (cosines.imag > 0)
    )
    if (
        not np.all(branches)
        or np.any(np.abs(cosines) <= 1e-12)
        or cosines[0].real <= 0
        or cosines[0].imag != 0
    ):
        raise ValueError(
            "Expected outgoing branches and a propagating incident order."
        )
    amplitude_series = np.asarray(amplitude_series)
    if (
        amplitude_series.shape != (2, 3, len(cosines))
        or amplitude_series.dtype.kind not in "iufc"
        or not np.all(np.isfinite(amplitude_series))
    ):
        raise ValueError(
            "Amplitude coefficients have invalid shape or values."
        )
    amplitude_series = amplitude_series.astype(np.complex128)
    amplitudes = (
        amplitude_series[:, 2] * thickness + amplitude_series[:, 1]
    ) * thickness + amplitude_series[:, 0]
    powers = np.abs(amplitudes) ** 2 * (cosines.real / cosines[0].real)
    if not np.all(np.isfinite(powers)):
        raise ValueError("The resulting efficiencies are not finite.")
    return powers

def layer_reversal_contrast(
    materials,
    fractions,
    incident_sine,
    reciprocal_ratio,
    thickness,
    order_count,
):
    import numpy as np

    materials = np.asarray(materials)
    fractions = np.asarray(fractions)
    if (
        materials.ndim != 3
        or materials.shape[:2] != (2, 2)
        or materials.shape[2] == 0
        or materials.dtype.kind not in "iufc"
        or not np.all(np.isfinite(materials))
    ):
        raise ValueError("materials must have finite numeric shape (2,2,H).")
    if fractions.shape != (2,):
        raise ValueError(
            "Expected one thickness fraction for each of two layers."
        )

    sines, cosines = grating_channels(
        incident_sine, reciprocal_ratio, order_count
    )
    if sines[-1] + reciprocal_ratio <= 1:
        raise ValueError("All open nonnegative orders must be retained.")

    nonspecular = []
    for ordering in (np.array([0, 1]), np.array([1, 0])):
        average_power = 0.0
        for alpha_index, beta_index in ((1, 0), (0, 1)):
            generators = np.array([
                bergmann_generator(
                    sines,
                    materials[layer, alpha_index],
                    materials[layer, beta_index],
                )
                for layer in ordering
            ])
            state = ordered_transfer_series(
                generators, fractions[ordering]
            )
            transfer = global_transfer_series(state, cosines)
            amplitudes = scattering_series(transfer)
            powers = diffraction_efficiencies(
                amplitudes, cosines, thickness
            )
            average_power += 0.5 * float(np.sum(powers[1, 1:]))
        nonspecular.append(average_power)

    denominator = nonspecular[0] + nonspecular[1]
    if denominator == 0:
        return 0.0
    return float((nonspecular[0] - nonspecular[1]) / denominator)
SCICODE_GOLD_EOF
