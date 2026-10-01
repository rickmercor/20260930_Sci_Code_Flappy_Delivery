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


def _finite_scalar(value, name):
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _real_array(value, shape, name):
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(array, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has an invalid shape or nonfinite entries")
    return array


def trace_shell_segments(
    radii_km: "np.ndarray",
    cos_zenith: float,
    detector_depth_km: float,
    subdivisions: int = 1,
) -> "np.ndarray":
    raw = np.asarray(radii_km)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("radii must be a nonempty vector")
    radii = _real_array(radii_km, raw.shape, "radii_km")
    direction = _finite_scalar(cos_zenith, "cos_zenith")
    depth = _finite_scalar(detector_depth_km, "detector_depth_km")
    if (
        np.any(radii <= 0.0)
        or np.any(np.diff(radii) <= 0.0)
        or not -1.0 <= direction <= 1.0
        or not 0.0 <= depth < radii[-1]
    ):
        raise ValueError("invalid spherical geometry")
    if (
        isinstance(subdivisions, (bool, np.bool_))
        or not isinstance(subdivisions, (int, np.integer))
        or subdivisions < 1
    ):
        raise ValueError("subdivisions must be a positive integer")
    detector_radius = radii[-1] - depth
    impact_sq = detector_radius**2 * (1.0 - direction**2)
    shift = -detector_radius * direction
    distance = shift + np.sqrt(max(radii[-1] ** 2 - impact_sq, 0.0))
    if distance == 0.0 or (depth == 0.0 and direction >= 0.0):
        return np.empty((0, 3))
    cuts = [0.0, float(distance)]
    for radius in radii[:-1]:
        discriminant = radius * radius - impact_sq
        roundoff = 32.0 * np.finfo(float).eps * max(radius * radius, impact_sq)
        if abs(discriminant) <= roundoff:
            discriminant = 0.0
        if discriminant >= 0.0:
            root = np.sqrt(discriminant)
            for crossing in (shift - root, shift + root):
                if 0.0 < crossing < distance:
                    cuts.append(float(crossing))
    cuts = sorted(set(cuts), reverse=True)
    rows = []
    for entry, exit_ in zip(cuts[:-1], cuts[1:]):
        midpoint = 0.5 * (entry + exit_)
        radius = np.sqrt(
            max(
                detector_radius**2
                + midpoint**2
                + 2.0 * detector_radius * direction * midpoint,
                0.0,
            )
        )
        shell = min(int(np.searchsorted(radii, radius, side="right")), len(radii) - 1)
        points = np.linspace(entry, exit_, subdivisions + 1)
        for start, stop in zip(points[:-1], points[1:]):
            if start > stop:
                rows.append((start, stop, shell))
    return np.asarray(rows, dtype=float).reshape(-1, 3)

import numpy as np


def average_segment_densities(
    segments: "np.ndarray",
    radii_km: "np.ndarray",
    density_coefficients: "np.ndarray",
    electron_fractions: "np.ndarray",
    cos_zenith: float,
    detector_depth_km: float,
    quadrature_order: int = 16,
) -> "np.ndarray":
    raw = np.asarray(radii_km)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("radii must be a nonempty vector")
    radii = _real_array(radii_km, raw.shape, "radii_km")
    coefficients = _real_array(
        density_coefficients, (len(radii), 4), "density_coefficients"
    )
    fractions = _real_array(electron_fractions, radii.shape, "electron_fractions")
    raw_segments = np.asarray(segments)
    if raw_segments.ndim != 2 or raw_segments.shape[1] != 3:
        raise ValueError("segments must have shape (N, 3)")
    paths = _real_array(segments, raw_segments.shape, "segments")
    direction = _finite_scalar(cos_zenith, "cos_zenith")
    depth = _finite_scalar(detector_depth_km, "detector_depth_km")
    if (
        np.any(radii <= 0)
        or np.any(np.diff(radii) <= 0)
        or not -1 <= direction <= 1
        or not 0 <= depth < radii[-1]
    ):
        raise ValueError("invalid geometry")
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("invalid electron fraction")
    if (
        isinstance(quadrature_order, (bool, np.bool_))
        or not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 1
    ):
        raise ValueError("quadrature_order must be a positive integer")
    if np.any(paths[:, 0] <= paths[:, 1]) or np.any(paths[:, 1] < 0):
        raise ValueError("invalid segment endpoints")
    if (
        np.any(paths[:, 2] != np.floor(paths[:, 2]))
        or np.any(paths[:, 2] < 0)
        or np.any(paths[:, 2] >= len(radii))
    ):
        raise ValueError("invalid shell index")
    result = np.empty((len(paths), 2))
    detector_radius = radii[-1] - depth
    nodes = (np.arange(quadrature_order) + 0.5) / quadrature_order
    for index, (entry, exit_, shell_value) in enumerate(paths):
        shell = int(shell_value)
        length = entry - exit_
        distances = exit_ + nodes * length
        radial = (
            np.sqrt(
                np.maximum(
                    detector_radius**2
                    + distances**2
                    + 2.0 * detector_radius * direction * distances,
                    0.0,
                )
            )
            / radii[-1]
        )
        density = np.polynomial.polynomial.polyval(radial, coefficients[shell])
        if np.any(density < 0) or not np.all(np.isfinite(density)):
            raise ValueError("sampled density must be nonnegative and finite")
        result[index] = length, float(fractions[shell] * np.mean(density))
    return result

import numpy as np


def _validate_mixing(s12_sq, s13_sq, dm21_ev2, dm31_ev2):
    s12 = _finite_scalar(s12_sq, "s12_sq")
    s13 = _finite_scalar(s13_sq, "s13_sq")
    m21 = _finite_scalar(dm21_ev2, "dm21_ev2")
    m31 = _finite_scalar(dm31_ev2, "dm31_ev2")
    if not 0 < s12 < 1 or not 0 < s13 < 1 or not 0 < m21 < m31:
        raise ValueError(
            "mixing angles or normal-ordering masses are outside the domain"
        )
    return s12, s13, m21, m31


def _spectral_coefficients(
    s12_sq: float, s13_sq: float, dm21_ev2: float, dm31_ev2: float
) -> "np.ndarray":
    s12, s13, m21, m31 = _validate_mixing(s12_sq, s13_sq, dm21_ev2, dm31_ev2)
    c12 = 1.0 - s12
    c13 = 1.0 - s13
    dmee = m31 - s12 * m21
    cross13 = np.sqrt(s13 * c13)
    see = m21 + dmee * c13
    sem = -m21 * np.sqrt(c13 * s12 * c12)
    set_ = -dmee * cross13
    tee = m31 * m21 * c13 * c12
    tem = m31 * sem
    tet = -m31 * m21 * cross13 * c12
    return np.array([[see, sem, set_], [tee, tem, tet]])


def compute_matter_spectrum(
    matter_ev2: float,
    s12_sq: float,
    s13_sq: float,
    dm21_ev2: float,
    dm31_ev2: float,
    n_newton: int = 2,
) -> "np.ndarray":
    a = _finite_scalar(matter_ev2, "matter_ev2")
    s12, s13, m21, m31 = _validate_mixing(s12_sq, s13_sq, dm21_ev2, dm31_ev2)
    if (
        isinstance(n_newton, (bool, np.bool_))
        or not isinstance(n_newton, (int, np.integer))
        or not 0 <= n_newton <= 8
    ):
        raise ValueError("n_newton must be an integer between zero and eight")
    coefficients = _spectral_coefficients(s12, s13, m21, m31)
    if a == 0.0:
        return np.vstack((np.array([0.0, m21, m31]), coefficients))
    aa = m21 + m31 + a
    bb = m21 * m31 + a * coefficients[0, 0]
    cc = a * coefficients[1, 0]
    dmee = m31 - s12 * m21
    x = a / dmee
    l3 = m31 + 0.5 * dmee * (x - 1.0 + np.sqrt((1.0 - x) ** 2 + 4.0 * s13 * x))
    for _ in range(n_newton):
        derivative = l3 * (3.0 * l3 - 2.0 * aa) + bb
        if derivative == 0.0:
            raise ValueError("zero characteristic derivative")
        l3 = (l3 * l3 * (2.0 * l3 - aa) + cc) / derivative
    if l3 == 0.0 or not np.isfinite(l3):
        raise ValueError("atmospheric eigenvalue cannot be recovered")
    first = (aa - l3) ** 2
    second = 4.0 * cc / l3
    radicand = first - second
    scale = max(abs(first), abs(second), np.finfo(float).tiny)
    if radicand < -64.0 * np.finfo(float).eps * scale:
        raise ValueError("negative eigenvalue recovery radicand")
    gap = np.sqrt(max(radicand, 0.0))
    l2 = 0.5 * (aa - l3 + gap)
    roots = np.array([l2 - gap, l2, l3])
    if not np.all(np.isfinite(roots)) or np.any(np.diff(roots) <= 0.0):
        raise ValueError("eigenvalues must be finite and strictly increasing")
    return np.vstack((roots, coefficients))

import numpy as np


def compute_eigenprojectors(
    eigenvalues_ev2: "np.ndarray", coefficients: "np.ndarray"
) -> "np.ndarray":
    roots = _real_array(eigenvalues_ev2, (3,), "eigenvalues_ev2")
    coeff = _real_array(coefficients, (2, 3), "coefficients")
    if np.any(np.diff(roots) <= 0.0):
        raise ValueError("eigenvalues must be strictly increasing")
    projectors = np.empty((2, 3, 3))
    for out, index in enumerate((1, 2)):
        value = roots[index]
        denominator = np.prod(value - np.delete(roots, index))
        electron = (-value * coeff[0] + coeff[1]) / denominator
        electron[0] += value * value / denominator
        ee, em, et = electron
        if ee <= 0.0:
            raise ValueError("electron projection must be positive")
        mm = em * em / ee
        mt = em * et / ee
        tt = 1.0 - ee - mm
        projectors[out] = [[ee, em, et], [em, mm, mt], [et, mt, tt]]
    if not np.all(np.isfinite(projectors)):
        raise ValueError("projectors must be finite")
    return projectors

import numpy as np


def _layer_amplitude(
    eigenvalues_ev2: "np.ndarray",
    projectors: "np.ndarray",
    length_km: float,
    energy_gev: float,
) -> "np.ndarray":
    roots = _real_array(eigenvalues_ev2, (3,), "eigenvalues_ev2")
    proj = _real_array(projectors, (2, 3, 3), "projectors")
    length = _finite_scalar(length_km, "length_km")
    energy = _finite_scalar(energy_gev, "energy_gev")
    if length < 0.0 or energy <= 0.0 or np.any(np.diff(roots) <= 0.0):
        raise ValueError("invalid length, energy, or eigenvalue ordering")
    if not np.allclose(proj, proj.transpose(0, 2, 1), rtol=0.0, atol=1e-10):
        raise ValueError("projectors must be symmetric")
    conversion = 1e-9 * 1e3 / (2.0 * 1.97327e-7)
    phases = conversion * (roots[1:] - roots[0]) * length / energy
    return np.eye(3, dtype=complex) + np.einsum(
        "i,ijk->jk", np.expm1(-1j * phases), proj
    )


def _compose_amplitude(
    layer_amplitudes: "np.ndarray", symmetric: bool = False
) -> "np.ndarray":
    layers = np.asarray(layer_amplitudes, dtype=complex)
    if (
        layers.ndim != 3
        or layers.shape[1:] != (3, 3)
        or not np.all(np.isfinite(layers))
    ):
        raise ValueError("layers must have finite shape (N, 3, 3)")
    if not isinstance(symmetric, (bool, np.bool_)):
        raise ValueError("symmetric must be Boolean")
    if symmetric:
        if not np.allclose(layers, layers.transpose(0, 2, 1), rtol=0.0, atol=1e-10):
            raise ValueError("individual layers must be symmetric")
        if not np.allclose(layers, layers[::-1], rtol=0.0, atol=1e-10):
            raise ValueError("the layer sequence must be palindromic")
        half = len(layers) // 2
        inbound = np.eye(3, dtype=complex)
        for layer in layers[:half]:
            inbound = layer @ inbound
        center = layers[half] if len(layers) % 2 else np.eye(3, dtype=complex)
        return inbound.T @ center @ inbound
    amplitude = np.eye(3, dtype=complex)
    for layer in layers:
        amplitude = layer @ amplitude
    return amplitude


def _vacuum_matrix(parameters):
    s12, s13, m21, m31 = parameters
    x, y = np.sqrt([s12, s13])
    u, v = np.sqrt([1.0 - s12, 1.0 - s13])
    r12 = np.array([[u, x, 0.0], [-x, u, 0.0], [0.0, 0.0, 1.0]])
    r13 = np.array([[v, 0.0, y], [0.0, 1.0, 0.0], [-y, 0.0, v]])
    rotation = r13 @ r12
    return rotation @ np.diag([0.0, m21, m31]) @ rotation.T


def propagate_spectral_pair(
    layers: "np.ndarray", energy_gev: float, mixing: "np.ndarray", charge: int = 1
) -> "np.ndarray":
    raw = np.asarray(layers)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("layers must have shape (N, 2)")
    data = _real_array(layers, raw.shape, "layers")
    energy = _finite_scalar(energy_gev, "energy_gev")
    parameters = _real_array(mixing, (4,), "mixing")
    _validate_mixing(*parameters)
    if np.any(data < 0.0) or energy <= 0.0:
        raise ValueError("invalid layer values or energy")
    if (
        isinstance(charge, (bool, np.bool_))
        or not isinstance(charge, (int, np.integer))
        or charge not in (-1, 1)
    ):
        raise ValueError("charge must be +1 or -1")
    vacuum = _vacuum_matrix(parameters)
    amplitudes = np.empty((2, len(data), 3, 3), dtype=complex)
    for index, (length, density) in enumerate(data):
        if length == 0.0:
            amplitudes[:, index] = np.eye(3)
            continue
        potential = charge * 1.526493231029146e-4 * density * energy
        spectrum = compute_matter_spectrum(potential, *parameters, 0)
        projectors = compute_eigenprojectors(spectrum[0], spectrum[1:])
        amplitudes[0, index] = _layer_amplitude(spectrum[0], projectors, length, energy)
        matrix = vacuum + np.diag([potential, 0.0, 0.0])
        roots, vectors = np.linalg.eigh(matrix)
        exact_projectors = np.array(
            [np.outer(vectors[:, j], vectors[:, j]) for j in (1, 2)]
        )
        amplitudes[1, index] = _layer_amplitude(roots, exact_projectors, length, energy)
    symmetric = np.array_equal(data, data[::-1])
    return np.array([_compose_amplitude(branch, symmetric) for branch in amplitudes])

import numpy as np


def _channel_harmonics(amplitude, s23_sq, charge):
    sine, cosine = np.sqrt([s23_sq, 1.0 - s23_sq])
    channels = np.array(
        [
            [sine * amplitude[0, 2], cosine * amplitude[0, 1], 0.0],
            [
                sine * cosine * amplitude[1, 2],
                (1.0 - s23_sq) * amplitude[1, 1] + s23_sq * amplitude[2, 2],
                sine * cosine * amplitude[2, 1],
            ],
        ],
        dtype=complex,
    )
    negative, constant, positive = channels.T
    first = constant * negative.conj() + positive * constant.conj()
    second = positive * negative.conj()
    return np.column_stack(
        (
            np.sum(np.abs(channels) ** 2, axis=1),
            2.0 * first.real,
            -2.0 * charge * first.imag,
            2.0 * second.real,
            -2.0 * charge * second.imag,
        )
    )


def compute_error_harmonics(
    amplitudes: "np.ndarray", s23_sq: float, charge: int = 1
) -> "np.ndarray":
    matrices = np.asarray(amplitudes, dtype=complex)
    if matrices.shape != (2, 3, 3) or not np.all(np.isfinite(matrices)):
        raise ValueError("amplitudes must have finite shape (2, 3, 3)")
    angle = _finite_scalar(s23_sq, "s23_sq")
    if not 0.0 <= angle <= 1.0:
        raise ValueError("s23_sq must be in [0, 1]")
    if (
        isinstance(charge, (bool, np.bool_))
        or not isinstance(charge, (int, np.integer))
        or charge not in (-1, 1)
    ):
        raise ValueError("charge must be +1 or -1")
    result = _channel_harmonics(matrices[0], angle, charge) - _channel_harmonics(
        matrices[1], angle, charge
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("probability harmonics must be finite")
    return result

import numpy as np


def _phase_extrema(coefficients):
    h0, a, b, c, d = coefficients
    polynomial = np.array(
        [-b + 2 * d, -2 * a + 8 * c, -12 * d, -2 * a - 8 * c, b + 2 * d]
    )
    scale = np.max(np.abs(polynomial))
    phases = [0.0, np.pi]
    if scale > 0.0:
        roots = np.roots(np.trim_zeros(polynomial / scale, "f"))
        for root in roots:
            if abs(root.imag) <= 1e-7 * (1.0 + abs(root.real)):
                phases.append(float((2.0 * np.arctan(root.real)) % (2.0 * np.pi)))
    phases = np.array(phases)
    values = (
        h0
        + a * np.cos(phases)
        + b * np.sin(phases)
        + c * np.cos(2 * phases)
        + d * np.sin(2 * phases)
    )
    return phases, values


def maximize_phase_error(coefficients: "np.ndarray") -> float:
    data = _real_array(coefficients, (5,), "coefficients")
    scale = float(np.max(np.abs(data)))
    if scale == 0.0:
        return 0.0
    _, values = _phase_extrema(data / scale)
    result = float(np.max(np.abs(values)) * scale)
    if not np.isfinite(result):
        raise ValueError("maximum must be finite")
    return result

import numpy as np


def compute_worst_spectral_error(
    radii_km: "np.ndarray",
    density_coefficients: "np.ndarray",
    electron_fractions: "np.ndarray",
    energies_gev: "np.ndarray",
    cos_zeniths: "np.ndarray",
    mixing: "np.ndarray",
    detector_depth_km: float = 2.0,
    subdivisions: int = 3,
    quadrature_order: int = 16,
) -> float:
    parameters = _real_array(mixing, (5,), "mixing")
    reduced = parameters[[0, 1, 3, 4]]
    _validate_mixing(*reduced)
    if not 0 <= parameters[2] <= 1:
        raise ValueError("s23_sq must be in [0, 1]")
    raw_energies, raw_cosines = np.asarray(energies_gev), np.asarray(cos_zeniths)
    if (
        raw_energies.ndim != 1
        or raw_energies.size == 0
        or raw_cosines.ndim != 1
        or raw_cosines.size == 0
    ):
        raise ValueError("energy and direction vectors must be nonempty")
    energies = _real_array(energies_gev, raw_energies.shape, "energies_gev")
    cosines = _real_array(cos_zeniths, raw_cosines.shape, "cos_zeniths")
    if np.any(energies <= 0) or np.any(np.abs(cosines) > 1):
        raise ValueError("invalid energy or direction")
    maximum = 0.0
    for direction in cosines:
        segments = trace_shell_segments(
            radii_km, direction, detector_depth_km, subdivisions
        )
        layers = average_segment_densities(
            segments,
            radii_km,
            density_coefficients,
            electron_fractions,
            direction,
            detector_depth_km,
            quadrature_order,
        )
        for energy in energies:
            for charge in (1, -1):
                amplitudes = propagate_spectral_pair(
                    layers, energy, reduced, charge
                )
                harmonics = compute_error_harmonics(
                    amplitudes, parameters[2], charge
                )
                for coefficients in harmonics:
                    maximum = max(maximum, maximize_phase_error(coefficients))
    result = float(1e6 * maximum)
    if not np.isfinite(result):
        raise ValueError("scaled maximum must be finite")
    return result
SCICODE_GOLD_EOF
