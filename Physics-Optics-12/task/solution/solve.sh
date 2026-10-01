#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math
import numpy as np

def relativistic_electron_constants(voltage_kv: float) -> 'np.ndarray':
    """Return [wavelength_A, interaction_constant_rad_per_VA]."""
    voltage_kv = float(voltage_kv)
    if not math.isfinite(voltage_kv) or voltage_kv <= 0.0:
        raise ValueError('voltage_kv must be finite and positive')
    h = 6.62607015e-34
    m0 = 9.1093837139e-31
    e = 1.602176634e-19
    c = 299792458.0
    voltage_v = 1000.0 * voltage_kv
    wavelength_a = h / math.sqrt(2.0 * m0 * e * voltage_v * (1.0 + e * voltage_v / (2.0 * m0 * c * c))) * 10000000000.0
    sigma = 2.0 * math.pi / (wavelength_a * voltage_v) * (m0 * c * c + e * voltage_v) / (2.0 * m0 * c * c + e * voltage_v)
    return np.asarray([wavelength_a, sigma], dtype=float)

import math
import numpy as np

def build_grouped_strong_scattering_stack(n: int, pixel_size_a: float, hypothesis: int, group_factor: int=1) -> 'np.ndarray':
    """Return grouped projected potentials for the fixed six-slice specimen."""
    n = int(n)
    pixel_size_a = float(pixel_size_a)
    hypothesis = int(hypothesis)
    group_factor = int(group_factor)
    if n < 2 or n & n - 1:
        raise ValueError('n must be a power of two at least 2')
    if not math.isfinite(pixel_size_a) or pixel_size_a <= 0.0:
        raise ValueError('pixel_size_a must be finite and positive')
    if hypothesis not in (0, 1):
        raise ValueError('hypothesis must be 0 or 1')
    if group_factor not in (1, 2, 3, 6):
        raise ValueError('group_factor must divide the six base slices')
    length = n * pixel_size_a
    axis = np.arange(n, dtype=float) * pixel_size_a
    (xx, yy) = np.meshgrid(axis, axis, indexing='ij')

    def _periodic_r2(cx, cy):
        dx = (xx - cx + 0.5 * length) % length - 0.5 * length
        dy = (yy - cy + 0.5 * length) % length - 0.5 * length
        return dx * dx + dy * dy
    width = 0.42
    d0 = _periodic_r2(0.0, 0.0)
    d1 = _periodic_r2(0.5 * length, 0.5 * length)
    d2 = _periodic_r2(0.25 * length, 0.75 * length)
    base = []
    for j in range(6):
        central = (1500.0 if hypothesis == 1 else 1050.0) * (1.0 + 0.08 * (j % 3 - 1))
        corner = 650.0 * (1.0 + 0.05 * (-1) ** j)
        offset = 400.0 * (1.0 - 0.04 * (-1) ** j)
        base.append(central * np.exp(-d0 / (2.0 * width ** 2)) + corner * np.exp(-d1 / (2.0 * (1.25 * width) ** 2)) + offset * np.exp(-d2 / (2.0 * (0.9 * width) ** 2)))
    base = np.asarray(base, dtype=float)
    return np.asarray([np.sum(base[j:j + group_factor], axis=0) for j in range(0, 6, group_factor)], dtype=float)

import math
import numpy as np

def amplitude_encoded_stem_probe(n: int, pixel_size_a: float, wavelength_a: float, convergence_mrad: float, defocus_a: float, cs_mm: float=0.05, scan_position_a: tuple=(0.0, 0.0)) -> 'np.ndarray':
    """Return the normalized N-by-N amplitudes stored on 2 log2(N) qubits."""
    n = int(n)
    values = [float(pixel_size_a), float(wavelength_a), float(convergence_mrad), float(defocus_a), float(cs_mm)]
    position = np.asarray(scan_position_a, dtype=float)
    if n < 2 or n & n - 1:
        raise ValueError('n must be a power of two at least 2')
    if any((not math.isfinite(x) for x in values)):
        raise ValueError('probe parameters must be finite')
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise ValueError('scan_position_a must contain two finite coordinates')
    if values[0] <= 0.0 or values[1] <= 0.0 or values[2] <= 0.0 or (values[4] < 0.0):
        raise ValueError('invalid probe scale')
    frequency = np.fft.fftfreq(n, d=values[0])
    (kx, ky) = np.meshgrid(frequency, frequency, indexing='ij')
    k2 = kx * kx + ky * ky
    aperture = np.sqrt(k2) <= values[2] * 0.001 / values[1] + 1e-15
    cs_a = values[4] * 10000000.0
    chi = np.pi * values[1] * values[3] * k2 + 0.5 * np.pi * cs_a * values[1] ** 3 * k2 ** 2
    shift = np.exp(-2j * np.pi * (kx * position[0] + ky * position[1]))
    reciprocal = aperture.astype(float) * np.exp(-1j * chi) * shift
    probe = np.fft.ifft2(reciprocal, norm='ortho')
    norm = float(np.linalg.norm(probe))
    if norm == 0.0:
        raise ValueError('empty aperture')
    return np.asarray(probe / norm, dtype=complex)

import math
import numpy as np

def quantum_slice_block(state: 'np.ndarray', projected_potential: 'np.ndarray', wavelength_a: float, interaction_constant: float, propagation_distance_a: float, pixel_size_a: float, propagate_after: bool=True) -> 'np.ndarray':
    """Apply a phase grating and, when requested, an inter-grating Fresnel segment."""
    wave = np.asarray(state, dtype=complex).copy()
    potential = np.asarray(projected_potential, dtype=float)
    scalars = [float(wavelength_a), float(interaction_constant), float(propagation_distance_a), float(pixel_size_a)]
    if wave.ndim != 2 or wave.shape[0] != wave.shape[1] or wave.shape != potential.shape:
        raise ValueError('state and potential must be equal square arrays')
    if wave.shape[0] < 2 or wave.shape[0] & wave.shape[0] - 1:
        raise ValueError('linear grid size must be a power of two')
    if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(potential)):
        raise ValueError('state and potential must be finite')
    if any((not math.isfinite(x) or x <= 0.0 for x in scalars)):
        raise ValueError('physical scalars must be finite and positive')
    if not isinstance(propagate_after, (bool, np.bool_)):
        raise ValueError('propagate_after must be boolean')
    wave *= np.exp(1j * scalars[1] * potential)
    if not propagate_after:
        return wave
    n = wave.shape[0]
    indices = np.arange(n, dtype=float)
    qft = np.exp(-2j * np.pi * np.outer(indices, indices) / n) / np.sqrt(n)
    frequency = np.fft.fftfreq(n, d=scalars[3])
    (kx, ky) = np.meshgrid(frequency, frequency, indexing='ij')
    propagator = np.exp(-1j * np.pi * scalars[0] * scalars[2] * (kx * kx + ky * ky))
    reciprocal = qft @ wave @ qft.T
    return np.asarray(qft.conj().T @ (reciprocal * propagator) @ qft.conj(), dtype=complex)

import numpy as np

def quantum_multislice_exit_state(grouped_stack: 'np.ndarray', incident_state: 'np.ndarray', wavelength_a: float, interaction_constant: float, base_slice_thickness_a: float, group_factor: int, pixel_size_a: float) -> 'np.ndarray':
    """Execute one QuScope quantum slice block for every grouped slice."""
    stack = np.asarray(grouped_stack, dtype=float)
    state = np.asarray(incident_state, dtype=complex).copy()
    group_factor = int(group_factor)
    if stack.ndim != 3 or stack.shape[0] < 1 or stack.shape[1] != stack.shape[2]:
        raise ValueError('grouped_stack must have shape (s,n,n)')
    if state.shape != stack.shape[1:]:
        raise ValueError('incident state shape mismatch')
    if group_factor not in (1, 2, 3, 6) or stack.shape[0] * group_factor != 6:
        raise ValueError('grouping must represent exactly six base slices')
    distance = float(base_slice_thickness_a) * group_factor
    for (j, potential) in enumerate(stack):
        state = quantum_slice_block(state, potential, wavelength_a, interaction_constant, distance, pixel_size_a, propagate_after=j < stack.shape[0] - 1)
    return np.asarray(state, dtype=complex)

import math
import numpy as np

def annular_detector_vector(exit_state: 'np.ndarray', wavelength_a: float, pixel_size_a: float, detector_edges_mrad: 'np.ndarray') -> 'np.ndarray':
    """Return the three probabilities for consecutive half-open annuli."""
    state = np.asarray(exit_state, dtype=complex)
    edges = np.asarray(detector_edges_mrad, dtype=float)
    wavelength_a = float(wavelength_a)
    pixel_size_a = float(pixel_size_a)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError('exit_state must be square')
    if edges.shape != (4,) or not np.all(np.isfinite(edges)):
        raise ValueError('detector_edges_mrad must have shape (4,)')
    if edges[0] < 0.0 or np.any(np.diff(edges) <= 0.0):
        raise ValueError('detector edges must be strictly increasing and nonnegative')
    if not math.isfinite(wavelength_a) or not math.isfinite(pixel_size_a) or wavelength_a <= 0.0 or (pixel_size_a <= 0.0):
        raise ValueError('invalid sampling')
    frequency = np.fft.fftfreq(state.shape[0], d=pixel_size_a)
    (kx, ky) = np.meshgrid(frequency, frequency, indexing='ij')
    theta = 1000.0 * wavelength_a * np.sqrt(kx * kx + ky * ky)
    intensity = np.abs(np.fft.fft2(state, norm='ortho')) ** 2 / float(np.vdot(state, state).real)
    values = []
    for (lo, hi) in zip(edges[:-1], edges[1:]):
        mask = (theta >= lo) & (theta < hi)
        if not np.any(mask):
            raise ValueError('every detector annulus must contain a grid point')
        values.append(float(np.sum(intensity[mask])))
    return np.asarray(values, dtype=float)

import numpy as np

def bivariate_multislice_orders(background_stack: 'np.ndarray', contrast_stack: 'np.ndarray', incident_state: 'np.ndarray', wavelength_a: float, interaction_constant: float, propagation_distance_a: float, pixel_size_a: float) -> 'np.ndarray':
    raw_B = np.asarray(background_stack)
    raw_D = np.asarray(contrast_stack)
    if np.iscomplexobj(raw_B) or np.iscomplexobj(raw_D):
        raise ValueError('projected potentials must be real')
    B = np.asarray(raw_B, dtype=float)
    D = np.asarray(raw_D, dtype=float)
    psi = np.asarray(incident_state, dtype=complex)
    if B.ndim != 3 or B.shape != D.shape or (not 1 <= B.shape[0] <= 6):
        raise ValueError('equal (s,N,N) stacks with 1<=s<=6 are required')
    (s, n, m) = B.shape
    if n != m or n < 2 or n & n - 1 or (psi.shape != (n, n)):
        raise ValueError('square power-of-two spatial arrays required')
    if not all((np.all(np.isfinite(v)) for v in (B, D, psi))) or np.linalg.norm(psi) == 0:
        raise ValueError('finite data and a nonzero incident wave are required')
    (lam, sigma, dz, dx) = map(float, (wavelength_a, interaction_constant, propagation_distance_a, pixel_size_a))
    if not np.all(np.isfinite([lam, sigma, dz, dx])) or min(lam, dz, dx) <= 0 or sigma < 0:
        raise ValueError('invalid phase or propagation scale')
    f = np.fft.fftfreq(n, d=dx)
    phase = np.exp(-1j * np.pi * lam * dz * (f[:, None] ** 2 + f[None, :] ** 2))
    coefficients = np.zeros((s + 1, s + 1, n, n), dtype=complex)
    coefficients[0, 0] = psi
    for j in range(s):
        previous = coefficients.copy()
        coefficients[1:] += 1j * sigma * B[j] * previous[:-1]
        coefficients[:, 1:] += 1j * sigma * D[j] * previous[:, :-1]
        if j < s - 1:
            coefficients = np.fft.ifft2(np.fft.fft2(coefficients, axes=(-2, -1), norm='ortho') * phase, axes=(-2, -1), norm='ortho')
    return np.asarray([coefficients[d - q, q] for d in range(s + 1) for q in range(d + 1)])

import numpy as np

def annular_order_coherence(order_amplitudes: 'np.ndarray', wavelength_a: float, pixel_size_a: float, detector_edges_mrad: 'np.ndarray') -> 'np.ndarray':
    modes = np.asarray(order_amplitudes, dtype=complex)
    edges = np.asarray(detector_edges_mrad, dtype=float)
    if modes.ndim != 3 or modes.shape[0] < 1 or (not np.all(np.isfinite(modes))):
        raise ValueError('finite (K,N,N) amplitude array required')
    (count, n, m) = modes.shape
    if n != m or n < 2 or n & n - 1:
        raise ValueError('square power-of-two spatial axes required')
    (lam, dx) = (float(wavelength_a), float(pixel_size_a))
    if not np.all(np.isfinite([lam, dx])) or min(lam, dx) <= 0:
        raise ValueError('invalid sampling')
    if edges.shape != (4,) or not np.all(np.isfinite(edges)) or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('four ordered nonnegative detector edges required')
    f = np.fft.fftfreq(n, d=dx)
    theta = 1000 * lam * np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    masks = [(theta >= lo) & (theta < hi) for (lo, hi) in zip(edges[:-1], edges[1:])]
    if any((not np.any(mask) for mask in masks)):
        raise ValueError('every annulus must contain a grid point')
    masks.append(np.ones((n, n), dtype=bool))
    spectral = np.fft.fft2(modes, axes=(-2, -1), norm='ortho')
    return np.asarray([spectral[:, mask].conj() @ spectral[:, mask].T for mask in masks])

import math
import numpy as np

def transpiled_multislice_resources(n: int, slice_count: int) -> 'np.ndarray':
    """Return [logical_qubits, diagonal_gates, 2-D QFTs, total_CX]."""
    n = int(n)
    slice_count = int(slice_count)
    if n < 2 or n & n - 1:
        raise ValueError('n must be a power of two at least 2')
    if slice_count < 1:
        raise ValueError('slice_count must be positive')
    register_qubits = int(round(math.log2(n)))
    logical_qubits = 2 * register_qubits
    diagonal_gates = 2 * slice_count - 1
    qft2_transforms = 2 * (slice_count - 1)
    arbitrary_diagonal_cx = 2 ** logical_qubits - 2
    one_register_qft_cx = register_qubits * (register_qubits - 1) + 3 * (register_qubits // 2)
    total_cx = diagonal_gates * arbitrary_diagonal_cx + qft2_transforms * 2 * one_register_qft_cx
    return np.asarray([logical_qubits, diagonal_gates, qft2_transforms, total_cx], dtype=float)

import math
import numpy as np

def quantum_resource_design_table(candidates: 'np.ndarray', voltages_kv: 'np.ndarray', defocuses_a: 'np.ndarray', n_values: 'np.ndarray'=(16, 32), group_factors: 'np.ndarray'=(1, 2, 3, 6), field_of_view_a: float=6.4, base_slice_thickness_a: float=1.8, cs_mm: float=0.05, reference_n: int=64, detector_error_tolerance: float=0.02, contrast_floor: float=0.121, fidelity_floor: float=0.999999999999, scan_positions_a: 'np.ndarray'=((0.0, 0.0), (0.18, 0.0), (0.0, 0.18), (0.18, 0.18), (0.31, -0.13))) -> 'np.ndarray':
    """Build the candidate-major QuScope accuracy/resource table."""
    candidates = np.asarray(candidates, dtype=float)
    voltages = np.asarray(voltages_kv, dtype=float)
    defocuses = np.asarray(defocuses_a, dtype=float)
    n_values = np.asarray(n_values, dtype=int)
    group_factors = np.asarray(group_factors, dtype=int)
    scan_positions = np.asarray(scan_positions_a, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 5 or candidates.shape[0] == 0:
        raise ValueError('candidates must have shape (C,5)')
    if voltages.ndim != 1 or voltages.size == 0 or defocuses.ndim != 1 or (defocuses.size == 0):
        raise ValueError('scenario vectors must be nonempty')
    if n_values.ndim != 1 or n_values.size == 0 or group_factors.ndim != 1 or (group_factors.size == 0):
        raise ValueError('design vectors must be nonempty')
    if scan_positions.ndim != 2 or scan_positions.shape[0] < 2 or scan_positions.shape[1] != 2 or (not np.all(np.isfinite(scan_positions))):
        raise ValueError('scan_positions_a must be a finite (S,2) array with S>=2')
    if np.any(np.diff(candidates[:, 1:], axis=1) <= 0.0) or np.any(candidates[:, 0] <= 0.0):
        raise ValueError('candidate angles must be ordered and positive')
    if any((int(n) < 2 or int(n) & int(n) - 1 for n in n_values)):
        raise ValueError('n values must be powers of two')
    if any((int(g) not in (1, 2, 3, 6) for g in group_factors)):
        raise ValueError('group factors must divide six')
    if int(reference_n) < max(n_values) or int(reference_n) & int(reference_n) - 1:
        raise ValueError('reference_n must be a power of two no smaller than all candidates')
    scalars = [field_of_view_a, base_slice_thickness_a, detector_error_tolerance, contrast_floor, fidelity_floor]
    if any((not math.isfinite(float(x)) for x in scalars)):
        raise ValueError('scalar controls must be finite')
    if field_of_view_a <= 0.0 or base_slice_thickness_a <= 0.0 or detector_error_tolerance <= 0.0 or (contrast_floor < 0.0) or (not 0.0 < fidelity_floor <= 1.0):
        raise ValueError('invalid scalar controls')

    def _constants(voltage_kv):
        value = relativistic_electron_constants(voltage_kv)
        return (float(value[0]), float(value[1]))

    def _grouped_stack(n, hypothesis, group):
        return build_grouped_strong_scattering_stack(n, field_of_view_a / n, hypothesis, group)

    def _probe(n, dx, wavelength, alpha, defocus, position):
        return amplitude_encoded_stem_probe(n, dx, wavelength, alpha, defocus, cs_mm, position)

    def _propagator_grid(n, dx, wavelength, distance):
        frequency = np.fft.fftfreq(n, d=dx)
        (kx, ky) = np.meshgrid(frequency, frequency, indexing='ij')
        return np.exp(-1j * np.pi * wavelength * distance * (kx * kx + ky * ky))

    def _quantum_exit(stack, incident, wavelength, sigma, distance, dx):
        group = int(round(float(distance) / float(base_slice_thickness_a)))
        return quantum_multislice_exit_state(stack, incident, wavelength, sigma, base_slice_thickness_a, group, dx)

    def _classical_exit(stack, incident, wavelength, sigma, distance, dx):
        propagation = _propagator_grid(incident.shape[0], dx, wavelength, distance)
        state = incident.copy()
        for (j, potential) in enumerate(stack):
            state *= np.exp(1j * sigma * potential)
            if j < len(stack) - 1:
                state = np.fft.ifft2(np.fft.fft2(state, norm='ortho') * propagation, norm='ortho')
        return state

    def _detector(state, wavelength, dx, edges):
        try:
            return annular_detector_vector(state, wavelength, dx, edges)
        except ValueError:
            return None

    def _resources(n, slices):
        value = transpiled_multislice_resources(n, slices)
        return tuple((float(x) for x in value))
    reference_dx = field_of_view_a / int(reference_n)
    references = {}
    for (candidate_index, candidate) in enumerate(candidates):
        alpha = float(candidate[0])
        edges = candidate[1:]
        for (voltage_index, voltage) in enumerate(voltages):
            (wavelength, sigma) = _constants(voltage)
            for (defocus_index, defocus) in enumerate(defocuses):
                for (scan_index, position) in enumerate(scan_positions):
                    incident = _probe(int(reference_n), reference_dx, wavelength, alpha, float(defocus), position)
                    for hypothesis in (0, 1):
                        stack = _grouped_stack(int(reference_n), hypothesis, 1)
                        state = _classical_exit(stack, incident, wavelength, sigma, base_slice_thickness_a, reference_dx)
                        references[candidate_index, voltage_index, defocus_index, hypothesis, scan_index] = _detector(state, wavelength, reference_dx, edges)
    rows = []
    for (candidate_index, candidate) in enumerate(candidates):
        alpha = float(candidate[0])
        edges = candidate[1:]
        for (n_position, n) in enumerate(n_values):
            n = int(n)
            dx = field_of_view_a / n
            for (group_position, group) in enumerate(group_factors):
                group = int(group)
                slice_count = 6 // group
                (nq, diagonals, qft2, per_scan_cx) = _resources(n, slice_count)
                cx = per_scan_cx * scan_positions.shape[0]
                nyquist_feasible = True
                minimum_fidelity = math.inf
                maximum_error = -math.inf
                minimum_contrast = math.inf
                worst_index = -1
                for (voltage_index, voltage) in enumerate(voltages):
                    (wavelength, sigma) = _constants(voltage)
                    nyquist = 1000.0 * wavelength / (2.0 * dx)
                    if alpha > nyquist + 1e-12 or edges[-1] > nyquist + 1e-12:
                        nyquist_feasible = False
                    for (defocus_index, defocus) in enumerate(defocuses):
                        for (scan_index, position) in enumerate(scan_positions):
                            incident = _probe(n, dx, wavelength, alpha, float(defocus), position)
                            detector_vectors = []
                            for hypothesis in (0, 1):
                                stack = _grouped_stack(n, hypothesis, group)
                                quantum = _quantum_exit(stack, incident, wavelength, sigma, base_slice_thickness_a * group, dx)
                                classical = _classical_exit(stack, incident, wavelength, sigma, base_slice_thickness_a * group, dx)
                                denominator = float(np.vdot(quantum, quantum).real) * float(np.vdot(classical, classical).real)
                                fidelity = float(abs(np.vdot(quantum, classical)) ** 2 / denominator)
                                minimum_fidelity = min(minimum_fidelity, fidelity)
                                value = _detector(quantum, wavelength, dx, edges)
                                if value is None:
                                    nyquist_feasible = False
                                    value = np.zeros(3, dtype=float)
                                detector_vectors.append(value)
                                reference = references[candidate_index, voltage_index, defocus_index, hypothesis, scan_index]
                                error = float(np.linalg.norm(value - reference) / np.linalg.norm(reference))
                                error_index = scan_positions.shape[0] * (2 * (voltage_index * defocuses.size + defocus_index) + hypothesis) + scan_index
                                if error > maximum_error:
                                    maximum_error = error
                                    worst_index = error_index
                            contrast = float(np.linalg.norm(detector_vectors[1] - detector_vectors[0]))
                            minimum_contrast = min(minimum_contrast, contrast)
                feasible = nyquist_feasible and minimum_fidelity >= fidelity_floor and (maximum_error <= detector_error_tolerance) and (minimum_contrast >= contrast_floor)
                rows.append([float(candidate_index), float(n), float(group), float(slice_count), float(nq), float(nyquist_feasible), float(minimum_fidelity), float(maximum_error), float(minimum_contrast), float(cx), float(worst_index), float(feasible), float(n_position), float(group_position)])
    return np.asarray(rows, dtype=float)

import numpy as np

def select_quantum_resource_design(design_table: 'np.ndarray') -> 'np.ndarray':
    """Return the selected row's decisive numeric certificate."""
    table = np.asarray(design_table, dtype=float)
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] != 14:
        raise ValueError('design_table must have shape (R,14)')
    if not np.all(np.isfinite(table)):
        raise ValueError('design_table must be finite')
    feasible = np.where(table[:, 11] == 1.0)[0]
    if feasible.size == 0:
        raise ValueError('no quantum design satisfies every constraint')
    selected = min(feasible.tolist(), key=lambda i: (table[i, 9], table[i, 7], -table[i, 8], table[i, 0], table[i, 12], table[i, 13], i))
    row = table[selected]
    return np.asarray([float(selected), row[0], row[1], row[2], row[3], row[4], row[10], row[6], row[7], row[8], row[9]], dtype=float)

import numpy as np

def quscope_quantum_resource_design(candidates: 'np.ndarray', voltages_kv: 'np.ndarray', defocuses_a: 'np.ndarray', n_values: 'np.ndarray'=(16, 32), group_factors: 'np.ndarray'=(1, 2, 3, 6), field_of_view_a: float=6.4, base_slice_thickness_a: float=1.8, cs_mm: float=0.05, reference_n: int=64, detector_error_tolerance: float=0.02, contrast_floor: float=0.121, fidelity_floor: float=0.999999999999, scan_positions_a: 'np.ndarray'=((0.0, 0.0), (0.18, 0.0), (0.0, 0.18), (0.18, 0.18), (0.31, -0.13))) -> float:
    """Compose the robust QuScope design and strong-phase resource score."""
    constants = relativistic_electron_constants(float(voltages_kv[0]))
    n0 = int(n_values[0])
    dx0 = float(field_of_view_a) / n0
    g0 = int(group_factors[0])
    stack = build_grouped_strong_scattering_stack(n0, dx0, 0, g0)
    probe = amplitude_encoded_stem_probe(n0, dx0, constants[0], float(candidates[0][0]), float(defocuses_a[0]), cs_mm, scan_positions_a[0])
    first_block = quantum_slice_block(probe, stack[0], constants[0], constants[1], base_slice_thickness_a * g0, dx0, propagate_after=len(stack) > 1)
    exit_state = quantum_multislice_exit_state(stack, probe, constants[0], constants[1], base_slice_thickness_a, g0, dx0)
    resources = transpiled_multislice_resources(n0, 6 // g0)
    if first_block.shape != probe.shape or resources.shape != (4,):
        raise ValueError('invalid composed public-step output')
    table = quantum_resource_design_table(candidates, voltages_kv, defocuses_a, n_values, group_factors, field_of_view_a, base_slice_thickness_a, cs_mm, reference_n, detector_error_tolerance, contrast_floor, fidelity_floor, scan_positions_a)
    selected = select_quantum_resource_design(table)
    candidate_index = int(selected[1])
    n = int(selected[2])
    group = int(selected[3])
    slice_count = int(selected[4])
    logical_qubits = int(selected[5])
    dx = float(field_of_view_a) / n
    maximum_linearized_detector_error = -np.inf
    detector_edges = np.asarray(candidates, dtype=float)[candidate_index, 1:]
    positions = np.asarray(scan_positions_a, dtype=float)
    baseline_stack = build_grouped_strong_scattering_stack(n, dx, 0, group)
    contrast_stack = build_grouped_strong_scattering_stack(n, dx, 1, group) - baseline_stack
    order_pairs = [(degree - q, q) for degree in range(slice_count + 1) for q in range(degree + 1)]
    for voltage in np.asarray(voltages_kv, dtype=float):
        (wavelength, sigma) = relativistic_electron_constants(float(voltage))
        for defocus in np.asarray(defocuses_a, dtype=float):
            for position in positions:
                incident = amplitude_encoded_stem_probe(n, dx, wavelength, float(np.asarray(candidates)[candidate_index, 0]), float(defocus), cs_mm, position)
                orders = bivariate_multislice_orders(baseline_stack, contrast_stack, incident, wavelength, sigma, base_slice_thickness_a * group, dx)
                coherence = annular_order_coherence(orders, wavelength, dx, detector_edges)
                for hypothesis in (0, 1):
                    grouped = baseline_stack + hypothesis * contrast_stack
                    exact = quantum_multislice_exit_state(grouped, incident, wavelength, sigma, base_slice_thickness_a, group, dx)
                    weights = np.asarray([float(hypothesis ** q) for (p, q) in order_pairs])
                    coherent_signal = np.einsum('a,cab,b->c', weights, coherence, weights).real
                    if coherent_signal[3] <= 0.0 or not np.all(np.isfinite(coherent_signal)):
                        raise ValueError('invalid reconstructed linearized wave norm')
                    linearized_detector = coherent_signal[:3] / coherent_signal[3]
                    exact_detector = annular_detector_vector(exact, wavelength, dx, detector_edges)
                    mismatch = float(np.linalg.norm(linearized_detector - exact_detector) / np.linalg.norm(exact_detector))
                    maximum_linearized_detector_error = max(maximum_linearized_detector_error, mismatch)
    detector_outer = max(float(candidates[candidate_index][0]), float(candidates[candidate_index][-1]))
    nyquist_margin = min((1000.0 * relativistic_electron_constants(float(voltage))[0] / (2.0 * dx) - detector_outer for voltage in np.asarray(voltages_kv, dtype=float)))
    scan_count = np.asarray(scan_positions_a, dtype=float).shape[0]
    diagonal_scan_cx = scan_count * (2 * slice_count - 1) * (2 ** logical_qubits - 2)
    diagonal_cx_fraction = diagonal_scan_cx / selected[10]
    angular_scale_mrad = 1.0
    dimensionless_margin = nyquist_margin / angular_scale_mrad
    score = selected[10] * selected[8] * maximum_linearized_detector_error / (selected[9] * dimensionless_margin * diagonal_cx_fraction)
    return float(round(score, 6))
SCICODE_GOLD_EOF
