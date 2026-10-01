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


def _integer(value, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError("invalid integer parameter")
    return int(value)


def _array(value, shape=None, real=False):
    result = np.asarray(value, dtype=complex)
    if (shape is not None and result.shape != shape) or not np.all(np.isfinite(result)):
        raise ValueError("invalid array shape or nonfinite entries")
    if real:
        if np.any(result.imag != 0):
            raise ValueError("real data required")
        return result.real
    return result


def _scalar(value):
    return float(_array(value, (), real=True))


def _spin_matrices():
    return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2), np.diag([1, 0, -1])


def _joint_vector(matrix, levels):
    return matrix.reshape(3, levels, 3, levels).transpose(0, 2, 1, 3).reshape(-1)


def _joint_matrix(vector, levels):
    return (
        vector.reshape(3, 3, levels, levels)
        .transpose(0, 2, 1, 3)
        .reshape(3 * levels, 3 * levels)
    )


def _trace_row(levels):
    return np.kron(np.eye(3).reshape(-1), np.eye(levels).reshape(-1))


def _reduce_qubits(vector, levels):
    joint = _joint_matrix(vector, levels).reshape(3, levels, 3, levels)
    triplet = np.trace(joint, axis1=1, axis2=3)
    embedding = np.array(
        [[1, 0, 0], [0, 1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2), 0], [0, 0, 1]]
    )
    return embedding @ triplet @ embedding.T


def build_collective_generators(
    levels: int, omega: float, bath_frequency: float, coupling: float, decay: float
) -> "np.ndarray":
    """Retain the oscillator while restricting only the qubit symmetry sector."""
    levels = _integer(levels, 2)
    omega, bath_frequency, coupling, decay = map(
        _scalar, (omega, bath_frequency, coupling, decay)
    )
    if min(omega, bath_frequency, decay) <= 0 or coupling < 0:
        raise ValueError(
            "positive frequencies and decay, nonnegative coupling required"
        )
    jx, jz = _spin_matrices()
    annihilation = np.diag(np.sqrt(np.arange(1, levels)), 1)
    dimension = 3 * levels
    identity = np.eye(dimension)
    bath_h = bath_frequency * np.kron(np.eye(3), annihilation.T @ annihilation)
    bath_h += coupling * np.kron(jz, annihilation + annihilation.T)
    jump = np.kron(np.eye(3), annihilation)
    number = jump.T @ jump
    bath_l = -1j * (np.kron(bath_h, identity) - np.kron(identity, bath_h.T))
    bath_l += decay * (
        np.kron(jump, jump)
        - 0.5 * (np.kron(number, identity) + np.kron(identity, number.T))
    )
    system_h = omega * np.kron(jx, np.eye(levels))
    system_l = -1j * (np.kron(system_h, identity) - np.kron(identity, system_h.T))
    permutation = (
        np.arange(dimension**2)
        .reshape(3, levels, 3, levels)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    grouped = np.ix_(permutation, permutation)
    return np.array([bath_l[grouped], (bath_l + system_l)[grouped]])

import numpy as np
from scipy.linalg import eig


def _physical_density(matrix, tolerance=1e-8):
    if (
        np.linalg.norm(matrix - matrix.conj().T) > tolerance
        or abs(np.trace(matrix) - 1) > tolerance
    ):
        raise ValueError("a Hermitian trace-one density matrix is required")
    values = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2)
    if values.min() < -tolerance:
        raise ValueError("density matrix is not positive")


def resolve_quench_modes(
    generator: "np.ndarray", initial_state: "np.ndarray", levels: int
) -> "np.ndarray":
    """Weight every mode by its dual spectral projection of the quench."""
    levels = _integer(levels, 2)
    dimension = 9 * levels**2
    generator = _array(generator, (dimension, dimension))
    initial_state = _array(initial_state, (dimension,))
    _physical_density(_joint_matrix(initial_state, levels))
    rates, right = eig(generator)
    stationary = np.flatnonzero(np.abs(rates) < 1e-9)
    separation = np.abs(rates[:, None] - rates[None, :]) + np.eye(dimension)
    if len(stationary) != 1 or separation.min() < 1e-8 or np.linalg.cond(right) > 1e10:
        raise ValueError("a simple, well-resolved stationary spectrum is required")
    stationary = int(stationary[0])
    transient = [n for n in range(dimension) if n != stationary]
    if np.max(rates[transient].real) >= -1e-10:
        raise ValueError("all transient rates must decay")
    coefficients = np.linalg.solve(right, initial_state)
    weighted = (right * coefficients[None, :]).T
    transient.sort(
        key=lambda n: (round(float(rates[n].imag), 10), float(rates[n].real))
    )
    order = [stationary] + transient
    result = np.column_stack((rates[order], weighted[order]))
    result[0, 0] = 0
    return result

import numpy as np


def _concurrence(density):
    _physical_density(density, 1e-8)
    density = (density + density.conj().T) / 2
    values, vectors = np.linalg.eigh(density)
    root = (vectors * np.sqrt(np.maximum(values, 0))[None, :]) @ vectors.conj().T
    y = np.array([[0, -1j], [1j, 0]])
    singular = np.linalg.svd(root @ np.kron(y, y) @ root.conj(), compute_uv=False)
    return float(max(0.0, singular[0] - np.sum(singular[1:])))


def select_entangling_mode(
    modes: "np.ndarray",
    levels: int,
    phase_count: int,
    decay_limit: float,
    frequency_limit: float,
) -> "np.ndarray":
    """Score weighted conjugate pairs, rejecting nonpositive reconstructions."""
    levels, phase_count = _integer(levels, 2), _integer(phase_count, 4)
    dimension = 9 * levels**2
    modes = _array(modes, (dimension, dimension + 1))
    decay_limit, frequency_limit = map(_scalar, (decay_limit, frequency_limit))
    if min(decay_limit, frequency_limit) <= 0:
        raise ValueError("spectral limits must be positive")
    stationary = _reduce_qubits(modes[0, 1:], levels)
    _physical_density(stationary)
    records = []
    for mode in modes[1:]:
        decay, frequency = -mode[0].real, mode[0].imag
        if not (0 < decay < decay_limit and 1e-9 < frequency < frequency_limit):
            continue
        contribution = _reduce_qubits(mode[1:], levels)
        scores = []
        for k in range(phase_count):
            phase = 2 * np.pi * k / phase_count
            oscillation = np.exp(1j * phase) * contribution
            density = stationary + oscillation + oscillation.conj().T
            density = (density + density.conj().T) / 2
            if np.linalg.eigvalsh(density).min() >= -1e-10:
                scores.append(_concurrence(density))
        if scores:
            records.append([decay, frequency, max(scores)])
    if not records:
        raise ValueError("no admissible transient pair in the spectral window")
    maximum = max(record[2] for record in records)
    tied = [record for record in records if record[2] >= maximum - 1e-10]
    return np.asarray(min(tied, key=lambda record: record[1]), dtype=float)

import numpy as np
from scipy.linalg import expm


def integrate_local_half_steps(
    omega: float, amplitude: float, drive_frequency: float, intervals: int, phase: float
) -> "np.ndarray":
    """Use exact integrals for two generally unequal local half intervals."""
    omega, amplitude, drive_frequency, phase = map(
        _scalar, (omega, amplitude, drive_frequency, phase)
    )
    intervals = _integer(intervals, 4)
    if min(omega, drive_frequency) <= 0 or amplitude < 0:
        raise ValueError("positive frequencies and nonnegative amplitude required")
    dt = 2 * np.pi / (intervals * drive_frequency)
    jx, _ = _spin_matrices()
    result = np.empty((intervals, 2, 9, 9), dtype=complex)
    for j in range(intervals):
        for half in range(2):
            start = phase + (j + half / 2) * 2 * np.pi / intervals
            end = start + np.pi / intervals
            angle = omega * dt / 2 + amplitude / drive_frequency * (
                np.sin(end) - np.sin(start)
            )
            unitary = expm(-1j * angle * jx)
            result[j, half] = np.kron(unitary, unitary.conj())
    return result

import numpy as np
from scipy.linalg import expm


def contract_auxiliary_cycle(
    bath_generator: "np.ndarray", half_steps: "np.ndarray", dt: float, levels: int
) -> "np.ndarray":
    """Preserve auxiliary memory through both dressing and period contraction."""
    levels = _integer(levels, 2)
    dimension = 9 * levels**2
    bath_generator = _array(bath_generator, (dimension, dimension))
    half_steps = _array(half_steps)
    dt = _scalar(dt)
    if (
        half_steps.ndim != 4
        or half_steps.shape[1:] != (2, 9, 9)
        or len(half_steps) < 4
        or dt <= 0
    ):
        raise ValueError("invalid half-step stack or nonpositive dt")
    count = len(half_steps)
    result = np.empty((count + 1, dimension, dimension), dtype=complex)
    bath = expm(dt * bath_generator)
    cycle = np.eye(dimension, dtype=complex)
    for j, (early, late) in enumerate(half_steps):
        result[j] = (
            np.kron(late, np.eye(levels**2)) @ bath @ np.kron(early, np.eye(levels**2))
        )
        cycle = result[j] @ cycle
    result[-1] = cycle
    return result

import numpy as np


def recover_periodic_qubit_states(
    cycle_stack: "np.ndarray", levels: int
) -> "np.ndarray":
    """Normalize once in auxiliary space, then trace each micromotion state."""
    levels = _integer(levels, 2)
    dimension = 9 * levels**2
    cycle_stack = _array(cycle_stack)
    if (
        cycle_stack.ndim != 3
        or cycle_stack.shape[1:] != (dimension, dimension)
        or len(cycle_stack) < 5
    ):
        raise ValueError(
            "cycle_stack must contain at least four channels and their product"
        )
    trace = _trace_row(levels)
    uniform = trace / (3 * levels)
    system = np.eye(dimension) - cycle_stack[-1] + np.outer(uniform, trace)
    if np.linalg.cond(system) > 1e12:
        raise ValueError("the normalized periodic state is not uniquely resolved")
    initial = np.linalg.solve(system, uniform)
    state = initial.copy()
    result = []
    for channel in cycle_stack[:-1]:
        reduced = _reduce_qubits(state, levels)
        _physical_density(reduced)
        result.append(reduced)
        state = channel @ state
    if np.linalg.norm(state - initial) > 1e-8:
        raise ValueError("joint micromotion fails cycle closure")
    return np.asarray(result)

import numpy as np


def _mean_driven_concurrence(
    bath_generator, omega, levels, amplitude, frequency, intervals, phase
):
    local = integrate_local_half_steps(
        omega, amplitude, frequency, intervals, phase
    )
    dt = 2 * np.pi / (intervals * frequency)
    cycle = contract_auxiliary_cycle(bath_generator, local, dt, levels)
    states = recover_periodic_qubit_states(cycle, levels)
    return float(np.mean([_concurrence(state) for state in states]))


def evaluate_stabilization_grid(
    bath_generator: "np.ndarray",
    omega: float,
    levels: int,
    amplitudes: "np.ndarray",
    frequencies: "np.ndarray",
    intervals: int,
    phase: float,
) -> "np.ndarray":
    """Average concurrence before combining the two time resolutions."""
    levels, intervals = _integer(levels, 2), _integer(intervals, 4)
    bath_generator = _array(bath_generator, (9 * levels**2, 9 * levels**2))
    amplitudes, frequencies = (
        _array(amplitudes, real=True),
        _array(frequencies, real=True),
    )
    omega, phase = _scalar(omega), _scalar(phase)
    if (
        amplitudes.ndim != 1
        or frequencies.ndim != 1
        or not len(amplitudes)
        or not len(frequencies)
        or np.min(amplitudes) < 0
        or np.min(frequencies) <= 0
        or omega <= 0
    ):
        raise ValueError("nonempty physical amplitude and frequency grids are required")
    result = np.empty((len(amplitudes), len(frequencies)))
    for row, amplitude in enumerate(amplitudes):
        for column, frequency in enumerate(frequencies):
            coarse = _mean_driven_concurrence(
                bath_generator, omega, levels, amplitude, frequency, intervals, phase
            )
            fine = _mean_driven_concurrence(
                bath_generator,
                omega,
                levels,
                amplitude,
                frequency,
                2 * intervals,
                phase,
            )
            result[row, column] = (4 * fine - coarse) / 3
    return result

import numpy as np


def compute_stabilized_entanglement(
    levels: int,
    omega: float,
    bath_frequency: float,
    coupling: float,
    decay: float,
    amplitudes: "np.ndarray",
    detunings: "np.ndarray",
    phase_count: int,
    decay_limit: float,
    frequency_limit: float,
    intervals: int,
    phase: float,
) -> float:
    """Couple the transient-mode selection to the full driven stationary calculation."""
    generators = build_collective_generators(
        levels, omega, bath_frequency, coupling, decay
    )
    initial = np.zeros(9 * levels**2)
    initial[0] = 1
    modes = resolve_quench_modes(generators[1], initial, levels)
    selection = select_entangling_mode(
        modes, levels, phase_count, decay_limit, frequency_limit
    )
    detunings = _array(detunings, real=True)
    if detunings.ndim != 1 or not len(detunings):
        raise ValueError("detunings must be a nonempty vector")
    frequencies = selection[1] + selection[0] * detunings
    estimates = evaluate_stabilization_grid(
        generators[0], omega, levels, amplitudes, frequencies, intervals, phase
    )
    return float(np.max(estimates))
SCICODE_GOLD_EOF
