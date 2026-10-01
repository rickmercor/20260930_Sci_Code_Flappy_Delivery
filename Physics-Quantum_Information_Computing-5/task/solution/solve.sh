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
from math import comb, factorial


def _integer(value, lower, upper, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} is outside its supported range")
    return int(value)


def _jet(value, shape, name):
    value = np.asarray(value, dtype=complex)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} has invalid shape or nonfinite entries")
    return value.copy()


def _state(state, normalized=True):
    state = np.asarray(state, dtype=complex)
    if state.ndim != 2 or state.shape[0] not in (3, 9):
        raise ValueError("state must have 3 or 9 derivative rows")
    d = state.shape[1]
    if d < 2 or d > 64 or d & (d - 1) or not np.all(np.isfinite(state)):
        raise ValueError("state dimension must be a power of two from 2 to 64")
    mass = float(np.vdot(state[0], state[0]).real)
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("zeroth-order state must have positive finite mass")
    if normalized and abs(mass - 1) > 1e-9:
        raise ValueError("zeroth-order state must be normalized")
    return state.copy()


def _unitary_jet(value, shape, name):
    value = _jet(value, shape, name)
    if (
        np.max(abs(value[0].conj().swapaxes(-1, -2) @ value[0] - np.eye(shape[-1])))
        > 1e-9
    ):
        raise ValueError(f"{name} must be unitary at order zero")
    return value


def _indices(qubits, n, length=None):
    qubits = np.asarray(qubits)
    if qubits.ndim != 1 or qubits.dtype.kind not in "iu":
        raise ValueError("qubits must be an integer vector")
    if len(qubits) not in (1, 2) or length is not None and len(qubits) != length:
        raise ValueError("unsupported local dimension")
    if (
        len(set(qubits.tolist())) != len(qubits)
        or np.any(qubits < 0)
        or np.any(qubits >= n)
    ):
        raise ValueError("qubits must be distinct and in range")
    return tuple(int(q) for q in qubits)


def _controls(max_passes, gap_tol, accept_tol):
    max_passes = _integer(max_passes, 1, 3, "max_passes")
    for value in (gap_tol, accept_tol):
        if (
            not np.isscalar(value)
            or np.iscomplexobj(value)
            or not np.isfinite(value)
            or not 0 <= value <= 1e-8
        ):
            raise ValueError("tolerances must be finite real numbers in [0, 1e-8]")
    return max_passes, float(gap_tol), float(accept_tol)


def _multi(length):
    return (
        [(j, 0) for j in range(3)]
        if length == 3
        else [(a, b) for a in range(3) for b in range(3)]
    )


def _terms(length, row):
    indices = _multi(length)
    a, b = indices[row]
    for i, (c, d) in enumerate(indices):
        if c <= a and d <= b:
            yield i, indices.index((a - c, b - d)), comb(a, c) * comb(b, d)


def _product(a, b, operation):
    return np.array(
        [
            sum(weight * operation(a[i], b[j]) for i, j, weight in _terms(len(a), row))
            for row in range(len(a))
        ]
    )


def _matmul(a, b):
    return _product(a, b, lambda x, y: x @ y)


def _kron(a, b):
    return _product(a, b, np.kron)


def _scale(a, b):
    return _product(a, b, lambda x, y: x * y)


def _one(length):
    result = np.zeros(length)
    result[0] = 1
    return result


def _power(a, p):
    # Binomial series terminates in the truncated derivative algebra.
    delta = a / a[0]
    delta = delta.copy()
    delta[0] = 0
    term = _one(len(a))
    result = term.copy()
    coefficient = 1.0
    for degree in range(1, max(sum(x) for x in _multi(len(a))) + 1):
        term = _scale(term, delta)
        coefficient *= (p - degree + 1) / degree
        result = result + coefficient * term
    return a[0] ** p * result


def _norm(state):
    return _product(state, state, np.vdot).real


def _pr(state):
    weights = _scale(state.conj(), state).real
    ipr = _scale(weights, weights).sum(axis=1)
    return _scale(_power(_norm(state), 2), _power(ipr, -1))


def _apply(state, matrix, qubits):
    n = state.size.bit_length() - 1
    axes = [n - 1 - q for q in qubits]
    order = axes + [a for a in range(n) if a not in axes]
    blocks = state.reshape((2,) * n).transpose(order)
    result = matrix @ blocks.reshape(2 ** len(qubits), -1)
    return result.reshape((2,) * n).transpose(np.argsort(order)).reshape(-1)


def _apply_jet(state, matrix, qubits):
    return _product(matrix, state, lambda m, v: _apply(v, m, qubits))


def propagate_frame_jets(
    state: "np.ndarray", bases: "np.ndarray", gate: "np.ndarray", qubits: "np.ndarray"
) -> "np.ndarray":
    state = _state(state)
    n = state.shape[1].bit_length() - 1
    bases = _unitary_jet(bases, (len(state), n, 2, 2), "bases")
    gate = _unitary_jet(gate, (len(state), 4, 4), "gate")
    q = _indices(qubits, n, 2)
    frame = _kron(bases[:, q[0]], bases[:, q[1]])
    working = _matmul(_matmul(frame.conj().swapaxes(-1, -2), gate), frame)
    return _apply_jet(state, working, q)

import numpy as np


def reduced_density_jets(
    state: "np.ndarray", qubits: "np.ndarray"
) -> "np.ndarray":
    state = _state(state)
    n = state.shape[1].bit_length() - 1
    q = _indices(qubits, n)
    axes = [n - 1 - j for j in q]
    order = axes + [j for j in range(n) if j not in axes]
    blocks = np.array(
        [v.reshape((2,) * n).transpose(order).reshape(2 ** len(q), -1) for v in state]
    )
    return _matmul(blocks, blocks.conj().swapaxes(-1, -2))

import numpy as np


def natural_frame_jets(
    rdm: "np.ndarray", gap_tol: float = 1e-10
) -> "np.ndarray":
    rdm = np.asarray(rdm, dtype=complex)
    if rdm.shape not in ((3, 2, 2), (3, 4, 4), (9, 2, 2), (9, 4, 4)):
        raise ValueError("rdm must have J derivative rows and local dimension 2 or 4")
    rdm = _jet(rdm, rdm.shape, "rdm")
    _controls(1, gap_tol, 0.0)
    if np.max(abs(rdm - rdm.conj().swapaxes(-1, -2))) > 1e-9:
        raise ValueError("every rdm derivative must be Hermitian")
    values, vectors = np.linalg.eigh(rdm[0])
    if values[0] < -1e-9 or abs(np.trace(rdm[0]) - 1) > 1e-9:
        raise ValueError("base rdm must be positive semidefinite with unit trace")
    length, d = len(rdm), len(values)
    result = np.zeros_like(rdm)
    if np.min(np.diff(values)) <= gap_tol:
        result[0] = np.eye(d)
        return result
    values, vectors = values[::-1], vectors[:, ::-1]
    indices = _multi(length)
    factors = np.array([factorial(a) * factorial(b) for a, b in indices])
    coefficients = rdm / factors[:, None, None]
    for column in range(d):
        v = vectors[:, column].copy()
        pivot = int(np.argmax(abs(v)))
        v *= np.conj(v[pivot]) / abs(v[pivot])
        wave = np.zeros((length, d), complex)
        energy = np.zeros(length, complex)
        wave[0], energy[0] = v, values[column]
        for row in sorted(range(1, length), key=lambda r: sum(indices[r])):
            rhs = np.zeros(d, complex)
            for i, j, _ in _terms(length, row):
                if i:
                    rhs += coefficients[i] @ wave[j]
                    if j:
                        rhs -= energy[i] * wave[j]
            energy[row] = np.vdot(v, rhs)
            for other in range(d):
                if other != column:
                    u = vectors[:, other]
                    wave[row] += u * np.vdot(u, rhs) / (values[column] - values[other])
        wave *= factors[:, None]
        wave = _scale(wave, _power(_norm(wave), -0.5))
        pivot_jet = wave[:, pivot]
        phase = _scale(
            pivot_jet.conj(), _power(_scale(pivot_jet.conj(), pivot_jet).real, -0.5)
        )
        result[:, :, column] = _scale(wave, phase)
    return result

import numpy as np


def compress_state_jets(state: "np.ndarray", k: int) -> tuple:
    state = _state(state, normalized=False)
    k = _integer(k, 1, state.shape[1], "k")
    support = np.lexsort((np.arange(state.shape[1]), -(abs(state[0]) ** 2)))[:k]
    retained = np.zeros_like(state)
    retained[:, support] = state[:, support]
    mass = _norm(retained)
    result = _scale(retained, _power(mass, -0.5))
    return result, mass, _pr(result)

import numpy as np


def guarded_single_jet(
    state: "np.ndarray",
    bases: "np.ndarray",
    frame: "np.ndarray",
    qubit: int,
    k: int,
    accept_tol: float = 1e-12,
) -> tuple:
    state = _state(state)
    n = state.shape[1].bit_length() - 1
    bases = _unitary_jet(bases, (len(state), n, 2, 2), "bases")
    frame = _unitary_jet(frame, (len(state), 2, 2), "frame")
    qubit = _integer(qubit, 0, n - 1, "qubit")
    k = _integer(k, 1, state.shape[1], "k")
    _controls(1, 0.0, accept_tol)
    trial = _apply_jet(state, frame.conj().swapaxes(-1, -2), (qubit,))
    trial, mass, pr = compress_state_jets(trial, k)
    if _pr(state)[0] - pr[0] > accept_tol:
        bases[:, qubit] = _matmul(bases[:, qubit], frame)
        return trial, bases, mass, 1
    return state, bases, _one(len(state)), 0

import numpy as np


def snapshot_jet_sweep(
    state: "np.ndarray",
    bases: "np.ndarray",
    k: int,
    max_passes: int = 3,
    gap_tol: float = 1e-10,
    accept_tol: float = 1e-12,
) -> tuple:
    state = _state(state)
    n = state.shape[1].bit_length() - 1
    bases = _unitary_jet(bases, (len(state), n, 2, 2), "bases")
    k = _integer(k, 1, state.shape[1], "k")
    max_passes, gap_tol, accept_tol = _controls(max_passes, gap_tol, accept_tol)
    retained, total = _one(len(state)), 0
    for _ in range(max_passes):
        frames = [
            natural_frame_jets(
                reduced_density_jets(state, np.array([j])), gap_tol
            )
            for j in range(n)
        ]
        count = 0
        for j, frame in enumerate(frames):
            state, bases, mass, accepted = guarded_single_jet(
                state, bases, frame, j, k, accept_tol
            )
            retained = _scale(retained, mass)
            count += accepted
        total += count
        if count == 0:
            break
    return state, bases, retained, total

import numpy as np


def transient_pair_jet(
    state: "np.ndarray",
    qubits: "np.ndarray",
    k: int,
    gap_tol: float = 1e-10,
    accept_tol: float = 1e-12,
) -> tuple:
    state = _state(state)
    n = state.shape[1].bit_length() - 1
    q = _indices(qubits, n, 2)
    k = _integer(k, 1, state.shape[1], "k")
    _controls(1, gap_tol, accept_tol)
    frame = natural_frame_jets(
        reduced_density_jets(state, np.array(q)), gap_tol
    )
    rotated = _apply_jet(state, frame.conj().swapaxes(-1, -2), q)
    compressed, mass1, _ = compress_state_jets(rotated, k)
    undone = _apply_jet(compressed, frame, q)
    candidate, mass2, pr = compress_state_jets(undone, k)
    if _pr(state)[0] - pr[0] > accept_tol:
        return candidate, _scale(mass1, mass2), 1
    return state, _one(len(state)), 0

import numpy as np


def circuit_response_jets(
    initial: "np.ndarray",
    gates: "np.ndarray",
    pairs: "np.ndarray",
    k: int,
    max_passes: int = 3,
) -> tuple:
    state = _state(initial)
    n = state.shape[1].bit_length() - 1
    _integer(n, 2, 6, "qubit count")
    gates = np.asarray(gates, dtype=complex)
    if gates.ndim != 4 or gates.shape[1:] != (len(state), 4, 4) or len(gates) > 24:
        raise ValueError(
            "gates must have shape (M, J, 4, 4), matching the initial jet, with M at most 24"
        )
    pairs = np.asarray(pairs)
    if pairs.shape != (len(gates), 2) or pairs.dtype.kind not in "iu":
        raise ValueError("pairs must have shape (M, 2) and integer entries")
    k = _integer(k, 1, state.shape[1], "k")
    _controls(max_passes, 1e-10, 1e-12)
    bases = np.zeros((len(state), n, 2, 2), dtype=complex)
    bases[0] = np.eye(2)
    retained = _one(len(state))
    brick = [(j, j + 1) for offset in (0, 1) for j in range(offset, n - 1, 2)]
    for gate, pair in zip(gates, pairs):
        state = propagate_frame_jets(state, bases, gate, pair)
        state, mass, _ = compress_state_jets(state, k)
        retained = _scale(retained, mass)
        state, bases, mass, _ = snapshot_jet_sweep(state, bases, k, max_passes)
        retained = _scale(retained, mass)
        for q in brick:
            state, mass, _ = transient_pair_jet(state, np.array(q), k)
            retained = _scale(retained, mass)
    return state, bases, retained

import numpy as np


def _rotation(angle, slope, pauli, cross_slope=None):
    unitary = np.cos(angle / 2) * np.eye(len(pauli)) - 1j * np.sin(angle / 2) * pauli
    if cross_slope is None:
        indices = _multi(3)
        cross_slope = 0.0
    else:
        indices = _multi(9)
    generator = -0.5j * pauli
    return np.array(
        [
            slope**a
            * cross_slope**b
            * np.linalg.matrix_power(generator, a + b)
            @ unitary
            for a, b in indices
        ]
    )


def _gate_jets(angles, slopes, cross_slopes=None):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1.0, -1.0])
    gates = []
    for row, (a, b) in enumerate(zip(angles, slopes)):
        c = [None] * 5 if cross_slopes is None else cross_slopes[row]
        left = _rotation(a[2], b[2], np.kron(z, z), c[2])
        middle = _kron(_rotation(a[0], b[0], y, c[0]), _rotation(a[1], b[1], x, c[1]))
        right = _kron(_rotation(a[3], b[3], z, c[3]), _rotation(a[4], b[4], y, c[4]))
        gates.append(_matmul(_matmul(left, middle), right))
    return np.array(gates).reshape((-1, 3 if cross_slopes is None else 9, 4, 4))


def _fidelity_jet(exact, physical):
    overlap = _product(exact, physical, np.vdot)
    return _scale(overlap.conj(), overlap).real


def fidelity_curvature(
    initial: "np.ndarray",
    angles: "np.ndarray",
    slopes: "np.ndarray",
    pairs: "np.ndarray",
    k: int,
    max_passes: int = 3,
    cross_slopes: "np.ndarray | None" = None,
) -> float:
    initial = np.asarray(initial, dtype=complex)
    if initial.ndim != 1:
        raise ValueError("initial must be a vector")
    state = np.zeros((3 if cross_slopes is None else 9, len(initial)), dtype=complex)
    state[0] = initial
    state = _state(state)
    angles, slopes = np.asarray(angles), np.asarray(slopes)
    if (
        angles.ndim != 2
        or angles.shape[1] != 5
        or angles.shape != slopes.shape
        or np.iscomplexobj(angles)
        or np.iscomplexobj(slopes)
    ):
        raise ValueError("angles and slopes must be real arrays of shape (M, 5)")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(slopes)):
        raise ValueError("angles and slopes must be finite")
    if cross_slopes is not None:
        cross_slopes = np.asarray(cross_slopes)
        if (
            cross_slopes.shape != angles.shape
            or np.iscomplexobj(cross_slopes)
            or not np.all(np.isfinite(cross_slopes))
        ):
            raise ValueError("cross_slopes must be finite real and match angles")
    gates = _gate_jets(angles, slopes, cross_slopes)
    approximate, bases, _ = circuit_response_jets(
        state, gates, pairs, k, max_passes
    )
    exact = state.copy()
    for gate, pair in zip(gates, pairs):
        exact = _apply_jet(exact, gate, tuple(pair))
    for j in range(bases.shape[1]):
        approximate = _apply_jet(approximate, bases[:, j], (j,))
    return float(_fidelity_jet(exact, approximate)[-1])
SCICODE_GOLD_EOF
