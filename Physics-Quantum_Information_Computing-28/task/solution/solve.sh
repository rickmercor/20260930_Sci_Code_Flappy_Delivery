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


def _binary(a, ndim, name):
    a = np.asarray(a)
    if a.ndim != ndim or not np.all((a == 0) | (a == 1)):
        raise ValueError(f"{name} must be a binary array with {ndim} axes")
    return a.astype(np.int64, copy=True)


def _real(a, name):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        raise ValueError(f"{name} must be real")
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a.copy()


def _int(a, lo, hi, name):
    if (
        isinstance(a, (bool, np.bool_))
        or not isinstance(a, (int, np.integer))
        or not lo <= a <= hi
    ):
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return int(a)


def _prob(a, hi, name):
    a = _real(a, name)
    if a.ndim or not 0 < a <= hi:
        raise ValueError(f"{name} must be a scalar in (0, {hi}]")
    return float(a)


def _words(n):
    return (np.arange(1 << n)[:, None] >> np.arange(n - 1, -1, -1)) & 1


def _rref(a):
    a = a.copy()
    pivots = []
    row = 0
    for col in range(a.shape[1]):
        found = np.flatnonzero(a[row:, col])
        if not found.size:
            continue
        pivot = row + found[0]
        a[[row, pivot]] = a[[pivot, row]]
        for i in range(a.shape[0]):
            if i != row and a[i, col]:
                a[i] ^= a[row]
        pivots.append(col)
        row += 1
        if row == a.shape[0]:
            break
    return a, pivots


def _inverse(a):
    n = len(a)
    b, pivots = _rref(np.concatenate((a, np.eye(n, dtype=int)), axis=1))
    if pivots[:n] != list(range(n)):
        raise ValueError("binary basis must be invertible")
    return b[:, n:]


def _quotient(q, m):
    q = _binary(q, 3, "quotient")
    m = _int(m, 1, 3, "check_count")
    if q.shape[0] != 2 or q.shape[1] != q.shape[2] or not m + 2 <= q.shape[1] <= 10:
        raise ValueError("quotient must have shape (2,n,n), m+2 <= n <= 10")
    if not np.array_equal((q[0] @ q[1]) % 2, np.eye(q.shape[1], dtype=int)):
        raise ValueError("quotient matrices must be inverses over GF(2)")
    return q, m


def build_binary_quotient(
    checks: "np.ndarray", stabilizers: "np.ndarray", logicals: "np.ndarray"
) -> "np.ndarray":
    h = _binary(checks, 2, "checks")
    s = _binary(stabilizers, 2, "stabilizers")
    logical_rows = _binary(logicals, 2, "logicals")
    m, n = h.shape
    if (
        not 1 <= m <= 3
        or not m + 2 <= n <= 10
        or s.shape != (n - m - 2, n)
        or logical_rows.shape != (2, n)
    ):
        raise ValueError("incompatible check, stabilizer and two-logical dimensions")
    if np.any((h @ np.concatenate((s, logical_rows)).T) % 2):
        raise ValueError("stabilizers and logicals must preserve the syndrome")
    _, pivots = _rref(h)
    if len(pivots) != m:
        raise ValueError("checks must be independent")
    pure = np.zeros((m, n), dtype=int)
    pure[:, pivots] = _inverse(h[:, pivots]).T
    basis = np.concatenate((pure, logical_rows, s))
    inverse = _inverse(basis)
    return np.stack((basis, inverse))

import numpy as np


def build_detector_constraints(
    quotient: "np.ndarray", check_count: int, rounds: int
) -> "np.ndarray":
    q, m = _quotient(quotient, check_count)
    tmax = _int(rounds, 1, 8, "rounds")
    n = q.shape[1]
    a = np.zeros((tmax * m + 2, tmax * n + (tmax - 1) * m), dtype=int)
    for t in range(tmax):
        a[t * m : (t + 1) * m, t * n : (t + 1) * n] = q[1, :, :m].T
        a[-2:, t * n : (t + 1) * n] = q[1, :, m : m + 2].T
        for j in (t - 1, t):
            if 0 <= j < tmax - 1:
                a[t * m : (t + 1) * m, tmax * n + j * m : tmax * n + (j + 1) * m] ^= (
                    np.eye(m, dtype=int)
                )
    return a

import numpy as np


def build_conditional_branches(
    quotient: "np.ndarray",
    check_count: int,
    opposite: "np.ndarray",
    p: float,
    counts: "np.ndarray",
    initial: bool,
) -> "np.ndarray":
    q, m = _quotient(quotient, check_count)
    opposite = _binary(opposite, 2, "opposite")
    tmax, n = opposite.shape
    if (
        not 1 <= tmax <= 8
        or n != q.shape[1]
        or not isinstance(initial, (bool, np.bool_))
    ):
        raise ValueError("opposite shape or initial flag is invalid")
    p = _prob(p, 0.75, "p")
    c = _real(counts, "counts")
    if c.shape != (2, 2) or np.any(c < 0) or np.any(c.sum(axis=1) <= 0):
        raise ValueError("counts must be nonnegative 2 by 2 with positive row sums")
    a, b = c[1, 1] / c[1].sum(), c[0, 0] / c[0].sum()
    if min(a, b) < 0.5:
        raise ValueError("both fitted reliabilities must be at least one half")
    low = p / (3 - 2 * p)
    probability = (
        np.full(opposite.shape, 2 * p / 3)
        if initial
        else np.where(opposite, a / 2 + (1 - a) * low, b * low + (1 - b) / 2)
    )
    coupling = 0.5 * np.log((1 - probability) / probability)
    words = _words(n)
    labels = ((words @ q[1, :, : m + 2]) % 2) @ (1 << np.arange(m + 1, -1, -1))
    result = np.empty((tmax, 1 << n, n + 2))
    result[:, :, 0] = labels
    result[:, :, 1] = coupling @ (2 * words - 1).T
    result[:, :, 2:] = words
    return result

import numpy as np


def _detector_layout(constraints, detectors):
    a = _binary(constraints, 2, "constraints")
    d = _binary(detectors, 2, "detectors")
    tmax, m = d.shape
    if not 1 <= tmax <= 8 or not 1 <= m <= 3 or a.shape[0] != tmax * m + 2:
        raise ValueError("detector dimensions are invalid")
    n, rem = divmod(a.shape[1] - (tmax - 1) * m, tmax)
    if rem or not m + 2 <= n <= 10:
        raise ValueError("constraint column count is invalid")
    h = a[:m, :n]
    logical = a[-2:, :n]
    expected = np.zeros_like(a)
    for t in range(tmax):
        expected[t * m : (t + 1) * m, t * n : (t + 1) * n] = h
        expected[-2:, t * n : (t + 1) * n] = logical
        for j in (t - 1, t):
            if 0 <= j < tmax - 1:
                expected[
                    t * m : (t + 1) * m, tmax * n + j * m : tmax * n + (j + 1) * m
                ] ^= np.eye(m, dtype=int)
    if (
        not np.array_equal(a, expected)
        or len(_rref(np.concatenate((h, logical)))[1]) != m + 2
    ):
        raise ValueError(
            "constraints must encode the stated independent chain and logical rows"
        )
    return a, d, tmax, m, n


def _emissions(detectors, pm):
    tmax, m = detectors.shape
    records = np.bitwise_xor.accumulate(detectors, axis=0)
    syndrome = _words(m)[np.arange(1 << (m + 2)) >> 2]
    errors = records[:, None, :] ^ syndrome[None, :, :]
    coupling = 0.5 * np.log((1 - pm) / pm)
    cost = coupling * np.sum(2 * errors - 1, axis=2)
    cost[-1] = 0
    return records, errors, cost


def _dyadic_costs(data, measurement, tolerance):
    shapes = (data.shape, measurement.shape)
    values = np.concatenate((data.ravel(), measurement.ravel(), [float(tolerance)]))
    ratios = [float(v).as_integer_ratio() for v in values]
    exponent = max(den.bit_length() - 1 for _, den in ratios)
    integers = [num << (exponent - (den.bit_length() - 1)) for num, den in ratios]
    cut = data.size
    last = cut + measurement.size
    return (
        np.array(integers[:cut], dtype=object).reshape(shapes[0]),
        np.array(integers[cut:last], dtype=object).reshape(shapes[1]),
        integers[-1],
        1 << exponent,
    )


def solve_spacetime_sectors(
    constraints: "np.ndarray",
    detectors: "np.ndarray",
    branches: "np.ndarray",
    pm: float,
    tie_tol: float,
) -> "np.ndarray":
    a, d, tmax, m, n = _detector_layout(constraints, detectors)
    br = _real(branches, "branches")
    pm = _prob(pm, 0.5, "pm")
    tol = _real(tie_tol, "tie_tol")
    if tol.ndim or tol < 0:
        raise ValueError("tie_tol must be a nonnegative finite scalar")
    words = _words(n)
    label_bits = np.concatenate((a[:m, :n], a[-2:, :n]))
    labels = ((words @ label_bits.T) % 2) @ (1 << np.arange(m + 1, -1, -1))
    if (
        br.shape != (tmax, 1 << n, n + 2)
        or not np.all(br[:, :, 0] == labels)
        or not np.all(br[:, :, 2:] == words)
    ):
        raise ValueError(
            "branches must contain the exact ordered words and quotient labels"
        )
    records, measurement, cost = _emissions(d, pm)
    states = np.arange(1 << (m + 2))
    terminal_syndrome = int(records[-1] @ (1 << np.arange(m - 1, -1, -1)))
    data_cost, measurement_cost, window, scale = _dyadic_costs(br[:, :, 1], cost, tol)
    reduced = np.full((tmax, len(states)), np.inf, dtype=object)
    for t in range(tmax):
        np.minimum.at(reduced[t], labels, data_cost[t])
    transition = states[:, None] ^ states[None, :]
    suffixes = np.full((4, tmax + 1, len(states)), np.inf, dtype=object)
    for logical in range(4):
        suffixes[logical, -1, 4 * terminal_syndrome + logical] = 0
        for t in range(tmax - 1, -1, -1):
            suffixes[logical, t] = np.min(
                reduced[t, transition]
                + measurement_cost[t, None, :]
                + suffixes[logical, t + 1, None, :],
                axis=1,
            )
    minima = suffixes[:, 0, 0]
    global_minimum = int(minima.min())
    logical = int(np.flatnonzero(minima <= global_minimum + window)[0])
    suffix = suffixes[logical]
    state = 0
    accumulated = 0
    chosen = []
    measures = []
    for t in range(tmax):
        next_states = state ^ labels
        trial = (
            accumulated
            + data_cost[t]
            + measurement_cost[t, next_states]
            + suffix[t + 1, next_states]
        )
        allowed = np.flatnonzero(trial <= global_minimum + window)
        if not allowed.size:
            raise ValueError("no numerically admissible reconstruction")
        word = int(allowed[0])
        state = int(next_states[word])
        accumulated += data_cost[t, word] + measurement_cost[t, state]
        chosen.extend(words[word])
        if t < tmax - 1:
            measures.extend(measurement[t, state])
    errors = np.array(chosen + measures, dtype=int)
    target = np.concatenate((d.ravel(), _words(2)[logical]))
    if not np.array_equal((a @ errors) % 2, target):
        raise ValueError("reconstructed path violates its detector or logical sector")
    return np.concatenate(
        ([logical, accumulated / scale], [int(v) / scale for v in minima], errors)
    )

import numpy as np


def iterate_spacetime_recovery(
    quotients: "np.ndarray",
    constraints: "np.ndarray",
    detectors: "np.ndarray",
    p: float,
    pm: float,
    counts: "np.ndarray",
    max_sweeps: int,
    tie_tol: float,
) -> "np.ndarray":
    qq = _binary(quotients, 4, "quotients")
    aa = _binary(constraints, 3, "constraints")
    dd = _binary(detectors, 3, "detectors")
    if qq.shape[0] != 2 or aa.shape[0] != 2 or dd.shape[0] != 2:
        raise ValueError("channel axis must contain X then Z")
    _, _, tmax, m, n = _detector_layout(aa[0], dd[0])
    if dd.shape != (2, tmax, m) or qq.shape != (2, 2, n, n):
        raise ValueError("channel dimensions disagree")
    for channel in range(2):
        expected = build_detector_constraints(qq[channel], m, tmax)
        if not np.array_equal(aa[channel], expected):
            raise ValueError("constraints disagree with quotient")
    budget = _int(max_sweeps, 1, 8, "max_sweeps")
    tol = _real(tie_tol, "tie_tol")
    if tol.ndim or tol < 0:
        raise ValueError("tie_tol must be a nonnegative finite scalar")
    previous = np.zeros((2, tmax, n), dtype=int)
    history = []
    for sweep in range(budget):
        pair = []
        current = previous.copy()
        for channel in range(2):
            opposite = previous[1] if channel == 0 else current[0]
            branches = build_conditional_branches(
                qq[channel], m, opposite, p, counts, sweep == 0 and channel == 0
            )
            record = solve_spacetime_sectors(
                aa[channel], dd[channel], branches, pm, tie_tol
            )
            pair.append(record)
            current[channel] = record[6 : 6 + tmax * n].reshape(tmax, n).astype(int)
        history.append(np.stack(pair))
        if sweep > 0 and np.array_equal(current, previous):
            break
        previous = current
    return np.stack(history)

import numpy as np


def build_joint_transition_jets(
    quotients: "np.ndarray", check_count: int, p: float
) -> "np.ndarray":
    qq = _binary(quotients, 4, "quotients")
    if qq.shape[0] != 2:
        raise ValueError("quotients must have two channels")
    qx, m = _quotient(qq[0], check_count)
    qz, _ = _quotient(qq[1], check_count)
    if qx.shape != qz.shape:
        raise ValueError("channel quotient dimensions disagree")
    p = _prob(p, 0.75, "p")
    n = qx.shape[1]
    width = m + 2
    count = 1 << (2 * width)
    weights = 1 << np.arange(width - 1, -1, -1)
    xlabels = qx[1, :, :width] @ weights
    zlabels = qz[1, :, :width] @ weights
    result = np.zeros((3, count))
    result[0, 0] = 1
    states = np.arange(count)
    for edge in range(n):
        shifts = (
            0,
            int(xlabels[edge]) << width,
            (int(xlabels[edge]) << width) ^ int(zlabels[edge]),
            int(zlabels[edge]),
        )
        updated = np.zeros_like(result)
        for pauli, shift in enumerate(shifts):
            probability = 1 - p if pauli == 0 else p / 3
            derivative = -1.0 if pauli == 0 else 1 / 3
            values = result[:, states ^ shift]
            updated[0] += probability * values[0]
            updated[1] += probability * values[1] + derivative * values[0]
            updated[2] += probability * values[2] + 2 * derivative * values[1]
        result = updated
    return result

import numpy as np


def _xor_convolution(a, b):
    # Positive zeroth-order sums avoid cancellation in rare detector fibers.
    index = np.arange(a.size)
    out = np.zeros_like(a)
    for shift in range(a.size):
        if b[shift] != 0:
            out += b[shift] * a[index ^ shift]
    return out


def propagate_conditioned_jets(
    constraints: "np.ndarray", detectors: "np.ndarray", kernel: "np.ndarray", pm: float
) -> "np.ndarray":
    aa = _binary(constraints, 3, "constraints")
    dd = _binary(detectors, 3, "detectors")
    if aa.shape[0] != 2 or dd.shape[0] != 2:
        raise ValueError("two channels are required")
    _, dx, tmax, m, n = _detector_layout(aa[0], dd[0])
    _, dz, tz, mz, nz = _detector_layout(aa[1], dd[1])
    if (tmax, m, n) != (tz, mz, nz):
        raise ValueError("channel dimensions disagree")
    pm = _prob(pm, 0.5, "pm")
    width = m + 2
    size = 1 << (2 * width)
    k = _real(kernel, "kernel")
    if (
        k.shape != (3, size)
        or np.any(k[0] < 0)
        or not np.allclose(k.sum(axis=1), [1, 0, 0], atol=1e-9, rtol=0)
    ):
        raise ValueError(
            "kernel must contain normalized probability and two derivative rows"
        )
    records = np.bitwise_xor.accumulate(dd, axis=1)
    states = np.arange(size)
    x = states >> width
    z = states & ((1 << width) - 1)
    sx = _words(m)[x >> 2]
    sz = _words(m)[z >> 2]
    values = np.zeros((3, size))
    values[0, 0] = 1
    for t in range(tmax):
        prediction = np.zeros_like(values)
        prediction[0] = _xor_convolution(values[0], k[0])
        prediction[1] = _xor_convolution(values[1], k[0]) + _xor_convolution(
            values[0], k[1]
        )
        prediction[2] = (
            _xor_convolution(values[2], k[0])
            + 2 * _xor_convolution(values[1], k[1])
            + _xor_convolution(values[0], k[2])
        )
        if t < tmax - 1:
            mistakes = np.sum(sx ^ records[0, t], axis=1) + np.sum(
                sz ^ records[1, t], axis=1
            )
            evidence = pm**mistakes * (1 - pm) ** (2 * m - mistakes)
        else:
            evidence = np.all(sx == records[0, t], axis=1) & np.all(
                sz == records[1, t], axis=1
            )
        weighted = prediction * evidence
        scale = weighted.sum(axis=1)
        if scale[0] <= 0:
            raise ValueError("detector record has zero evidence")
        values[0] = weighted[0] / scale[0]
        values[1] = (weighted[1] - values[0] * scale[1]) / scale[0]
        values[2] = (
            weighted[2] - 2 * values[1] * scale[1] - values[0] * scale[2]
        ) / scale[0]
    labels = 4 * (x & 3) + (z & 3)
    return np.stack(
        [np.bincount(labels, weights=row, minlength=16).reshape(4, 4) for row in values]
    )

import numpy as np


def compute_log_odds_curvature(
    checks: "np.ndarray",
    stabilizers: "np.ndarray",
    logicals: "np.ndarray",
    detectors: "np.ndarray",
    p: float,
    pm: float,
    counts: "np.ndarray",
    max_sweeps: int,
    tie_tol: float,
) -> float:
    h = _binary(checks, 3, "checks")
    s = _binary(stabilizers, 3, "stabilizers")
    logical_rows = _binary(logicals, 3, "logicals")
    d = _binary(detectors, 3, "detectors")
    if any(a.shape[0] != 2 for a in (h, s, logical_rows, d)):
        raise ValueError("all channel axes must contain X then Z")
    if d.shape[2] != h.shape[1]:
        raise ValueError("detector and check dimensions disagree")
    quotients = np.stack(
        [build_binary_quotient(h[c], s[c], logical_rows[c]) for c in range(2)]
    )
    constraints = np.stack(
        [
            build_detector_constraints(q, h.shape[1], d.shape[1])
            for q in quotients
        ]
    )
    history = iterate_spacetime_recovery(
        quotients, constraints, d, p, pm, counts, max_sweeps, tie_tol
    )
    kernel = build_joint_transition_jets(quotients, h.shape[1], p)
    posterior = propagate_conditioned_jets(constraints, d, kernel, pm)
    ix, iz = history[-1, :, 0].astype(int)
    success, first, second = posterior[:, ix, iz]
    if not 0 < success < 1:
        raise ValueError(
            "selected success probability must lie strictly between zero and one"
        )
    return float(
        -second / (success * (1 - success))
        + (1 - 2 * success) * first**2 / (success**2 * (1 - success) ** 2)
    )
SCICODE_GOLD_EOF
