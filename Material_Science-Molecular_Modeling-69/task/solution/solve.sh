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


def _finite(value, name):
    import numpy as np

    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(name + " must be finite")
    return out


def _positive(value, name, zero=False):
    out = _finite(value, name)
    if out.ndim != 0 or (out < 0 if zero else out <= 0):
        raise ValueError(name + " must be a valid nonnegative/positive scalar")
    return float(out)


def _positions(positions):
    import numpy as np

    pos = _finite(positions, "positions")
    if pos.ndim != 2 or pos.shape[0] < 2 or pos.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2), N >= 2")
    dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    if np.any(dist[np.triu_indices(len(pos), 1)] <= 0):
        raise ValueError("molecular sites must be distinct")
    return pos, dist


def build_molecular_basis(
    positions: "np.ndarray",
    homo: "np.ndarray",
    lumo: "np.ndarray",
    binding: "np.ndarray",
    j0: float,
    rj: float,
    field: "np.ndarray",
    disorder: "np.ndarray",
    rthr: float,
    direction: "np.ndarray",
) -> "np.ndarray":
    import numpy as np

    pos, dist = _positions(positions)
    n = len(pos)
    homo, lumo, binding = [
        _finite(a, "site array") for a in (homo, lumo, binding)
    ]
    disorder, field = _finite(disorder, "disorder"), _finite(field, "field")
    if any(a.shape != (n,) for a in (homo, lumo, binding)):
        raise ValueError("site arrays must have shape (N,)")
    if disorder.shape != (n, n) or field.shape != (2,) or np.any(binding < 0):
        raise ValueError("invalid disorder, field or binding")
    j0, rj, rthr = [_positive(v, "scale") for v in (j0, rj, rthr)]
    e, h = np.indices((n, n)).reshape(2, -1)
    radius = dist[e, h]
    energy = lumo[e] - homo[h] + disorder[e, h]
    energy -= np.where(e == h, binding[e], j0 / (1 + radius / rj))
    energy -= (pos[h] - pos[e]) @ field
    label = np.where(e == h, 0, np.where(radius >= rthr, 2, 1))
    direction = _finite(direction, "direction")
    if direction.shape != (2,):
        raise ValueError("direction must have shape (2,)")
    slope = -(pos[h] - pos[e]) @ direction
    return np.column_stack((e, h, energy, radius, label, slope))

import numpy as np


def build_couplings(
    positions: "np.ndarray",
    cutoff: float,
    a: float,
    rd: float,
    rt: float,
    d0: float,
    te: float,
    th: float,
) -> "np.ndarray":
    import numpy as np

    pos, dist = _positions(positions)
    n = len(pos)
    cutoff, a, rd, rt = [_positive(v, "length") for v in (cutoff, a, rd, rt)]
    d0, te, th = [_positive(v, "coupling", zero=True) for v in (d0, te, th)]
    active = (dist > 0) & (dist <= cutoff + 1e-12)
    if np.any(1 + (dist[active] - a) / rd <= 0):
        raise ValueError(
            ("dipole denominator must be positive on " "active edges")
        )
    dip = np.zeros_like(dist)
    hop = np.zeros_like(dist)
    dip[active] = d0 / (1 + (dist[active] - a) / rd) ** 3
    hop[active] = np.exp(-(dist[active] - a) / rt)
    matrix = te * np.kron(hop, np.eye(n)) + th * np.kron(np.eye(n), hop)
    le = np.arange(n) * (n + 1)
    matrix[np.ix_(le, le)] = dip
    return matrix

import numpy as np


def thermal_vibronic_weights(
    high: float,
    quantum: float,
    temperature: float,
    nmax: int,
    mmax: int,
) -> "np.ndarray":
    import numpy as np
    from scipy.special import eval_genlaguerre, gammaln

    high = _positive(high, "high", zero=True)
    quantum, temperature = [
        _positive(v, "thermal scale") for v in (quantum, temperature)
    ]
    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, np.integer))
        or v < 0
        for v in (nmax, mmax)
    ):
        raise ValueError("cutoffs must be nonnegative integers")
    n = np.arange(nmax + 1)[:, None]
    m = np.arange(mmax + 1)[None, :]
    small, large = np.minimum(n, m), np.maximum(n, m)
    s = high / quantum
    if s == 0:
        overlap = (n == m).astype(float)
    else:
        overlap = np.exp(
            -s
            + (large - small) * np.log(s)
            + gammaln(small + 1)
            - gammaln(large + 1)
        )
        overlap *= eval_genlaguerre(small, large - small, s) ** 2
    thermal = np.exp(-n * quantum / (8.617333262145e-5 * temperature))
    thermal /= thermal.sum()
    return thermal * overlap

import numpy as np


def _density_response(delta, slope, low, weights, quantum, temperature):
    import numpy as np

    n, m = np.indices(weights.shape)
    kt = 8.617333262145e-5 * temperature
    gap = np.asarray(delta)[..., None, None] + low + (m - n) * quantum
    terms = (
        weights
        * np.exp(-(gap**2) / (4 * low * kt))
        / np.sqrt(4 * np.pi * low * kt)
    )
    log_derivative = -gap / (2 * low * kt)
    slope = np.asarray(slope)
    return np.array(
        [
            terms.sum(axis=(-2, -1)),
            slope * (terms * log_derivative).sum(axis=(-2, -1)),
            slope**2
            * (terms * (log_derivative**2 - 1 / (2 * low * kt))).sum(
                axis=(-2, -1)
            ),
        ]
    )


def build_kinetics_response(
    basis: "np.ndarray",
    couplings: "np.ndarray",
    low_x: float,
    high_x: float,
    low_p: float,
    high_p: float,
    quantum: float,
    temperature: float,
    nmax: int,
    mmax: int,
    vx: float,
    vct: float,
    cutoff: float,
    kext: float,
    illumination: "np.ndarray",
) -> "np.ndarray":
    import numpy as np

    basis = _finite(basis, "basis")
    coupling = _finite(couplings, "couplings")
    if basis.ndim != 2 or basis.shape[1] != 6 or len(basis) < 4:
        raise ValueError("basis must have shape (N*N,6), N>=2")
    size = len(basis)
    n = int(np.sqrt(size))
    e, h = np.indices((n, n)).reshape(2, -1)
    if n * n != size or not np.array_equal(
        basis[:, :2], np.column_stack((e, h))
    ):
        raise ValueError("basis must follow e*N+h order")
    labels = basis[:, 4]
    if not np.all(np.isin(labels, (0, 1, 2))) or not np.array_equal(
        labels == 0, e == h
    ):
        raise ValueError("invalid labels")
    if np.any(basis[:, 3] < 0) or coupling.shape != (size, size):
        raise ValueError("invalid separations or coupling shape")
    if not np.allclose(coupling, coupling.T, rtol=0, atol=1e-12) or np.any(
        np.diag(coupling) != 0
    ):
        raise ValueError("couplings must be symmetric with zero diagonal")
    low_x, low_p = [_positive(x, "low") for x in (low_x, low_p)]
    high_x, high_p = [
        _positive(x, "high", zero=True) for x in (high_x, high_p)
    ]
    quantum, temperature, cutoff = [
        _positive(x, "scale") for x in (quantum, temperature, cutoff)
    ]
    vx, vct, kext = [_positive(x, "sink", zero=True) for x in (vx, vct, kext)]
    light = _finite(illumination, "illumination")
    if light.shape != (n,) or np.any(light < 0):
        raise ValueError("illumination must be a nonnegative (N,) vector")
    wx = thermal_vibronic_weights(
        high_x, quantum, temperature, nmax, mmax
    )
    wxx = thermal_vibronic_weights(
        2 * high_x, quantum, temperature, nmax, mmax
    )
    wpp = thermal_vibronic_weights(
        2 * high_p, quantum, temperature, nmax, mmax
    )
    energy, slope = basis[:, 2], basis[:, 5]
    delta = energy[None, :] - energy[:, None]
    delta_slope = slope[None, :] - slope[:, None]
    le = labels == 0
    lele = le[:, None] & le[None, :]
    density = np.where(
        lele[None],
        _density_response(
            delta, delta_slope, 2 * low_x, wxx, quantum, temperature
        ),
        _density_response(
            delta, delta_slope, 2 * low_p, wpp, quantum, temperature
        ),
    )
    factor = 2 * np.pi / 6.582119569e-7
    rates = factor * coupling[None] ** 2 * density
    rec = factor * np.where(
        le[None],
        vx**2
        * _density_response(-energy, -slope, low_x, wx, quantum, temperature),
        vct**2
        * _density_response(
            -energy, -slope, 2 * low_p, wpp, quantum, temperature
        )
        * (basis[:, 3] <= cutoff + 1e-12)[None],
    )
    result = np.zeros((3, size, size + 3))
    result[:, :, :size] = rates
    result[:, :, size] = rec
    result[0, :, size + 1] = kext * (labels == 2)
    result[0, :, size + 2] = np.where(le, light[e], 0.0)
    return result

import numpy as np


def _response_array(kinetics):
    import numpy as np

    data = _finite(kinetics, "kinetics")
    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError("kinetics must have shape (3,M,M+3)")
    m = data.shape[1]
    if m < 1 or data.shape[2] != m + 3:
        raise ValueError("kinetics must have shape (3,M,M+3)")
    if np.any(data[0] < 0):
        raise ValueError(
            "zeroth-order rates and generation must be nonnegative"
        )
    if any(np.any(np.diag(c[:, :m]) != 0) for c in data):
        raise ValueError("all transfer diagonals must be zero")
    return data


def _solve_response(kinetics: "np.ndarray") -> "np.ndarray":
    import numpy as np

    data = _response_array(kinetics)
    m = data.shape[1]
    reaches = data[0, :, m] + data[0, :, m + 1] > 0
    for _ in range(m):
        new = reaches | np.any((data[0, :, :m] > 0) & reaches[None, :], axis=1)
        if np.array_equal(new, reaches):
            break
        reaches = new
    if not np.all(reaches):
        raise ValueError("every state must reach a positive sink at s=0")
    loss = np.array(
        [
            np.diag(c[:, :m].sum(axis=1) + c[:, m] + c[:, m + 1]) - c[:, :m].T
            for c in data
        ]
    )
    gen = data[:, :, -1]
    population = np.zeros((3, m))
    try:
        population[0] = np.linalg.solve(loss[0], gen[0])
        population[1] = np.linalg.solve(
            loss[0], gen[1] - loss[1] @ population[0]
        )
        population[2] = np.linalg.solve(
            loss[0],
            gen[2] - loss[2] @ population[0] - 2 * loss[1] @ population[1],
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("zeroth-order balance must be nonsingular") from exc
    if not np.all(np.isfinite(population)):
        raise ValueError("nonfinite response")
    return population


def _reduce_response(
    energy: "np.ndarray",
    labels: "np.ndarray",
    kinetics: "np.ndarray",
    temperature: float,
) -> "np.ndarray":
    import numpy as np

    data = _response_array(kinetics)
    m = data.shape[1]
    energy, labels = _finite(energy, "energy"), _finite(labels, "labels")
    temperature = _positive(temperature, "temperature")
    if (
        energy.shape != (m, 2)
        or labels.shape != (m,)
        or set(labels) != {0, 1, 2}
    ):
        raise ValueError(
            "energy must be (M,2) and labels cover all three pools"
        )
    member = np.eye(3)[labels.astype(int)]
    weight = np.zeros((3, m, 3))
    kt = 8.617333262145e-5 * temperature
    for a in range(3):
        mask = labels == a
        w = np.exp(-(energy[mask, 0] - energy[mask, 0].min()) / kt)
        w /= w.sum()
        score = -energy[mask, 1] / kt
        centered = score - w @ score
        weight[0, mask, a] = w
        weight[1, mask, a] = w * centered
        weight[2, mask, a] = w * (centered**2 - w @ (centered**2))
    out = np.zeros((3, 3, 6))
    for order in range(3):
        for k in range(order + 1):
            choose = 2 if order == 2 and k == 1 else 1
            out[order, :, :3] += (
                choose * weight[k].T @ data[order - k, :, :m] @ member
            )
            out[order, :, 3:5] += (
                choose * weight[k].T @ data[order - k, :, m: m + 2]
            )
        np.fill_diagonal(out[order, :, :3], 0.0)
        out[order, :, 5] = member.T @ data[order, :, -1]
    return out


def coupled_population_response(
    energy: 'np.ndarray',
    labels: 'np.ndarray',
    kinetics: 'np.ndarray',
    temperature: float,
) -> "np.ndarray":
    reduced = _reduce_response(energy, labels, kinetics, temperature)
    full = _solve_response(kinetics)
    pools = _solve_response(reduced)
    e = np.asarray(energy, dtype=float)
    lab = np.asarray(labels, dtype=int)
    kt = 8.617333262145e-5 * temperature
    lifted = np.zeros_like(full)
    for a in range(3):
        mask = lab == a
        w = np.exp(-(e[mask, 0] - e[mask, 0].min()) / kt)
        w /= w.sum()
        score = -e[mask, 1] / kt
        centered = score - w @ score
        wp = w * centered
        wpp = w * (centered**2 - w @ (centered**2))
        lifted[0, mask] = w * pools[0, a]
        lifted[1, mask] = wp * pools[0, a] + w * pools[1, a]
        lifted[2, mask] = (
            wpp * pools[0, a] + 2 * wp * pools[1, a]
            + w * pools[2, a]
        )
    if not np.all(np.isfinite(lifted)):
        raise ValueError("nonfinite canonical population response")
    return np.stack((full, lifted))

import numpy as np


def charge_yield_response(
    kinetics: "np.ndarray",
    population: "np.ndarray",
    labels: "np.ndarray",
) -> "np.ndarray":
    import numpy as np

    data = _response_array(kinetics)
    m = data.shape[1]
    population, labels = _finite(population, "population"), _finite(
        labels, "labels"
    )
    if (
        population.shape != (3, m)
        or labels.shape != (m,)
        or np.any(population[0] < 0)
    ):
        raise ValueError(
            "invalid population shape or negative base population"
        )
    if not np.all(np.isin(labels, (0, 1, 2))):
        raise ValueError("labels must be 0/1/2")
    gen = data[:, :, -1].sum(axis=1)
    if gen[0] <= 0:
        raise ValueError("total generation at s=0 must be positive")
    flux = np.zeros((3, 3))
    for order in range(3):
        for k in range(order + 1):
            choose = 2 if order == 2 and k == 1 else 1
            rec = data[order - k, :, m] * population[k]
            flux[order] += choose * np.array(
                [
                    data[order - k, :, m + 1] @ population[k],
                    rec[labels == 0].sum(),
                    rec[labels != 0].sum(),
                ]
            )
    out = np.zeros((3, 3))
    out[0] = flux[0] / gen[0]
    out[1] = (flux[1] - gen[1] * out[0]) / gen[0]
    out[2] = (flux[2] - gen[2] * out[0] - 2 * gen[1] * out[1]) / gen[0]
    return out

import numpy as np


def canonical_bias_curvature(
    positions: "np.ndarray",
    homo: "np.ndarray",
    lumo: "np.ndarray",
    binding: "np.ndarray",
    disorder: "np.ndarray",
    field: "np.ndarray",
    illumination: "np.ndarray",
    parameters: "np.ndarray",
    direction: "np.ndarray",
) -> float:
    import numpy as np

    p = _finite(parameters, "parameters")
    if p.shape != (21,):
        raise ValueError("parameters must have shape (21,)")
    if any(x != int(x) or x < 0 for x in p[16:18]):
        raise ValueError("vibrational cutoffs must be nonnegative integers")
    basis = build_molecular_basis(
        positions,
        homo,
        lumo,
        binding,
        p[0],
        p[1],
        field,
        disorder,
        p[2],
        direction,
    )
    coupling = build_couplings(positions, *p[3:10])
    kinetics = build_kinetics_response(
        basis,
        coupling,
        *p[10:16],
        int(p[16]),
        int(p[17]),
        p[18],
        p[19],
        p[3],
        p[20],
        illumination
    )
    population = coupled_population_response(
        basis[:, [2, 5]], basis[:, 4], kinetics, p[15]
    )
    full = charge_yield_response(
        kinetics, population[0], basis[:, 4]
    )
    canonical = charge_yield_response(
        kinetics, population[1], basis[:, 4]
    )
    return float(100 * (canonical[2, 0] - full[2, 0]))
SCICODE_GOLD_EOF
