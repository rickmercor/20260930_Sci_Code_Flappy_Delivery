#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import itertools
import numpy as np


def commutator_words(order: int) -> np.ndarray:
    rows = []
    for mask in range(1 << order):
        left, right, sign = [], [], 1
        for k in range(order):
            if (mask >> k) & 1:
                right.append(k)
                sign = -sign
            else:
                left.insert(0, k)
        rows.append([sign] + right + [order] + left)
    return np.asarray(rows, dtype=int)

import itertools
import numpy as np


def electronic_paths(mu0: np.ndarray, mu1: np.ndarray, vertices: int) -> np.ndarray:
    mu0 = np.asarray(mu0)
    mu1 = np.asarray(mu1)
    support = (mu0 != 0) | np.any(mu1 != 0, axis=0)
    rows = []
    for inner in itertools.product(range(mu0.shape[0]), repeat=vertices-1):
        path = (0,) + inner + (0,)
        if all(support[path[k], path[k+1]] for k in range(vertices)):
            rows.append(path)
    return np.asarray(rows, dtype=int).reshape(-1, vertices+1)

import numpy as np


def contour_gaussian(omega: np.ndarray, z: np.ndarray, dt: np.ndarray, h: np.ndarray, beta: float = np.inf) -> tuple:
    omega, z, dt, h = map(np.asarray, (omega, z, dt, h))
    count, modes = h.shape
    log_matrix = np.zeros((count+1, count+1), dtype=complex)
    rotation = np.ones(modes, dtype=complex)
    total_a = np.zeros((modes, count+1), dtype=complex)
    total_b = np.zeros_like(total_a)
    phase = 0j
    for ell in range(count+1):
        angle = omega*dt[ell]
        r = np.exp(-1j*angle)
        d = z[ell]*(r-1)
        a = np.zeros_like(total_a)
        b = np.zeros_like(total_b)
        a[:,0] = rotation*d
        b[:,0] = -np.conj(rotation*d)
        phase -= 1j*np.sum(z[ell]**2*np.sin(angle))
        log_matrix += 0.5*(total_b.T@a-total_a.T@b)
        total_a += a
        total_b += b
        rotation *= r
        if ell < count:
            a = np.zeros_like(total_a)
            b = np.zeros_like(total_b)
            a[:,ell+1] = rotation*h[ell]
            b[:,ell+1] = np.conj(rotation)*h[ell]
            log_matrix += 0.5*(total_b.T@a-total_a.T@b)
            total_a += a
            total_b += b
    thermal_width = 1/np.tanh(0.5*beta*omega)
    log_matrix += 0.5*total_a.T@(thermal_width[:,None]*total_b)
    g = np.exp(phase+log_matrix[0,0])
    linear = log_matrix[0,1:]+log_matrix[1:,0]
    quadratic = log_matrix[1:,1:]+log_matrix[1:,1:].T
    return complex(g), linear, quadratic

import itertools
import numpy as np


def ht_coefficients(mu: np.ndarray, L: np.ndarray, K: np.ndarray) -> np.ndarray:
    mu, L, K = map(np.asarray, (mu, L, K))
    n = len(mu)
    moments = np.zeros(1 << n, dtype=complex)
    moments[0] = 1
    for mask in range(1, 1 << n):
        low = mask & -mask
        i = low.bit_length()-1
        rest = mask ^ low
        value = L[i]*moments[rest]
        bits = rest
        while bits:
            bit = bits & -bits
            j = bit.bit_length()-1
            value += K[i, j]*moments[rest ^ bit]
            bits ^= bit
        moments[mask] = value
    result = np.zeros(n+1, dtype=complex)
    for mask in range(1 << n):
        weight = 1+0j
        for i in range(n):
            if not (mask >> i) & 1:
                weight *= mu[i]
        result[mask.bit_count()] += weight*moments[mask]
    return result

import itertools
import numpy as np


def pathway_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, times: np.ndarray, path: np.ndarray, beta: float = np.inf) -> np.ndarray:
    energies, omega, displacements, mu0, mu1, times, path = map(
        np.asarray, (energies, omega, displacements, mu0, mu1, times, path))
    dt = np.concatenate(([-times[0]], times[:-1]-times[1:], [times[-1]]))
    z = displacements[path]
    mu = mu0[path[:-1], path[1:]]
    h = mu1[:, path[:-1], path[1:]].T
    g, linear, quadratic = contour_gaussian(omega, z, dt, h, beta)
    coefficients = ht_coefficients(mu, linear, quadratic)
    phase = np.exp(-1j*np.dot(energies[path], dt))
    return phase*g*coefficients

import itertools
import numpy as np


def word_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, times: np.ndarray, beta: float = np.inf) -> np.ndarray:
    times = np.asarray(times)
    paths = electronic_paths(mu0, mu1, len(times))
    result = np.zeros(len(times)+1, dtype=complex)
    for path in paths:
        result += pathway_coefficients(
            energies, omega, displacements, mu0, mu1, times, path, beta)
    return result

import itertools
import numpy as np


def response_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, waits: np.ndarray, beta: float = np.inf) -> np.ndarray:
    waits = np.asarray(waits)
    order = len(waits)
    times = np.concatenate(([0.], np.cumsum(waits)))
    result = np.zeros(order+2, dtype=complex)
    for row in commutator_words(order):
        result += row[0]*word_coefficients(
            energies, omega, displacements, mu0, mu1, times[row[1:]], beta)
    return np.real((1j**order)*result)

import itertools
import numpy as np


def solve(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, dmu_dX: np.ndarray, waits: np.ndarray, beta: float = np.inf) -> float:
    mu1 = 0.5*np.asarray(dmu_dX)
    coefficients = response_coefficients(
        energies, omega, displacements, mu0, mu1, waits, beta)
    return float(np.sum(coefficients[3:]))
SCICODE_GOLD_EOF
