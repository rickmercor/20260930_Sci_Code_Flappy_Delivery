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

def majorana_product_table(left: 'np.ndarray', right: int) -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    a_values = np.asarray(left)
    b = int(right)
    if b < 0 or np.any(a_values < 0):
        raise ValueError('Masks must be nonnegative.')
    qb = b.bit_count() * (b.bit_count() - 1) // 2 & 1
    out = np.empty((a_values.size, 3), dtype=float)
    for row, value in enumerate(a_values):
        a = int(value)
        x = a ^ b
        qa = a.bit_count() * (a.bit_count() - 1) // 2 & 1
        qx = x.bit_count() * (x.bit_count() - 1) // 2 & 1
        crossings = 0
        remaining = a
        while remaining:
            bit = remaining & -remaining
            crossings += (b & bit - 1).bit_count()
            remaining -= bit
        zeta = (-1) ** crossings * 1j ** ((qa + qb - qx) % 4)
        out[row] = (x, zeta.real, zeta.imag)
    return out

import numpy as np

def compile_hubbard_layer(n_sites: int, edges: 'np.ndarray', hopping: 'np.ndarray', interaction: float, dt: float) -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    edges = np.asarray(edges)
    hopping = np.asarray(hopping, dtype=float)
    if len(edges) != len(hopping) or np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError('Edge order or hopping length is invalid.')
    half = []
    for (x, y), t in zip(edges, hopping):
        for spin in range(2):
            i = 2 * int(x) + spin
            j = 2 * int(y) + spin
            half.extend([[1 << 2 * i | 1 << 2 * j + 1, -t * dt / 2], [1 << 2 * i + 1 | 1 << 2 * j, t * dt / 2]])
    center = []
    for site in range(n_sites):
        a = 3 << 4 * site
        b = 3 << 4 * site + 2
        center.extend([[a, interaction * dt / 2], [b, interaction * dt / 2], [a | b, -interaction * dt / 2]])
    return np.array(half + center + half[::-1], float).reshape(-1, 2)

from math import cos, fsum, sin
import numpy as np

def rotate_and_project_expansion(expansion: 'np.ndarray', generator: int, angle: float, n_modes: int, epsilon: float, cap: int) -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    expansion = np.asarray(expansion, dtype=float)
    if epsilon < 0 or cap < 0:
        raise ValueError('Threshold and cap must be nonnegative.')
    out = {}
    ct, st = (cos(angle), sin(angle))
    phases = majorana_product_table(expansion[:, 0], generator)
    for (mask, coefficient), (target, real_phase, imag_phase) in zip(expansion, phases):
        mask, target = (int(mask), int(target))
        if imag_phase == 0:
            out.setdefault(mask, []).append(float(coefficient))
        else:
            out.setdefault(mask, []).append(float(coefficient) * ct)
            out.setdefault(target, []).append(float(coefficient) * float(imag_phase) * st)
    rows = []
    for mask, contributions in sorted(out.items()):
        coefficient = fsum(contributions)
        unpaired = sum((mask >> 2 * j & 3 in (1, 2) for j in range(n_modes)))
        if coefficient != 0 and abs(coefficient) >= epsilon and (unpaired <= cap):
            rows.append([mask, coefficient])
    return np.array(rows, dtype=float).reshape(-1, 2)

import numpy as np

def propagate_trotter_layer(expansion: 'np.ndarray', gates: 'np.ndarray', n_modes: int, epsilon: float, cap: int) -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    if epsilon < 0 or cap < 0:
        raise ValueError('Threshold and cap must be nonnegative.')
    result = np.array(expansion, dtype=float, copy=True)
    temporary_cap = min(n_modes, cap + 2)
    for generator, angle in np.asarray(gates)[::-1]:
        result = rotate_and_project_expansion(result, int(generator), float(angle), n_modes, epsilon, temporary_cap)
    rows = [row for row in result if sum((int(row[0]) >> 2 * j & 3 in (1, 2) for j in range(n_modes))) <= cap]
    return np.array(rows, dtype=float).reshape(-1, 2)

from itertools import combinations
from math import fsum
import numpy as np

def pull_back_gaussian_state(expansion: 'np.ndarray', one_body: 'np.ndarray') -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    V = np.asarray(one_body, complex)
    n = len(V)
    d = 2 * n
    if not np.allclose(V.conj().T @ V, np.eye(n), atol=1e-10, rtol=0):
        raise ValueError('The preparation matrix must be unitary.')
    R = np.zeros((d, d))
    R[0::2, 0::2] = V.real
    R[0::2, 1::2] = -V.imag
    R[1::2, 0::2] = V.imag
    R[1::2, 1::2] = V.real
    sectors = {}
    for mask, c in expansion:
        mask = int(mask)
        I = tuple((j for j in range(d) if mask >> j & 1))
        sectors.setdefault(len(I), []).append((I, float(c)))
    result = []
    for w, rows in sorted(sectors.items()):
        if w == 0:
            result.append([0, fsum((c for _, c in rows))])
            continue
        targets = np.array(list(combinations(range(d), w)), dtype=int)
        totals = np.zeros(len(targets))
        for I, c in rows:
            minor = R[np.array(I)[None, :, None], targets[:, None, :]]
            totals += c * np.linalg.det(minor)
        result.extend([[sum((1 << int(j) for j in J)), float(c)] for J, c in zip(targets, totals)])
    return np.array(sorted(result), float).reshape(-1, 2)

from math import fsum
import numpy as np

def contract_fock_expansion(expansion: 'np.ndarray', occupation: 'np.ndarray') -> float:
    """Evaluate the specified deterministic reference operation."""
    occupation = np.asarray(occupation)
    if np.any((occupation != 0) & (occupation != 1)):
        raise ValueError('Occupation entries must be binary.')
    n_modes = len(occupation)
    terms = []
    for mask, coefficient in np.asarray(expansion):
        mask = int(mask)
        if any((mask >> 2 * j & 3 in (1, 2) for j in range(n_modes))):
            continue
        pairs = mask.bit_count() // 2
        occupied_pairs = sum((int(occupation[j]) for j in range(n_modes) if mask >> 2 * j & 1))
        sign = (-1) ** (((pairs + 1) // 2 + occupied_pairs) % 2)
        terms.append(float(coefficient) * sign)
    return float(fsum(terms))

import numpy as np
from scipy.linalg import expm

def compute_prepared_hole_probability(n_sites: int, edges: 'np.ndarray', hopping: 'np.ndarray', interaction: float, dt: float, n_steps: int, cap: int, epsilon: float, prep_h: 'np.ndarray', prep_time: float, occupation: 'np.ndarray', target: int) -> float:
    """Evaluate the specified deterministic reference operation."""
    if n_steps < 0 or not 0 <= target < n_sites:
        raise ValueError('The layer count or target site is invalid.')
    a = 3 << 4 * target
    b = 3 << 4 * target + 2
    observable = np.array([[0, 0.25], [a, -0.25], [b, -0.25], [a | b, -0.25]], dtype=float)
    gates = compile_hubbard_layer(n_sites, edges, hopping, interaction, dt)
    for _ in range(n_steps):
        observable = propagate_trotter_layer(observable, gates, 2 * n_sites, epsilon, cap)
    one_body = expm(-1j * prep_time * np.asarray(prep_h))
    pulled = pull_back_gaussian_state(observable, one_body)
    return contract_fock_expansion(pulled, occupation)
SCICODE_GOLD_EOF
