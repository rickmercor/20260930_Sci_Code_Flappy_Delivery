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
from scipy.special import eval_genlaguerre, gammaln, roots_legendre


def _morse_wavefunctions(n_levels, mass, depth, alpha):
    """Exact normalised Morse eigenfunctions on a Gauss-Legendre panel grid in q; returns (q, weights, psi)."""
    import numpy as np
    from scipy.special import eval_genlaguerre, gammaln, roots_legendre
    lam = np.sqrt(2.0 * mass * depth) / alpha
    s_min = lam - (n_levels - 1) - 0.5
    # inner wall: z = 2 lam exp(-alpha q) far beyond the turning point of the ground state
    z_hi = 2.0 * lam + 40.0 * np.sqrt(2.0 * lam) + 150.0
    # outer tail of the least bound level: z^(2 s_min) e^(-z) negligible below z_lo
    z_lo = 2.0 * s_min * np.exp(-(110.0 + 2.0 * s_min) / (2.0 * s_min))
    q_lo = -np.log(z_hi / (2.0 * lam)) / alpha
    q_hi = -np.log(z_lo / (2.0 * lam)) / alpha
    n_pan = int(np.ceil((q_hi - q_lo) / 0.05))
    x, w = roots_legendre(16)
    edges = np.linspace(q_lo, q_hi, n_pan + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    half = 0.5 * (edges[1:] - edges[:-1])
    q = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    z = 2.0 * lam * np.exp(-alpha * q)
    psi = np.empty((n_levels, q.size))
    for v in range(n_levels):
        s = lam - v - 0.5
        log_norm = 0.5 * (np.log(alpha) + np.log(2.0 * s) + gammaln(v + 1.0) - gammaln(2.0 * lam - v))
        psi[v] = np.exp(log_norm + s * np.log(z) - 0.5 * z) * eval_genlaguerre(v, 2.0 * s, z)
    return q, weights, psi


def morse_dipole_matrix(n_levels: int, mass: float, depth: float, alpha: float,
                                dipole_coeffs: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    n = int(n_levels)
    if n < 1:
        raise ValueError("n_levels must be at least 1")
    if not (mass > 0.0 and depth > 0.0 and alpha > 0.0):
        raise ValueError("mass, depth and alpha must be positive")
    lam = np.sqrt(2.0 * mass * depth) / alpha
    if n - 1 >= lam - 0.5:
        raise ValueError("level n_levels - 1 is not bound")
    coeffs = np.atleast_1d(np.asarray(dipole_coeffs, dtype=float))
    q, weights, psi = _morse_wavefunctions(n, mass, depth, alpha)
    mu = np.zeros_like(q)
    for k in range(coeffs.size - 1, -1, -1):
        mu = mu * q + coeffs[k]
    matrix = np.einsum("vi,i,wi->vw", psi, weights * mu, psi)
    return 0.5 * (matrix + matrix.T)

import numpy as np


def _apply_sequence(matrices, vector, block=32):
    """States v_k = A_(k-1) ... A_0 v for k = 0..K from a stack of K matrices, using batched block products."""
    import numpy as np
    count, n, _ = matrices.shape
    dtype = np.result_type(matrices.dtype, np.asarray(vector).dtype)
    n_blocks = -(-count // block)
    pad = n_blocks * block - count
    mats = matrices.astype(dtype)
    if pad:
        mats = np.concatenate([mats, np.broadcast_to(np.eye(n, dtype=dtype), (pad, n, n))], axis=0)
    mats = mats.reshape(n_blocks, block, n, n)
    prefix = np.empty_like(mats)
    prefix[:, 0] = mats[:, 0]
    for j in range(1, block):
        prefix[:, j] = mats[:, j] @ prefix[:, j - 1]
    starts = np.empty((n_blocks, n), dtype=dtype)
    current = np.asarray(vector, dtype=dtype)
    for b in range(n_blocks):
        starts[b] = current
        current = prefix[b, -1] @ current
    states = np.einsum("bjik,bk->bji", prefix, starts).reshape(n_blocks * block, n)[:count]
    return np.concatenate([np.asarray(vector, dtype=dtype)[None], states], axis=0)


def driven_level_populations(energies: "np.ndarray", dipole: "np.ndarray", field_amplitude: float,
                                     omega: float, t_end: float, dt: float) -> "np.ndarray":
    """Reference implementation: fourth-order Magnus propagator (two Gauss points) on fine substeps."""
    import numpy as np
    e = np.asarray(energies, dtype=float).ravel()
    m = np.asarray(dipole, dtype=float)
    n = e.size
    if n < 2 or m.shape != (n, n):
        raise ValueError("energies must have shape (n,) with n >= 2 and dipole shape (n, n)")
    if not (dt > 0.0 and t_end > 0.0):
        raise ValueError("dt and t_end must be positive")
    n_samples = int(round(t_end / dt))
    if n_samples < 1 or abs(n_samples * dt - t_end) > 1e-9 * max(1.0, t_end):
        raise ValueError("t_end must be an integer multiple of dt")
    # substeps short against the field period and the coupling strength (H0 itself is treated exactly)
    rate = max(omega, abs(field_amplitude) * np.max(np.abs(m).sum(axis=1)))
    n_sub = max(1, int(np.ceil(dt * rate / 0.02)))
    h = dt / n_sub
    t = h * np.arange(n_samples * n_sub)
    g = np.sqrt(3.0) / 6.0
    f1 = -field_amplitude * np.cos(omega * (t + h * (0.5 - g)))
    f2 = -field_amplitude * np.cos(omega * (t + h * (0.5 + g)))
    h0 = np.diag(e)
    comm = m @ h0 - h0 @ m
    # exp(Omega) with Omega = -i h (H0 + fbar M) - (sqrt(3) h^2 / 12) (f2 - f1) [M, H0] = -i K, K Hermitian
    k_mat = (h * (h0[None] + (0.5 * (f1 + f2))[:, None, None] * m[None])
             - 1j * (np.sqrt(3.0) * h * h / 12.0) * (f2 - f1)[:, None, None] * comm[None])
    norm = float(np.max(np.abs(k_mat).sum(axis=-1)))
    squarings = 0
    while norm / 2 ** squarings > 0.1:
        squarings += 1
    a_mat = (-1j / 2 ** squarings) * k_mat
    identity = np.broadcast_to(np.eye(n, dtype=complex), k_mat.shape)
    prop = identity.copy()
    for j in range(18, 0, -1):
        prop = identity + (a_mat @ prop) / j
    for _ in range(squarings):
        prop = prop @ prop
    if n_sub > 1:
        prop = _apply_sequence_products(prop.reshape(n_samples, n_sub, n, n))
    c0 = np.zeros(n, dtype=complex)
    c0[0] = 1.0
    states = _apply_sequence(prop, c0)
    return states.real ** 2 + states.imag ** 2


def _apply_sequence_products(groups):
    """Ordered products A_(s-1) ... A_0 within each group of a stack of shape (K, s, n, n)."""
    total = groups[:, 0]
    for j in range(1, groups.shape[1]):
        total = groups[:, j] @ total
    return total

import numpy as np


def least_flow_transition_matrices(populations: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    rho = np.asarray(populations, dtype=float)
    if rho.ndim != 2 or rho.shape[0] < 2 or rho.shape[1] < 2:
        raise ValueError("populations must have shape (K, n) with K >= 2 and n >= 2")
    if np.any(rho < 0.0):
        raise ValueError("populations must be non-negative")
    totals = rho.sum(axis=1)
    if np.max(np.abs(totals - totals[0])) > 1e-6:
        raise ValueError("all rows must have the same total population")
    before = rho[:-1]
    change = np.diff(rho, axis=0)
    gain = np.where(change > 0.0, change, 0.0)
    loss = np.where(change < 0.0, -change, 0.0)
    total_gain = gain.sum(axis=1)
    falling = loss > 0.0
    fraction = np.where(falling, loss / np.where(falling, before, 1.0), 0.0)
    share = fraction / np.where(total_gain > 0.0, total_gain, 1.0)[:, None]
    matrices = gain[:, :, None] * share[:, None, :]
    idx = np.arange(rho.shape[1])
    matrices[:, idx, idx] = 1.0 - fraction
    return matrices

import numpy as np


def nonparticipation_probability(populations: "np.ndarray", target: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    rho = np.asarray(populations, dtype=float)
    if rho.ndim != 2:
        raise ValueError("populations must be a 2-D array")
    n = rho.shape[1]
    m = int(target)
    if not 0 <= m < n:
        raise ValueError("target out of range")
    if rho[0, m] > 1e-12:
        raise ValueError("the target level must be empty at the first sample")
    matrices = least_flow_transition_matrices(rho)
    keep = np.array([k for k in range(n) if k != m])
    reduced = np.ascontiguousarray(matrices[:, keep][:, :, keep])
    states = _apply_sequence(reduced, rho[0, keep])
    return states.sum(axis=1)

import numpy as np


def excitation_delay(times: "np.ndarray", nonparticipation: "np.ndarray", confidence: float) -> float:
    """Reference implementation."""
    import numpy as np
    t = np.asarray(times, dtype=float).ravel()
    p = np.asarray(nonparticipation, dtype=float).ravel()
    if t.size != p.size or t.size < 2:
        raise ValueError("times and nonparticipation must have the same length of at least 2")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if not 0.0 < confidence < 100.0:
        raise ValueError("confidence must lie strictly between 0 and 100")
    level = 1.0 - confidence / 100.0
    reached = np.nonzero(p <= level)[0]
    if reached.size == 0:
        return float("inf")
    k = int(reached[0])
    if k == 0:
        return float(t[0])
    return float(t[k - 1] + (t[k] - t[k - 1]) * (p[k - 1] - level) / (p[k - 1] - p[k]))

import numpy as np


def overtone_nonparticipation(intensity: float, t_end: int, n_levels: int, target: int, mass: float,
                                      depth: float, alpha: float, dipole_coeffs: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if not intensity > 0.0:
        raise ValueError("intensity must be positive")
    if int(t_end) != t_end or t_end < 1:
        raise ValueError("t_end must be a positive integer")
    n = int(n_levels)
    order = int(target)
    if not 1 <= order <= n - 1:
        raise ValueError("target must lie between 1 and n_levels - 1")
    dipole = morse_dipole_matrix(n, mass, depth, alpha, dipole_coeffs)
    omega0 = alpha * np.sqrt(2.0 * depth / mass)
    v = np.arange(n) + 0.5
    energies = omega0 * v - omega0 ** 2 * v ** 2 / (4.0 * depth)
    field_si = np.sqrt(2.0 * intensity * 1e16 / (299792458.0 * 8.8541878128e-12))
    field = field_si / 5.14220674763e11
    omega = (energies[order] - energies[0]) / order
    populations = driven_level_populations(energies, dipole, field, omega, float(t_end), 1.0)
    return nonparticipation_probability(populations, order)

import numpy as np
from scipy.optimize import brentq


def threshold_intensity(t_end: int, confidence: float, intensity_min: float, intensity_max: float,
                                n_scan: int, n_levels: int, target: int, mass: float, depth: float, alpha: float,
                                dipole_coeffs: "np.ndarray") -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    if not intensity_min < intensity_max:
        raise ValueError("intensity_min must be smaller than intensity_max")
    if int(n_scan) < 2:
        raise ValueError("n_scan must be at least 2")
    times = np.arange(int(t_end) + 1, dtype=float)
    level = 1.0 - confidence / 100.0

    history = lambda intensity: overtone_nonparticipation(intensity, t_end, n_levels, target, mass, depth,
                                                                  alpha, dipole_coeffs)
    previous = None
    for intensity in np.linspace(intensity_min, intensity_max, int(n_scan)):
        p_not = history(intensity)
        if excitation_delay(times, p_not, confidence) <= t_end:
            if previous is None:
                return float(intensity)
            return float(brentq(lambda x: history(x)[-1] - level, previous, intensity, xtol=1e-12, rtol=1e-12))
        previous = float(intensity)
    raise ValueError("the confidence level is not reached on the intensity scan")
SCICODE_GOLD_EOF
