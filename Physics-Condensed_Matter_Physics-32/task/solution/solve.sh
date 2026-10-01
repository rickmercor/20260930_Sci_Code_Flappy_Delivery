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
import math

import numpy as np


def extended_ssh_hamiltonian(n_cells: int, t1: float, t2: float, t3: float, periodic: bool) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 2:
        raise ValueError("n_cells must be an integer >= 2")
    hops = np.array([t1, t2, t3], dtype=float)
    if not np.all(np.isfinite(hops)) or np.any(hops < 0.0):
        raise ValueError("hopping amplitudes must be finite and non-negative")
    n = int(n_cells)
    h = np.zeros((2 * n, 2 * n))
    for j in range(n):
        a, b = 2 * j, 2 * j + 1
        h[a, b] -= t1
        h[b, a] -= t1
        if j + 1 < n or periodic:
            a_next, b_next = 2 * ((j + 1) % n), 2 * ((j + 1) % n) + 1
            h[b, a_next] -= t2
            h[a_next, b] -= t2
            h[a, b_next] -= t3
            h[b_next, a] -= t3
    return h

import numpy as np


def _clean_spectrum(h):
    h = np.asarray(h)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 1 or np.iscomplexobj(h) or not np.all(np.isfinite(h)):
        raise ValueError("h must be a finite real square matrix")
    if np.max(np.abs(h - h.T)) > 1e-12:
        raise ValueError("h must be symmetric")
    energies, vectors = np.linalg.eigh(h.astype(float))
    cutoff = 1e-12 * max(1.0, float(np.max(np.abs(energies))))
    energies = np.where(np.abs(energies) <= cutoff, 0.0, energies)
    return energies, vectors


def _occupations(energies, beta):
    if np.isnan(beta) or beta < 0.0:
        raise ValueError("beta must be non-negative")
    if np.isinf(beta):
        return np.where(energies < 0.0, 1.0, np.where(energies > 0.0, 0.0, 0.5))
    x = beta * energies
    with np.errstate(over="ignore", invalid="ignore"):
        occ = 0.5 * (1.0 - np.tanh(0.5 * x))
        tail = np.exp(-np.abs(x))
        occ = np.where(x > 20.0, tail / (1.0 + tail), np.where(x < -20.0, 1.0 / (1.0 + tail), occ))
    return np.where(energies == 0.0, 0.5, occ)


def fermi_correlation_matrix(h: "np.ndarray", beta: float) -> "np.ndarray":
    energies, vectors = _clean_spectrum(h)
    occ = _occupations(energies, float(beta))
    return (vectors * occ) @ vectors.T

import numpy as np


def log_partition_function(energies: "np.ndarray", beta: float) -> float:
    energies = np.asarray(energies)
    if energies.ndim != 1 or energies.size == 0 or np.iscomplexobj(energies) or not np.all(np.isfinite(energies)):
        raise ValueError("energies must be a nonempty finite real 1-D array")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    energies = energies.astype(float)
    cutoff = 1e-12 * max(1.0, float(np.max(np.abs(energies))))
    energies = np.where(np.abs(energies) <= cutoff, 0.0, energies)
    x = -float(beta) * energies
    with np.errstate(over="ignore", invalid="ignore"):
        terms = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    terms = np.where(energies == 0.0, np.log(2.0), terms)
    return float(np.sum(terms))

import numpy as np


def bloch_hamiltonian(k: float, t1: float, t2: float, t3: float) -> "np.ndarray":
    if not np.isfinite(k):
        raise ValueError("k must be finite")
    hops = np.array([t1, t2, t3], dtype=float)
    if not np.all(np.isfinite(hops)) or np.any(hops < 0.0):
        raise ValueError("hopping amplitudes must be finite and non-negative")
    off = t1 + t2 * np.exp(-1j * k) + t3 * np.exp(1j * k)
    hk = -np.array([[0.0, off], [np.conj(off), 0.0]])
    return np.stack([hk.real, hk.imag])

import math

import numpy as np


def global_twist_expectation(n_cells: int, t1: float, t2: float, t3: float, beta: float) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 2:
        raise ValueError("n_cells must be an integer >= 2")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    n = int(n_cells)
    ks = 2.0 * np.pi * np.arange(n) / n
    bloch_hamiltonian(0.0, t1, t2, t3)  # validates the hopping amplitudes
    off = t1 + t2 * np.exp(-1j * ks) + t3 * np.exp(1j * ks)
    hk = np.zeros((n, 2, 2), dtype=complex)
    hk[:, 0, 1] = -off
    hk[:, 1, 0] = -np.conj(off)
    spectra, bases = np.linalg.eigh(hk)
    spectra = np.concatenate([spectra, spectra[:1]])
    bases = np.concatenate([bases, bases[:1]])
    overlaps = np.einsum("mji,mjk->mik", bases[1:].conj(), bases[:-1])
    if np.min(np.abs(overlaps[:, 0, 0])) < 0.1:
        raise ValueError("adjacent occupied-band overlap is below 0.1")
    weights = -beta * spectra[1:]
    tops = weights.max(axis=1)
    factors = np.exp(weights - tops[:, None])[:, :, None] * overlaps
    # Each factor diag(exp(-beta E)) W is divided by exp(beta |e|) (the larger weight, since
    # E = -|e|, +|e|); the same exp(beta |e|) factors appear in Z and cancel exactly, so only
    # O(1) logarithms are ever accumulated.
    product = np.eye(2, dtype=complex)
    log_scales = []
    for m in range(n):
        product = factors[m] @ product
        scale = np.abs(product).max()
        product = product / scale
        log_scales.append(float(np.log(scale)))
    det_overlaps = np.linalg.det(overlaps)
    log_det_abs = np.log(np.abs(det_overlaps)) + weights.sum(axis=1) - 2.0 * tops
    det_phase = math.fsum(np.angle(det_overlaps))
    sign = (-1.0) ** (n - 1)
    eig = np.linalg.eigvals(product)
    lead = eig[np.argmax(np.abs(eig))]
    log_l1 = math.fsum(log_scales) + np.log(abs(lead))
    total_top = beta * math.fsum(np.abs(spectra[:n]).max(axis=1))
    log_l2 = math.fsum(log_det_abs) - log_l1
    z1 = np.exp(1j * np.angle(lead))
    z2 = np.exp(1j * (det_phase - np.angle(lead)))
    # Check the stated conditioning domain without forming exp(beta sum E).
    # Dividing numerator and denominator by max(1, |lambda|) bounds each term.
    for true_log, direction in ((log_l1 + total_top, z1),
                                (log_l2 + total_top, z2)):
        numerator = abs(np.exp(-max(true_log, 0.0))
                        + sign * direction * np.exp(min(true_log, 0.0)))
        denominator = 1.0 + np.exp(-abs(true_log))
        if numerator / denominator < 0.1:
            raise ValueError("transfer cancellation ratio is below 0.1")
    with np.errstate(over="ignore", under="ignore"):
        part1 = log_l1 + np.log(sign * z1) + np.log1p(np.exp(-(log_l1 + total_top)) / (sign * z1))
        true_log_l2 = log_l2 + total_top
        part2 = np.log1p(sign * z2 * np.exp(true_log_l2)) if true_log_l2 < 700.0 else true_log_l2 + np.log(sign * z2)
        positive_energies = np.abs(spectra[:n]).max(axis=1)
        cutoff = 1e-12 * max(1.0, float(positive_energies.max()))
        if np.all(positive_energies > cutoff):
            # For the positive band, step 03 computes the small thermal
            # correction directly, without an extensive cancellation.
            softplus = 2.0 * log_partition_function(positive_energies, beta)
        else:
            # Step 05 does not impose step 03's zero-mode replacement.
            softplus = math.fsum(2.0 * np.log1p(np.exp(-beta * positive_energies)))
    real = float(part1.real + part2.real) - softplus
    phase = float(np.angle(np.exp(1j * (part1.imag + part2.imag))))
    if phase <= -np.pi + 1e-9:
        phase = float(np.pi)
    return np.array([real, phase])

import numpy as np


def diagonal_twist_expectation(f: "np.ndarray", theta: "np.ndarray") -> "np.ndarray":
    f = np.asarray(f)
    theta = np.asarray(theta)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or np.iscomplexobj(f) or not np.all(np.isfinite(f)):
        raise ValueError("f must be a finite real square matrix")
    if np.max(np.abs(f - f.T)) > 1e-12:
        raise ValueError("f must be symmetric")
    if theta.shape != (f.shape[0],) or np.iscomplexobj(theta) or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a finite real vector matching f")
    matrix = np.eye(f.shape[0]) - f + f * np.exp(1j * theta)[None, :]
    sign, log_abs = np.linalg.slogdet(matrix)
    if sign == 0:
        raise ValueError("<O> vanishes")
    phase = float(np.angle(sign))
    if phase <= -np.pi + 1e-9:
        phase = float(np.pi)
    return np.array([log_abs, phase])

import numpy as np


def local_twist_phases(n_cells: int, kind: str, j: int, delta_x: float) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 2:
        raise ValueError("n_cells must be an integer >= 2")
    if isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or not 0 <= j < n_cells:
        raise ValueError("j out of range")
    if not np.isfinite(delta_x):
        raise ValueError("delta_x must be finite")
    n = int(n_cells)
    j = int(j)
    dk = 2.0 * np.pi / n
    theta = np.zeros(2 * n)
    a_j, b_j = 2 * j, 2 * j + 1
    a_next, b_next = 2 * ((j + 1) % n), 2 * ((j + 1) % n) + 1
    terms = {
        "intra": [(a_j, j - delta_x), (b_j, j + delta_x)],
        "inter": [(b_j, j + delta_x), (a_next, j + 1 - delta_x)],
        "nnn": [(a_j, j - delta_x), (b_next, j + 1 + delta_x)],
        "intra2": [(a_j, j - delta_x), (b_j, j + delta_x), (a_next, j + 1 - delta_x), (b_next, j + 1 + delta_x)],
    }
    if kind not in terms:
        raise ValueError("unknown kind")
    for index, position in terms[kind]:
        theta[index] += dk * position
    return theta

import numpy as np


def local_indicators(n_cells: int, t1: float, t2: float, t3: float, beta: float, delta_x: float) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 3 or n_cells % 2 == 0:
        raise ValueError("n_cells must be an odd integer >= 3")
    f = fermi_correlation_matrix(extended_ssh_hamiltonian(n_cells, t1, t2, t3, True), beta)
    centre = (int(n_cells) - 1) // 2
    moduli = {}
    for kind in ("intra2", "inter", "nnn"):
        log_abs, _ = diagonal_twist_expectation(f, local_twist_phases(n_cells, kind, centre, delta_x))
        moduli[kind] = np.exp(log_abs)
    return np.array([moduli["intra2"] - max(moduli["inter"], moduli["nnn"]), moduli["inter"] - moduli["nnn"]])

import numpy as np


def bloch_local_indicators(n_cells: int, t1: float, t2: float, t3: float, beta: float, delta_x: float) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 3 or n_cells % 2 == 0:
        raise ValueError("n_cells must be an odd integer >= 3")
    if np.isnan(beta) or beta < 0.0:
        raise ValueError("beta must be non-negative")
    n = int(n_cells)
    centre = (n - 1) // 2
    ks = 2.0 * np.pi * np.arange(n) / n
    hk = np.array([(lambda parts: parts[0] + 1j * parts[1])(bloch_hamiltonian(k, t1, t2, t3)) for k in ks])
    energies, vectors = np.linalg.eigh(hk)
    cutoff = 1e-12 * max(1.0, float(np.max(np.abs(energies))))
    energies = np.where(np.abs(energies) <= cutoff, 0.0, energies)
    occupations = _occupations(energies, float(beta))
    bloch = np.einsum("kai,ki,kbi->kab", vectors, occupations, vectors.conj())
    # <c^dag_{j,a} c_{j+d,b}> = (1/N) sum_k exp(-i k d) [F(k)]_{ab}; the block on the four
    # centre orbitals (A_c, B_c, A_{c+1}, B_{c+1}) is real for this inversion-symmetric chain.
    same = bloch.mean(axis=0)
    forward = np.einsum("k,kab->ab", np.exp(-1j * ks), bloch) / n
    backward = np.einsum("k,kab->ab", np.exp(1j * ks), bloch) / n
    block = np.block([[same, forward], [backward, same]]).real
    block = 0.5 * (block + block.T)
    moduli = {}
    for kind in ("intra2", "inter", "nnn"):
        theta = local_twist_phases(n, kind, centre, delta_x)[2 * centre:2 * centre + 4]
        moduli[kind] = np.exp(diagonal_twist_expectation(block, theta)[0])
    return np.array([moduli["intra2"] - max(moduli["inter"], moduli["nnn"]), moduli["inter"] - moduli["nnn"]])

import numpy as np


def twist_crossing_beta(n_cells: int, t1: float, t2: float, t3: float, target: float,
                                beta_bracket: "np.ndarray", delta_x: float, n_local: int) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    lo, hi = (float(b) for b in beta_bracket)
    if not 0.0 <= lo < hi < np.inf:
        raise ValueError("beta_bracket must be increasing and finite")
    goal = np.log(target)

    def _excess(beta):
        return global_twist_expectation(n_cells, t1, t2, t3, beta)[0] - goal

    if not (_excess(lo) < 0.0 < _excess(hi)):
        raise ValueError("bracket does not enclose the crossing")
    while hi - lo > 1e-12:
        mid = 0.5 * lo + 0.5 * hi
        if mid == lo or mid == hi:
            break
        if _excess(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    beta_star = 0.5 * lo + 0.5 * hi
    phase = global_twist_expectation(n_cells, t1, t2, t3, beta_star)[1]
    local = bloch_local_indicators(n_local, t1, t2, t3, beta_star, delta_x)
    if n_local <= 151:
        dense_local = local_indicators(n_local, t1, t2, t3, beta_star, delta_x)
        if np.max(np.abs(local - dense_local)) > 1e-10:
            raise ValueError("local representations disagree")
        local = dense_local
    delta_t3 = local[0]
    parity = 0.0 if int(n_cells) % 2 == 1 else np.pi

    def _near(value, reference):
        return abs(np.angle(np.exp(1j * (value - reference)))) < 1e-6

    if _near(phase, parity):
        consistent = delta_t3 > 0.0
    elif _near(phase, parity + np.pi):
        consistent = delta_t3 < 0.0
    else:
        consistent = False
    if not consistent:
        raise ValueError("local indicators disagree with the geometric phase")
    return float(beta_star)
SCICODE_GOLD_EOF
