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
 
 
def _occupancy_states(m, d):
    """Occupancy vectors of m lineages in d demes, ascending lexicographic order."""
    return [s for s in itertools.product(range(m + 1), repeat=d) if sum(s) == m]
 
 
def _occupancy_coalescence_rates(states, sizes):
    """First-coalescence rate lambda(x) = sum_i C(x_i, 2) / (2 N_i) of each state."""
    eta = 1.0 / (2.0 * np.asarray(sizes, dtype=float))
    return np.array([sum(0.5 * s[i] * (s[i] - 1) * eta[i] for i in range(len(s))) for s in states], dtype=float)
 
 
def _check_count(value, name, minimum):
    """Validate an integer count that must be at least ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)
 
 
def _check_sizes_migration(sizes, migration):
    """Validate deme sizes and a backward-time migration matrix."""
    n = np.asarray(sizes, dtype=float)
    mig = np.asarray(migration, dtype=float)
    if n.ndim != 1 or n.size < 1 or not np.all(np.isfinite(n)) or np.any(n <= 0.0):
        raise ValueError("sizes must be a finite, strictly positive 1-D array")
    d = n.size
    if mig.shape != (d, d) or not np.all(np.isfinite(mig)):
        raise ValueError("migration must be a finite (d, d) array")
    off = mig[~np.eye(d, dtype=bool)]
    if np.any(off < 0.0):
        raise ValueError("off-diagonal migration rates must be nonnegative")
    return n, mig
 
 
def occupancy_rate_matrix(m: int, sizes: "np.ndarray", migration: "np.ndarray") -> "np.ndarray":
    m = _check_count(m, "m", 0)
    n, mig = _check_sizes_migration(sizes, migration)
    d = n.size
    states = _occupancy_states(m, d)
    index = {s: a for a, s in enumerate(states)}
    rate = -np.diag(_occupancy_coalescence_rates(states, n))
    for s in states:
        a = index[s]
        for i in range(d):
            if s[i] == 0:
                continue
            for j in range(d):
                if j == i or mig[i, j] == 0.0:
                    continue
                target = list(s)
                target[i] -= 1
                target[j] += 1
                r = s[i] * mig[i, j]
                rate[a, index[tuple(target)]] += r
                rate[a, a] -= r
    return rate

import math
 
import numpy as np
 
 
def pulse_occupancy_kernel(m: int, d: int, source: int, dest: int, alpha: float) -> "np.ndarray":
    m = _check_count(m, "m", 0)
    d = _check_count(d, "d", 2)
    source = _check_count(source, "source", 0)
    dest = _check_count(dest, "dest", 0)
    if source >= d or dest >= d or source == dest:
        raise ValueError("source and dest must be distinct indices in [0, d)")
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")
    a = float(alpha)
    states = _occupancy_states(m, d)
    index = {s: i for i, s in enumerate(states)}
    kernel = np.zeros((len(states), len(states)), dtype=float)
    for s in states:
        n = s[dest]
        for w in range(n + 1):
            target = list(s)
            target[dest] -= w
            target[source] += w
            kernel[index[s], index[tuple(target)]] += math.comb(n, w) * a**w * (1.0 - a) ** (n - w)
    return kernel

import numpy as np
from scipy.linalg import expm
 
 
def _check_times(times):
    """Validate a nonempty 1-D array of finite nonnegative times."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 1 or not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("times must be a nonempty 1-D array of finite nonnegative values")
    return t
 
 
def _check_counts(counts, d, name):
    """Validate a length-d vector of nonnegative integer lineage counts."""
    c = np.asarray(counts, dtype=float)
    if c.shape != (d,) or not np.all(np.isfinite(c)) or np.any(c < 0.0) or np.any(c != np.round(c)):
        raise ValueError(f"{name} must hold {d} nonnegative integers")
    return tuple(int(v) for v in c)
 
 
def _check_demography(demography):
    """Validate the pulse-then-split demography and return its parsed fields."""
    keys = ("sizes", "migration", "t_pulse", "t_split", "pulse_source",
            "pulse_dest", "alpha", "ancestral_size")
    if not isinstance(demography, dict) or any(k not in demography for k in keys):
        raise ValueError("demography must be a dict with the documented keys")
    sizes = np.asarray(demography["sizes"], dtype=float)
    migration = np.asarray(demography["migration"], dtype=float)
    if sizes.ndim != 2 or sizes.shape[0] != 2 or sizes.shape[1] < 2:
        raise ValueError("sizes must have shape (2, d) with d >= 2")
    d = sizes.shape[1]
    if migration.shape != (2, d, d):
        raise ValueError("migration must have shape (2, d, d)")
    for e in range(2):
        _check_sizes_migration(sizes[e], migration[e])
    t_pulse = float(demography["t_pulse"])
    t_split = float(demography["t_split"])
    if not (np.isfinite(t_pulse) and np.isfinite(t_split) and 0.0 < t_pulse < t_split):
        raise ValueError("event times must satisfy 0 < t_pulse < t_split")
    source = _check_count(demography["pulse_source"], "pulse_source", 0)
    dest = _check_count(demography["pulse_dest"], "pulse_dest", 0)
    if source >= d or dest >= d or source == dest:
        raise ValueError("pulse_source and pulse_dest must be distinct indices in [0, d)")
    alpha = demography["alpha"]
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")
    ancestral = float(demography["ancestral_size"])
    if not np.isfinite(ancestral) or ancestral <= 0.0:
        raise ValueError("ancestral_size must be finite and positive")
    return {"sizes": sizes, "migration": migration, "t_pulse": t_pulse, "t_split": t_split,
            "source": source, "dest": dest, "alpha": float(alpha), "ancestral": ancestral, "d": d}
 
 
def _propagate(p, q, tau):
    """Return (p exp(q tau) normalized to unit mass, log of its total mass).
 
    The interval is split into chunks over which the mass can fall by at most
    a factor exp(-30), so the survival never underflows even for long times.
    """
    v = np.array(p, dtype=float)
    if tau <= 0.0:
        return v, 0.0
    rate = float(np.max(np.abs(np.diag(q))))
    chunk = tau if rate == 0.0 else min(tau, 30.0 / rate)
    n_full = int(tau // chunk)
    remainder = tau - n_full * chunk
    log_mass = 0.0
    if n_full > 0:
        step = expm(q * chunk)
        for _ in range(n_full):
            v = v @ step
            mass = v.sum()
            log_mass += np.log(mass)
            v = v / mass
    if remainder > 0.0:
        v = v @ expm(q * remainder)
        mass = v.sum()
        log_mass += np.log(mass)
        v = v / mass
    return v, log_mass
 
 
def first_coalescence_curve(sample_counts: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    demo = _check_demography(demography)
    d = demo["d"]
    counts = _check_counts(sample_counts, d, "sample_counts")
    k = sum(counts)
    if k < 2:
        raise ValueError("at least two lineages are required")
    t = _check_times(times)
    t_pulse, t_split = demo["t_pulse"], demo["t_split"]
 
    states = _occupancy_states(k, d)
    q0 = occupancy_rate_matrix(k, demo["sizes"][0], demo["migration"][0])
    q1 = occupancy_rate_matrix(k, demo["sizes"][1], demo["migration"][1])
    lam0 = _occupancy_coalescence_rates(states, demo["sizes"][0])
    lam1 = _occupancy_coalescence_rates(states, demo["sizes"][1])
    p0 = np.zeros(len(states))
    p0[states.index(counts)] = 1.0
    v, log_s_pulse = _propagate(p0, q0, t_pulse)
    kernel = pulse_occupancy_kernel(k, d, demo["source"], demo["dest"], demo["alpha"])
    p_pulse = v @ kernel
    _, log_gain = _propagate(p_pulse, q1, t_split - t_pulse)
    log_s_split = log_s_pulse + log_gain
    rate_anc = 0.5 * k * (k - 1) / (2.0 * demo["ancestral"])
 
    curve = np.zeros((2, t.size), dtype=float)
    for n, time in enumerate(t):
        if time < t_pulse:
            v, log_s = _propagate(p0, q0, time)
            curve[:, n] = (v @ lam0, log_s)
        elif time < t_split:
            v, log_s = _propagate(p_pulse, q1, time - t_pulse)
            curve[:, n] = (v @ lam1, log_s_pulse + log_s)
        else:
            curve[:, n] = (rate_anc, log_s_split - rate_anc * (time - t_split))
    return curve

import itertools
 
import numpy as np
 
 
def _colour_states(n_red, n_blue, d):
    """Interleaved red/blue count vectors in ascending lexicographic order."""
    reds = [s for s in itertools.product(range(n_red + 1), repeat=d) if sum(s) <= n_red]
    blues = [s for s in itertools.product(range(n_blue + 1), repeat=d) if sum(s) <= n_blue]
    return sorted(tuple(v for pair in zip(r, b) for v in pair) for r in reds for b in blues)
 
 
def _cross_coalescence_rates(states, sizes):
    """Cross-coalescence rate sum_i rho_i beta_i / (2 N_i) of each colour state."""
    eta = 1.0 / (2.0 * np.asarray(sizes, dtype=float))
    return np.array([sum(s[2 * i] * s[2 * i + 1] * eta[i] for i in range(len(eta))) for s in states], dtype=float)
 
 
def cross_coalescence_rate_matrix(n_red: int, n_blue: int, sizes: "np.ndarray", migration: "np.ndarray") -> "np.ndarray":
    n_red = _check_count(n_red, "n_red", 1)
    n_blue = _check_count(n_blue, "n_blue", 1)
    n, mig = _check_sizes_migration(sizes, migration)
    d = n.size
    eta = 1.0 / (2.0 * n)
    states = _colour_states(n_red, n_blue, d)
    index = {s: a for a, s in enumerate(states)}
    rate = -np.diag(_cross_coalescence_rates(states, n))
    for s in states:
        a = index[s]
        for colour in range(2):
            for i in range(d):
                count = s[2 * i + colour]
                if count == 0:
                    continue
                for j in range(d):
                    if j == i or mig[i, j] == 0.0:
                        continue
                    target = list(s)
                    target[2 * i + colour] -= 1
                    target[2 * j + colour] += 1
                    r = count * mig[i, j]
                    rate[a, index[tuple(target)]] += r
                    rate[a, a] -= r
                if count >= 2:
                    target = list(s)
                    target[2 * i + colour] -= 1
                    r = 0.5 * count * (count - 1) * eta[i]
                    rate[a, index[tuple(target)]] += r
                    rate[a, a] -= r
    return rate

import numpy as np
 
 
def _colour_pulse_kernel(n_red, n_blue, d, source, dest, alpha):
    """Pulse transition matrix on colour states; colours move independently."""
    states = _colour_states(n_red, n_blue, d)
    index = {s: a for a, s in enumerate(states)}
    red_kernels = [pulse_occupancy_kernel(r, d, source, dest, alpha) for r in range(n_red + 1)]
    blue_kernels = [pulse_occupancy_kernel(b, d, source, dest, alpha) for b in range(n_blue + 1)]
    occupancy = {m: _occupancy_states(m, d) for m in range(max(n_red, n_blue) + 1)}
    kernel = np.zeros((len(states), len(states)), dtype=float)
    for s in states:
        red, blue = s[0::2], s[1::2]
        r, b = sum(red), sum(blue)
        k_red, k_blue = red_kernels[r], blue_kernels[b]
        i_red, i_blue = occupancy[r].index(red), occupancy[b].index(blue)
        for j_red, red_new in enumerate(occupancy[r]):
            if k_red[i_red, j_red] == 0.0:
                continue
            for j_blue, blue_new in enumerate(occupancy[b]):
                weight = k_red[i_red, j_red] * k_blue[i_blue, j_blue]
                if weight == 0.0:
                    continue
                target = tuple(v for pair in zip(red_new, blue_new) for v in pair)
                kernel[index[s], index[target]] += weight
    return kernel
 
 
def cross_coalescence_curve(red_counts: "np.ndarray", blue_counts: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    demo = _check_demography(demography)
    d = demo["d"]
    red = _check_counts(red_counts, d, "red_counts")
    blue = _check_counts(blue_counts, d, "blue_counts")
    n_red, n_blue = sum(red), sum(blue)
    if n_red < 1 or n_blue < 1:
        raise ValueError("at least one red and one blue lineage are required")
    t = _check_times(times)
    t_pulse, t_split = demo["t_pulse"], demo["t_split"]
 
    states = _colour_states(n_red, n_blue, d)
    q0 = cross_coalescence_rate_matrix(n_red, n_blue, demo["sizes"][0], demo["migration"][0])
    q1 = cross_coalescence_rate_matrix(n_red, n_blue, demo["sizes"][1], demo["migration"][1])
    lam0 = _cross_coalescence_rates(states, demo["sizes"][0])
    lam1 = _cross_coalescence_rates(states, demo["sizes"][1])
    p0 = np.zeros(len(states))
    p0[states.index(tuple(v for pair in zip(red, blue) for v in pair))] = 1.0
    v, log_s_pulse = _propagate(p0, q0, t_pulse)
    kernel = _colour_pulse_kernel(n_red, n_blue, d, demo["source"], demo["dest"], demo["alpha"])
    p_pulse = v @ kernel
    v, log_gain = _propagate(p_pulse, q1, t_split - t_pulse)
    log_s_split = log_s_pulse + log_gain
 
    ancestral_states = _colour_states(n_red, n_blue, 1)
    p_anc = np.zeros(len(ancestral_states))
    for s, w in zip(states, v):
        p_anc[ancestral_states.index((sum(s[0::2]), sum(s[1::2])))] += w
    q_anc = cross_coalescence_rate_matrix(n_red, n_blue, np.array([demo["ancestral"]]), np.zeros((1, 1)))
    lam_anc = _cross_coalescence_rates(ancestral_states, np.array([demo["ancestral"]]))
 
    curve = np.zeros((2, t.size), dtype=float)
    for n, time in enumerate(t):
        if time < t_pulse:
            v, log_s = _propagate(p0, q0, time)
            curve[:, n] = (v @ lam0, log_s)
        elif time < t_split:
            v, log_s = _propagate(p_pulse, q1, time - t_pulse)
            curve[:, n] = (v @ lam1, log_s_pulse + log_s)
        else:
            v, log_s = _propagate(p_anc, q_anc, time - t_split)
            curve[:, n] = (v @ lam_anc, log_s_split + log_s)
    return curve

import numpy as np
 
 
def _event_curve(kind, samples, demography, times):
    """Hazard and log-survival curve of the requested first-event time."""
    s = np.asarray(samples, dtype=float)
    if kind == "icr":
        if s.ndim != 1:
            raise ValueError("icr samples must be one-dimensional")
        return first_coalescence_curve(s, demography, times)
    if kind == "ccr":
        if s.ndim != 2 or s.shape[0] != 2:
            raise ValueError("ccr samples must have shape (2, d)")
        return cross_coalescence_curve(s[0], s[1], demography, times)
    raise ValueError("kind must be 'icr' or 'ccr'")
 
 
def first_event_density_score(kind: str, samples: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    if kind not in ("icr", "ccr"):
        raise ValueError("kind must be 'icr' or 'ccr'")
    demo = _check_demography(demography)
    alpha = demo["alpha"]
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    base = _event_curve(kind, samples, demography, times)
    density = base[0] * np.exp(base[1])
 
    def _log_density(a):
        shifted = dict(demography)
        shifted["alpha"] = a
        c = _event_curve(kind, samples, shifted, times)
        with np.errstate(divide="ignore"):
            return np.log(c[0]) + c[1]
 
    h = min(1e-3, 0.5 * alpha, 0.5 * (1.0 - alpha))
    with np.errstate(invalid="ignore"):
        coarse = (_log_density(alpha + h) - _log_density(alpha - h)) / (2.0 * h)
        fine = (_log_density(alpha + 0.5 * h) - _log_density(alpha - 0.5 * h)) / h
        score = (4.0 * fine - coarse) / 3.0
    score = np.where(base[0] > 0.0, score, 0.0)
    return np.vstack([density, score])

import numpy as np
 
 
def _gauss_legendre_panels(edges, order):
    """Composite Gauss-Legendre nodes and weights on consecutive panels."""
    x, w = np.polynomial.legendre.leggauss(order)
    nodes, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (b - a) * x + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)
 
 
def pulse_fisher_information(kind: str, samples: "np.ndarray", demography: dict) -> float:
    if kind not in ("icr", "ccr"):
        raise ValueError("kind must be 'icr' or 'ccr'")
    demo = _check_demography(demography)
    t_pulse, t_split = demo["t_pulse"], demo["t_split"]
    decay = 1.0 / (2.0 * demo["ancestral"])
    middle = np.linspace(t_pulse, t_split, 17)
    tail = t_split + np.concatenate([[0.0], 0.025 * 2.0 ** np.arange(13)]) / decay
    nodes_mid, weights_mid = _gauss_legendre_panels(middle, 16)
    nodes_tail, weights_tail = _gauss_legendre_panels(tail, 16)
    nodes = np.concatenate([nodes_mid, nodes_tail])
    weights = np.concatenate([weights_mid, weights_tail])
    fs = first_event_density_score(kind, samples, demography, nodes)
    return float(np.sum(weights * fs[0] * fs[1] ** 2))

import numpy as np
 
 
def pulse_information_ratio(sample_counts: "np.ndarray", red_counts: "np.ndarray", blue_counts: "np.ndarray", demography: dict) -> float:
    red = np.asarray(red_counts, dtype=float)
    blue = np.asarray(blue_counts, dtype=float)
    if red.ndim != 1 or blue.shape != red.shape:
        raise ValueError("red_counts and blue_counts must be 1-D arrays of equal shape")
    info_k = pulse_fisher_information("icr", sample_counts, demography)
    info_x = pulse_fisher_information("ccr", np.vstack([red, blue]), demography)
    if not info_k > 0.0:
        raise ValueError("the first-coalescence information must be positive")
    return float(info_x / info_k)
SCICODE_GOLD_EOF
