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


def sampling_record(lam_max: np.ndarray, interfaces: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lam_max = np.asarray(lam_max, dtype=float)
    interfaces = np.asarray(interfaces, dtype=float)
    if lam_max.ndim != 1 or lam_max.size < 1:
        raise ValueError("lam_max must be a 1D array with at least one entry")
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if not np.all(np.isfinite(lam_max)):
        raise ValueError("lam_max must be finite")
    n = interfaces.size - 1
    npaths = lam_max.size
    mu = np.zeros((npaths, n), dtype=float)
    w = np.ones((npaths, n), dtype=float)
    for j in range(npaths):
        for k in range(n):
            if interfaces[k] < lam_max[j]:
                mu[j, k] = float(((j + 2 * k + 1) % 4) + 1)
                w[j, k] = 1.0 if k == 0 else 1.0 + 0.75 * ((j + 3 * k) % 4)
    return np.stack([mu, w])

import numpy as np


def ensemble_path_totals(mu: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 2 or mu.size == 0:
        raise ValueError("mu must be a non-empty 2D array")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu entries must be finite")
    if np.any(mu < 0.0):
        raise ValueError("mu entries must be non-negative")
    return mu.sum(axis=0)

import numpy as np


def unbiased_sampling_weights(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    if mu.ndim != 2 or mu.size == 0:
        raise ValueError("mu must be a non-empty 2D array")
    if mu.shape != w.shape:
        raise ValueError("mu and w must have the same shape")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(w)):
        raise ValueError("mu and w must be finite")
    if np.any(mu < 0.0):
        raise ValueError("mu entries must be non-negative")
    if np.any(w[mu > 0.0] <= 0.0):
        raise ValueError("w must be strictly positive wherever mu is non-zero")
    eta = mu.sum(axis=0)
    t = np.zeros_like(mu)
    for k in range(mu.shape[1]):
        safe = np.where(w[:, k] > 0.0, w[:, k], 1.0)
        ratio = np.where(mu[:, k] > 0.0, mu[:, k] / safe, 0.0)
        total = ratio.sum()
        if total > 0.0:
            t[:, k] = ratio * eta[k] / total
    return t

import numpy as np


def highest_ensemble_index(lam_max: np.ndarray, interfaces: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lam_max = np.asarray(lam_max, dtype=float)
    interfaces = np.asarray(interfaces, dtype=float)
    if lam_max.ndim != 1 or lam_max.size < 1:
        raise ValueError("lam_max must be a 1D array with at least one entry")
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if not np.all(np.isfinite(lam_max)):
        raise ValueError("lam_max must be finite")
    if np.any(lam_max <= interfaces[0]):
        raise ValueError("every trajectory must pass the first interface")
    n = interfaces.size - 1
    index = np.empty(lam_max.size, dtype=int)
    for j in range(lam_max.size):
        if lam_max[j] > interfaces[n - 1]:
            index[j] = n - 1
        else:
            index[j] = int(np.searchsorted(interfaces[:n], lam_max[j], side="left")) - 1
    return index

import numpy as np


def interface_crossing_totals(t: np.ndarray, lam_max: np.ndarray,
                                      interfaces: np.ndarray, level: int) -> np.ndarray:
    """Reference implementation."""
    t = np.asarray(t, dtype=float)
    lam_max = np.asarray(lam_max, dtype=float)
    interfaces = np.asarray(interfaces, dtype=float)
    if t.ndim != 2 or t.size == 0:
        raise ValueError("t must be a non-empty 2D array")
    if lam_max.ndim != 1 or t.shape[0] != lam_max.size:
        raise ValueError("t must carry one row per trajectory")
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
        raise ValueError("level must be an integer")
    if not 0 <= int(level) < interfaces.size:
        raise ValueError("level must index one of the interfaces")
    passed = (lam_max > interfaces[int(level)]).astype(float)
    return (t * passed[:, None]).sum(axis=0)

import numpy as np


def crossing_probabilities(crossing_totals: np.ndarray,
                                   ensemble_totals: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    crossing_totals = np.asarray(crossing_totals, dtype=float)
    eta = np.asarray(ensemble_totals, dtype=float)
    if eta.ndim != 1 or eta.size < 1:
        raise ValueError("ensemble_totals must be a non-empty 1D array")
    n = eta.size
    if crossing_totals.ndim != 2 or crossing_totals.shape != (n, n):
        raise ValueError("crossing_totals must be square with one row per interface")
    if np.any(eta < 0.0) or np.any(crossing_totals < 0.0):
        raise ValueError("totals must be non-negative")
    if eta[0] <= 0.0:
        raise ValueError("the lowest ensemble must carry non-zero weight")
    probs = np.ones(n + 1, dtype=float)
    norm = np.zeros(n, dtype=float)
    norm[0] = 1.0 / eta[0]
    for i in range(1, n + 1):
        probs[i] = norm[i - 1] * crossing_totals[i - 1, :i].sum()
        if i < n:
            if probs[i] <= 0.0:
                raise ValueError("crossing probability vanished before the last ensemble")
            norm[i] = 1.0 / np.sum(eta[: i + 1] / probs[: i + 1])
    return probs

import numpy as np


def wham_normalisers(eta: np.ndarray, crossing_probs: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    eta = np.asarray(eta, dtype=float)
    probs = np.asarray(crossing_probs, dtype=float)
    if eta.ndim != 1 or eta.size < 1:
        raise ValueError("eta must be a 1D array with at least one entry")
    if probs.ndim != 1 or probs.size != eta.size + 1:
        raise ValueError("crossing_probs must hold exactly one more entry than eta")
    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(probs)):
        raise ValueError("eta and crossing_probs must be finite")
    if np.any(eta < 0.0):
        raise ValueError("eta entries must be non-negative")
    if np.any(probs[: eta.size] <= 0.0):
        raise ValueError("crossing probabilities must be strictly positive")
    n = eta.size
    norm = np.empty(n, dtype=float)
    for i in range(n):
        total = np.sum(eta[: i + 1] / probs[: i + 1])
        if total <= 0.0:
            raise ValueError("cumulative normalising sum must be strictly positive")
        norm[i] = 1.0 / total
    return norm

import numpy as np


def plus_path_weights(t: np.ndarray, ensemble_index: np.ndarray,
                              normalisers: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    t = np.asarray(t, dtype=float)
    index = np.asarray(ensemble_index)
    norm = np.asarray(normalisers, dtype=float)
    if t.ndim != 2 or t.size == 0:
        raise ValueError("t must be a non-empty 2D array")
    if index.ndim != 1 or index.size != t.shape[0]:
        raise ValueError("ensemble_index must carry one entry per trajectory")
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError("ensemble_index must hold integers")
    if norm.ndim != 1 or norm.size != t.shape[1]:
        raise ValueError("normalisers must carry one entry per ensemble")
    if np.any(index < 0) or np.any(index >= norm.size):
        raise ValueError("ensemble_index entries must address the normalisers")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(norm)):
        raise ValueError("t and normalisers must be finite")
    return norm[index] * t.sum(axis=1)

import numpy as np


def minus_path_weights(minus_multiplicities: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    m = np.asarray(minus_multiplicities, dtype=float)
    if m.ndim != 1 or m.size < 1:
        raise ValueError("minus_multiplicities must be a 1D array with at least one entry")
    if not np.all(np.isfinite(m)):
        raise ValueError("minus_multiplicities must be finite")
    if np.any(m < 0.0):
        raise ValueError("minus_multiplicities must be non-negative")
    total = m.sum()
    if total <= 0.0:
        raise ValueError("the reactant-side ensemble must carry non-zero weight")
    return m / total

import numpy as np


def end_interface_fraction(minus_weights: np.ndarray,
                                   ends_at_first_interface: np.ndarray) -> float:
    """Reference implementation."""
    weights = np.asarray(minus_weights, dtype=float)
    flags = np.asarray(ends_at_first_interface)
    if weights.ndim != 1 or weights.size < 1:
        raise ValueError("minus_weights must be a 1D array with at least one entry")
    if flags.ndim != 1 or flags.size != weights.size:
        raise ValueError("ends_at_first_interface must match minus_weights in length")
    if not np.all(np.isfinite(weights)):
        raise ValueError("minus_weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("minus_weights must be non-negative")
    flags = flags.astype(bool)
    return float(np.sum(weights * flags))

import numpy as np


def path_slice_values(lam_max_value: float, path_length: int, interfaces: np.ndarray,
                              lower_turning_point: float = None) -> np.ndarray:
    """Reference implementation."""
    interfaces = np.asarray(interfaces, dtype=float)
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if isinstance(path_length, bool) or not isinstance(path_length, (int, np.integer)):
        raise ValueError("path_length must be an integer")
    if int(path_length) < 3:
        raise ValueError("path_length must be at least 3")
    if not np.isfinite(float(lam_max_value)):
        raise ValueError("lam_max_value must be finite")
    interior = int(path_length) - 1
    start = float(interfaces[0])
    if lower_turning_point is not None:
        if not np.isfinite(float(lower_turning_point)):
            raise ValueError("lower_turning_point must be finite")
        peak = float(lower_turning_point)
    elif float(lam_max_value) > float(interfaces[-1]):
        rise = (np.arange(interior, dtype=float) + 1.0) / (interior + 1.0)
        return start + (float(interfaces[-1]) - start) * rise
    else:
        peak = float(lam_max_value)
    turn = (interior - 1) // 2
    values = np.empty(interior, dtype=float)
    for m in range(interior):
        if m <= turn:
            frac = (m + 1.0) / (turn + 1.0)
        else:
            frac = (interior - m) / float(interior - turn)
        values[m] = start + (peak - start) * frac
    return values

import numpy as np


def conditional_density_histogram(plus_values: np.ndarray, plus_weights: np.ndarray,
                                          minus_values: np.ndarray, minus_weights: np.ndarray,
                                          end_fraction: float, lower_bound: float,
                                          upper_bound: float, n_bins: int) -> np.ndarray:
    """Reference implementation."""
    plus_values = np.asarray(plus_values, dtype=float)
    plus_weights = np.asarray(plus_weights, dtype=float)
    minus_values = np.asarray(minus_values, dtype=float)
    minus_weights = np.asarray(minus_weights, dtype=float)
    if plus_values.ndim != 1 or plus_weights.ndim != 1 or plus_values.size != plus_weights.size:
        raise ValueError("the barrier-side positions and weights must be 1D and equal in length")
    if minus_values.ndim != 1 or minus_weights.ndim != 1 or minus_values.size != minus_weights.size:
        raise ValueError("the reactant-side positions and weights must be 1D and equal in length")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 1:
        raise ValueError("n_bins must be a positive integer")
    if not np.isfinite(float(lower_bound)) or not np.isfinite(float(upper_bound)):
        raise ValueError("the histogram bounds must be finite")
    if float(lower_bound) >= float(upper_bound):
        raise ValueError("lower_bound must lie below upper_bound")
    if not np.isfinite(float(end_fraction)) or float(end_fraction) < 0.0:
        raise ValueError("end_fraction must be finite and non-negative")
    nb = int(n_bins)
    edges = np.linspace(float(lower_bound), float(upper_bound), nb + 1)
    density = np.zeros(nb, dtype=float)
    for values, weights, scale in ((plus_values, plus_weights, float(end_fraction)),
                                   (minus_values, minus_weights, 1.0)):
        for value, weight in zip(values, weights):
            slot = int(np.searchsorted(edges, float(value), side="right")) - 1
            density[min(max(slot, 0), nb - 1)] += scale * float(weight)
    return density

import numpy as np


def conditional_free_energy_barrier(interfaces: np.ndarray, lower_bound: float,
                                            lam_max: np.ndarray, plus_lengths: np.ndarray,
                                            minus_lengths: np.ndarray,
                                            minus_multiplicities: np.ndarray,
                                            ends_at_first_interface: np.ndarray,
                                            n_bins: int) -> float:
    """Reference implementation. Chains steps 1 to 12 through their oracle functions."""
    interfaces = np.asarray(interfaces, dtype=float)
    lam_max = np.asarray(lam_max, dtype=float)
    plus_lengths = np.asarray(plus_lengths)
    minus_lengths = np.asarray(minus_lengths)
    minus_multiplicities = np.asarray(minus_multiplicities, dtype=float)
    flags = np.asarray(ends_at_first_interface)
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if lam_max.ndim != 1 or lam_max.size != plus_lengths.size:
        raise ValueError("lam_max and plus_lengths must have equal length")
    if minus_lengths.size != minus_multiplicities.size or minus_lengths.size != flags.size:
        raise ValueError("the reactant-side arrays must have equal length")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 1:
        raise ValueError("n_bins must be a positive integer")
    if not np.isfinite(float(lower_bound)) or float(lower_bound) >= float(interfaces[0]):
        raise ValueError("lower_bound must be finite and below the first interface")
    if np.any(lam_max <= interfaces[0]):
        raise ValueError("every barrier-side trajectory must pass the first interface")
    if np.any(plus_lengths < 3) or np.any(minus_lengths < 3):
        raise ValueError("every trajectory must hold at least three phase points")

    n = interfaces.size - 1
    record = sampling_record(lam_max, interfaces)
    mu, w = record[0], record[1]
    eta = ensemble_path_totals(mu)
    t = unbiased_sampling_weights(mu, w)
    index = highest_ensemble_index(lam_max, interfaces)
    crossing_totals = np.vstack([
        interface_crossing_totals(t, lam_max, interfaces, level)
        for level in range(1, n + 1)])
    probs = crossing_probabilities(crossing_totals, eta)
    norm = wham_normalisers(eta, probs)
    plus_weights = plus_path_weights(t, index, norm)
    minus_weights = minus_path_weights(minus_multiplicities)
    fraction = end_interface_fraction(minus_weights, flags)

    plus_values, plus_point_weights = [], []
    for j in range(lam_max.size):
        slices = path_slice_values(float(lam_max[j]), int(plus_lengths[j]), interfaces)
        plus_values.append(slices)
        plus_point_weights.append(np.full(slices.size, float(plus_weights[j])))
    minus_values, minus_point_weights = [], []
    for j in range(minus_lengths.size):
        slices = path_slice_values(float(interfaces[0]), int(minus_lengths[j]), interfaces,
                                           lower_turning_point=float(lower_bound))
        minus_values.append(slices)
        minus_point_weights.append(np.full(slices.size, float(minus_weights[j])))

    density = conditional_density_histogram(
        np.concatenate(plus_values), np.concatenate(plus_point_weights),
        np.concatenate(minus_values), np.concatenate(minus_point_weights),
        fraction, float(lower_bound), float(interfaces[-1]), int(n_bins))

    if np.any(density <= 0.0):
        raise ValueError("every histogram bin must receive weight")
    profile = -np.log(density / density.max())
    return float(profile[-1])
SCICODE_GOLD_EOF
