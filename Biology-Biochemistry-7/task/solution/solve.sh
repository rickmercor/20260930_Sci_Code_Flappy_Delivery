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
def pack_gene_parameters(beta_m: float, beta_mstar: float, beta_p: float,
                                  gamma_m: float, gamma_p: float, alpha: float,
                                  m_a: float, resource_v: float, hill_h: float,
                                  p_a: float) -> "np.ndarray":
    values = np.asarray([beta_m, beta_mstar, beta_p, gamma_m, gamma_p,
                         alpha, m_a, resource_v, hill_h, p_a], dtype=float)
    if values.shape != (10,) or not np.all(np.isfinite(values)):
        raise ValueError("all parameters must be finite scalars")
    positive = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9])
    if np.any(values[positive] <= 0.0):
        raise ValueError("rates, scales, and exponents must be positive")
    if values[7] < 0.0:
        raise ValueError("resource_v must be non-negative")
    return values

import numpy as np
def gene_channel_propensities(state: "np.ndarray",
                                       parameters: "np.ndarray") -> "np.ndarray":
    x = np.asarray(state, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if x.shape != (3,) or p.shape != (10,):
        raise ValueError("state and parameters must have lengths 3 and 10")
    if not np.all(np.isfinite(x)) or np.any(x < 0.0) or np.any(x != np.floor(x)):
        raise ValueError("state must contain non-negative counts")
    if not np.all(np.isfinite(p)) or np.any(p[[0, 1, 2, 3, 4, 5, 6, 8, 9]] <= 0.0) or p[7] < 0.0:
        raise ValueError("parameters are inadmissible")
    m, protein = x[:2]
    initiation = p[0] * p[9] ** p[8] / (p[9] ** p[8] + protein ** p[8])
    return np.array([initiation, p[2] * m, p[3] * m, p[4] * protein], dtype=float)

import numpy as np
def completion_hazard_increment(ages: "np.ndarray", dt: float,
                                         state: "np.ndarray",
                                         parameters: "np.ndarray") -> "np.ndarray":
    a = np.asarray(ages, dtype=float)
    x = np.asarray(state, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if a.ndim != 1 or x.shape != (3,) or p.shape != (10,):
        raise ValueError("ages, state, or parameters have the wrong shape")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("ages must be finite and non-negative")
    if not isinstance(dt, (int, float, np.integer, np.floating)) or not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    if not np.all(np.isfinite(x)) or np.any(x < 0.0) or p[1] <= 0 or p[5] <= 0 or p[6] <= 0 or p[7] < 0:
        raise ValueError("state or parameters are inadmissible")
    scale = p[6] ** p[5] * p[1] ** p[5]
    scale /= p[6] ** p[5] + p[7] * (x[0] + x[2]) ** p[5]
    return scale * ((a + float(dt)) ** p[5] - a ** p[5])

import numpy as np
def invert_completion_waits(ages: "np.ndarray", residuals: "np.ndarray",
                                     state: "np.ndarray",
                                     parameters: "np.ndarray") -> "np.ndarray":
    a = np.asarray(ages, dtype=float)
    r = np.asarray(residuals, dtype=float)
    x = np.asarray(state, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if a.ndim != 1 or r.shape != a.shape or x.shape != (3,) or p.shape != (10,):
        raise ValueError("input shapes are inconsistent")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("ages must be finite and non-negative")
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("residual clocks must be finite and positive")
    if np.any(x < 0.0) or not np.all(np.isfinite(x)) or p[1] <= 0 or p[5] <= 0 or p[6] <= 0 or p[7] < 0:
        raise ValueError("state or parameters are inadmissible")
    coefficient = p[6] ** p[5] * p[1] ** p[5]
    coefficient /= p[6] ** p[5] + p[7] * (x[0] + x[2]) ** p[5]
    waits = (a ** p[5] + r / coefficient) ** (1.0 / p[5]) - a
    return np.maximum(waits, 0.0)

import numpy as np
def advance_remaining_clocks(reaction_residuals: "np.ndarray",
                                      ages: "np.ndarray",
                                      completion_residuals: "np.ndarray",
                                      dt: float, state: "np.ndarray",
                                      parameters: "np.ndarray") -> "np.ndarray":
    rr = np.asarray(reaction_residuals, dtype=float)
    a = np.asarray(ages, dtype=float)
    cr = np.asarray(completion_residuals, dtype=float)
    if rr.shape != (4,) or a.ndim != 1 or cr.shape != a.shape:
        raise ValueError("clock arrays have inconsistent shapes")
    if np.any(rr <= 0.0) or np.any(cr <= 0.0) or np.any(a < 0.0):
        raise ValueError("residuals must be positive and ages non-negative")
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    prop = gene_channel_propensities(state, parameters)
    used_completion = completion_hazard_increment(a, dt, state, parameters)
    next_rr = rr - prop * float(dt)
    next_cr = cr - used_completion
    tolerance = 1e-10
    if np.any(next_rr < -tolerance) or np.any(next_cr < -tolerance):
        raise ValueError("dt advances beyond the next scheduled event")
    next_rr = np.maximum(next_rr, 0.0)
    next_cr = np.maximum(next_cr, 0.0)
    return np.concatenate((next_rr, next_cr, a + float(dt)))

import numpy as np
def simulate_exact_gene_path(parameters: "np.ndarray", final_time: float,
                                      seed: int,
                                      max_events: int = 2000000) -> "np.ndarray":
    p = np.asarray(parameters, dtype=float)
    if p.shape != (10,) or not np.all(np.isfinite(p)):
        raise ValueError("parameters must be a finite length-10 vector")
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if isinstance(max_events, bool) or not isinstance(max_events, (int, np.integer)) or max_events < 1:
        raise ValueError("max_events must be a positive integer")
    rng = np.random.default_rng(int(seed))
    state = np.zeros(3, dtype=int)
    reaction_residuals = rng.exponential(size=4)
    ages, completion_residuals = np.empty(0), np.empty(0)
    # Net change for an initiation, a translation, and the two decays.
    zeta = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    time, events = 0.0, 0
    while time < final_time:
        prop = gene_channel_propensities(state, p)
        reaction_waits = np.full(4, np.inf)
        active = prop > 0.0
        reaction_waits[active] = reaction_residuals[active] / prop[active]
        completion_waits = invert_completion_waits(
            ages, completion_residuals, state, p)
        waits = np.concatenate((reaction_waits, completion_waits))
        event = int(np.argmin(waits))
        dt = float(waits[event])
        if not np.isfinite(dt) or time + dt > final_time:
            break
        packed = advance_remaining_clocks(
            reaction_residuals, ages, completion_residuals, dt, state, p)
        count = ages.size
        reaction_residuals = packed[:4]
        completion_residuals = packed[4:4 + count]
        ages = packed[4 + count:]
        if event < 4:
            reaction_residuals[event] = rng.exponential()
            state += zeta[event]
            if event == 0:
                state[2] += 1
                ages = np.append(ages, 0.0)
                completion_residuals = np.append(completion_residuals, rng.exponential())
        else:
            index = event - 4
            state += np.array([1, 0, -1])
            ages = np.delete(ages, index)
            completion_residuals = np.delete(completion_residuals, index)
        time += dt
        events += 1
        if events >= max_events:
            raise ValueError("maximum event count reached")
    return np.array([state[0], state[1], state[2], events], dtype=float)

import numpy as np
def compute_group_completion_means(group_ages: "np.ndarray",
                                            group_sizes: "np.ndarray", dt: float,
                                            state: "np.ndarray",
                                            parameters: "np.ndarray") -> "np.ndarray":
    ages = np.asarray(group_ages, dtype=float)
    sizes = np.asarray(group_sizes, dtype=float)
    if ages.ndim != 1 or sizes.shape != ages.shape:
        raise ValueError("group arrays must be one-dimensional and aligned")
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0) or np.any(sizes != np.floor(sizes)):
        raise ValueError("group sizes must be positive integer counts")
    means = sizes * completion_hazard_increment(ages, dt, state, parameters)
    if np.any(means < 0.0) or not np.all(np.isfinite(means)):
        raise ValueError("completion means must be finite and non-negative")
    return means

import numpy as np
def simulate_tau_gene_path(parameters: "np.ndarray", final_time: float,
                                    tau: float, seed: int) -> "np.ndarray":
    p = np.asarray(parameters, dtype=float)
    if p.shape != (10,) or not np.all(np.isfinite(p)):
        raise ValueError("parameters must be a finite length-10 vector")
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be positive and finite")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    state = np.zeros(3, dtype=int)
    ages = np.empty(0, dtype=float)
    sizes = np.empty(0, dtype=int)
    time = 0.0
    steps = 0
    while time < final_time:
        dt = min(float(tau), float(final_time) - time)
        prop = gene_channel_propensities(state, p)
        reaction_counts = rng.poisson(prop * dt)
        means = compute_group_completion_means(ages, sizes, dt, state, p)
        completion_counts = rng.poisson(means)
        completion_counts = np.minimum(completion_counts, sizes)
        transcription = int(reaction_counts[0])
        translation = int(reaction_counts[1])
        mrna_loss = min(int(state[0]), int(reaction_counts[2]))
        protein_loss = min(int(state[1]), int(reaction_counts[3]))
        completed = int(np.sum(completion_counts))
        state[0] += completed - mrna_loss
        state[1] += translation - protein_loss
        if sizes.size:
            sizes = sizes - completion_counts
            ages = ages + dt
            keep = sizes > 0
            sizes = sizes[keep]
            ages = ages[keep]
        if transcription > 0:
            sizes = np.append(sizes, transcription)
            ages = np.append(ages, 0.0)
        state[2] = int(np.sum(sizes))
        time += dt
        steps += 1
    return np.array([state[0], state[1], state[2], steps], dtype=float)

import numpy as np
def estimate_endpoint_means(parameters: "np.ndarray", final_time: float,
                                     tau: float, n_paths: int,
                                     base_seed: int) -> "np.ndarray":
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
        raise ValueError("n_paths must be an integer of at least two")
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    exact = np.empty(int(n_paths), dtype=float)
    approximate = np.empty(int(n_paths), dtype=float)
    for index in range(int(n_paths)):
        exact[index] = simulate_exact_gene_path(
            parameters, final_time, int(base_seed) + 2 * index)[1]
        approximate[index] = simulate_tau_gene_path(
            parameters, final_time, tau, int(base_seed) + 2 * index + 1)[1]
    exact_mean = float(np.mean(exact))
    approximate_mean = float(np.mean(approximate))
    exact_se = float(np.std(exact, ddof=1) / np.sqrt(n_paths))
    approximate_se = float(np.std(approximate, ddof=1) / np.sqrt(n_paths))
    return np.array([exact_mean, approximate_mean, exact_se, approximate_se])

import numpy as np
def run_nonmarkovian_gene_pipeline(final_time: float = 60.0,
                                            tau: float = 2.5,
                                            n_paths: int = 80,
                                            base_seed: int = 314159) -> float:
    parameters = pack_gene_parameters(
        10.0, 0.175, 1.0, 0.08, 0.05, 2.5, 10.0, 0.5, 1.5, 5.0)
    # Exercise each paper-specific primitive directly before the ensemble run.
    empty = np.empty(0, dtype=float)
    initial_state = np.zeros(3, dtype=int)
    propensities = gene_channel_propensities(initial_state, parameters)
    increments = completion_hazard_increment(
        empty, 0.0, initial_state, parameters)
    waits = invert_completion_waits(
        empty, empty, initial_state, parameters)
    clocks = advance_remaining_clocks(
        np.ones(4), empty, empty, 0.0, initial_state, parameters)
    group_means = compute_group_completion_means(
        empty, empty, 0.0, initial_state, parameters)
    exact_probe = simulate_exact_gene_path(
        parameters, min(float(final_time), 0.1), int(base_seed))
    tau_probe = simulate_tau_gene_path(
        parameters, min(float(final_time), 0.1), min(float(tau), 0.1),
        int(base_seed) + 1)
    certificate = np.concatenate((propensities, increments, waits, clocks,
                                  group_means, exact_probe, tau_probe))
    if not np.all(np.isfinite(certificate)):
        raise ValueError("the direct oracle chain produced a non-finite value")
    estimates = estimate_endpoint_means(
        parameters, final_time, tau, n_paths, base_seed)
    exact_mean = float(estimates[0])
    approximate_mean = float(estimates[1])
    if not np.isfinite(exact_mean) or exact_mean <= 0.0:
        raise ValueError("the exact ensemble mean must be positive and finite")
    return float(100.0 * (approximate_mean - exact_mean) / exact_mean)
SCICODE_GOLD_EOF
