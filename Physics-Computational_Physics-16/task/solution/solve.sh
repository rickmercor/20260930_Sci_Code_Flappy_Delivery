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

def compute_fission_moments(probabilities: "np.ndarray", beta: float, alpha: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Implement compute_fission_moments which computes the first and raw second moments of a fission population increment."""
    probabilities = np.asarray(probabilities, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    yields = np.arange(probabilities.size, dtype=float)
    prompt_mean = float(probabilities @ yields)
    delayed = alpha * beta * prompt_mean / (1.0 - beta)
    mean_increment = np.concatenate(([prompt_mean - 1.0], delayed))
    raw_second = np.outer(mean_increment, mean_increment)
    raw_second[0, 0] = probabilities @ ((yields - 1.0) ** 2)
    raw_second[np.arange(1, 7), np.arange(1, 7)] += delayed
    return mean_increment, raw_second

import numpy as np

def build_mean_drift(rho: float, tau_core: float, tau_excore: float, generation_time: float, beta: float, alpha: "np.ndarray", decay: "np.ndarray") -> "np.ndarray":
    """Implement build_mean_drift which constructs the linear mean-population operator for two perfectly mixed fuel regions."""
    alpha = np.asarray(alpha, dtype=float)
    decay = np.asarray(decay, dtype=float)
    drift = np.zeros((13, 13))
    drift[0, 0] = (rho - beta) / generation_time
    drift[0, 1:7] = decay
    drift[1:7, 0] = beta * alpha / generation_time
    core = np.arange(1, 7)
    excore = np.arange(7, 13)
    drift[core, core] = -decay - 1.0 / tau_core
    drift[excore, excore] = -decay - 1.0 / tau_excore
    drift[core, excore] = 1.0 / tau_excore
    drift[excore, core] = 1.0 / tau_core
    return drift

import numpy as np

def build_jump_noise(mean: "np.ndarray", phi: float, gamma: float, source: float, tau_core: float, tau_excore: float, decay: "np.ndarray", fission_second: "np.ndarray") -> "np.ndarray":
    """Implement build_jump_noise which computes the instantaneous covariance-production matrix of the discrete-event model."""
    mean = np.asarray(mean, dtype=float)
    decay = np.asarray(decay, dtype=float)
    noise = np.zeros((13, 13))
    noise[:7, :7] = phi * mean[0] * np.asarray(fission_second, dtype=float)
    noise[0, 0] += gamma * mean[0] + source
    for i in range(6):
        c, e = i + 1, i + 7
        core_decay = decay[i] * mean[c]
        noise[0, 0] += core_decay
        noise[c, c] += core_decay
        noise[0, c] -= core_decay
        noise[c, 0] -= core_decay
        noise[e, e] += decay[i] * mean[e]
        transfer = mean[c] / tau_core + mean[e] / tau_excore
        noise[c, c] += transfer
        noise[e, e] += transfer
        noise[c, e] -= transfer
        noise[e, c] -= transfer
    return noise

import numpy as np

def build_sde_noise(mean: "np.ndarray", phi: float, gamma: float, prompt_increment_second: float) -> "np.ndarray":
    """Implement build_sde_noise which computes covariance production for the source paper’s continuous stochastic differential equation model."""
    noise = np.zeros((13, 13))
    noise[0, 0] = (phi * prompt_increment_second + gamma) * mean[0]
    return noise

import numpy as np

def evaluate_moment_rhs(t: float, state: "np.ndarray", fission_mean: "np.ndarray", fission_second: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray") -> "np.ndarray":
    """Implement evaluate_moment_rhs which evaluates the coupled mean and covariance derivatives for both reactor models during a ramp transient."""
    state = np.asarray(state, dtype=float)
    mean = state[:13]
    covariance_jump = state[13:182].reshape(13, 13)
    covariance_sde = state[182:].reshape(13, 13)
    def _ramp(schedule):
        start, end, duration = (float(x) for x in schedule)
        return start + (end - start) * min(t / duration, 1.0)
    rho = _ramp(rho_schedule)
    tau_core = _ramp(tau_core_schedule)
    tau_excore = _ramp(tau_excore_schedule)
    phi = (1.0 - beta) / ((fission_mean[0] + 1.0) * generation_time)
    gamma = (1.0 - rho) / generation_time - phi
    drift = build_mean_drift(rho, tau_core, tau_excore, generation_time, beta, alpha, decay)
    jump_noise = build_jump_noise(mean, phi, gamma, source, tau_core, tau_excore, decay, fission_second)
    sde_noise = build_sde_noise(mean, phi, gamma, fission_second[0, 0])
    dmean = drift @ mean
    dmean[0] += source
    djump = drift @ covariance_jump + covariance_jump @ drift.T + jump_noise
    dsde = drift @ covariance_sde + covariance_sde @ drift.T + sde_noise
    return np.concatenate((dmean, djump.ravel(), dsde.ravel()))

import numpy as np
from scipy.integrate import solve_ivp

def propagate_moments(t_end: float, probabilities: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Implement propagate_moments which evolves the ensemble mean and both population covariance matrices from an empty reactor."""
    if t_end < 0:
        raise ValueError("t_end must be nonnegative")
    fission_mean, fission_second = compute_fission_moments(probabilities, beta, alpha)
    state = np.zeros(351)
    def _rhs(t, y):
        return evaluate_moment_rhs(t, y, fission_mean, fission_second, generation_time, beta, source, alpha, decay, rho_schedule, tau_core_schedule, tau_excore_schedule)
    kinks = [float(s[2]) for s in (rho_schedule, tau_core_schedule, tau_excore_schedule)]
    start = 0.0
    for stop in sorted(set([x for x in kinks if x < t_end] + [t_end])):
        if stop > start:
            result = solve_ivp(_rhs, (start, stop), state, method="DOP853", rtol=2e-11, atol=1e-11)
            state = result.y[:, -1]
            start = stop
    return state[13:182].reshape(13, 13).copy(), state[182:].reshape(13, 13).copy()

import numpy as np

def calculate_variance_shortfall(t_end: float, probabilities: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray", group: int = 1) -> float:
    """Implement calculate_variance_shortfall which compares the ex-core precursor variances predicted by the two benchmark models."""
    jump, sde = propagate_moments(t_end, probabilities, generation_time, beta, source, alpha, decay, rho_schedule, tau_core_schedule, tau_excore_schedule)
    index = 6 + group
    return float(100.0 * (jump[index, index] - sde[index, index]) / jump[index, index])
SCICODE_GOLD_EOF
