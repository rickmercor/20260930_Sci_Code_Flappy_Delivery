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

def glauber_transition_matrix(energies: np.ndarray) -> np.ndarray:
    """Reference implementation."""

    def _glauber_f(delta_e: np.ndarray) -> np.ndarray:
        """Numerically stable Glauber acceptance function f(dE) = 1 / (1 + exp(dE))."""
        out = np.empty_like(delta_e, dtype=np.float64)
        pos = delta_e >= 0
        out[pos] = np.exp(-delta_e[pos]) / (1.0 + np.exp(-delta_e[pos]))
        out[~pos] = 1.0 / (1.0 + np.exp(delta_e[~pos]))
        return out

    try:
        e = np.asarray(energies, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("energies must be convertible to an array of real numbers.")

    if e.ndim != 1:
        raise ValueError("energies must be one-dimensional.")
    m = e.shape[0]
    if m < 2:
        raise ValueError("energies must contain at least 2 states.")
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must not contain NaN or infinite values.")

    pi = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        row_sum = 0.0
        if i - 1 >= 0:
            p = 0.5 * _glauber_f(np.array([e[i - 1] - e[i]]))[0]
            pi[i, i - 1] = p
            row_sum += p
        if i + 1 < m:
            p = 0.5 * _glauber_f(np.array([e[i + 1] - e[i]]))[0]
            pi[i, i + 1] = p
            row_sum += p
        pi[i, i] = 1.0 - row_sum
    return pi

import numpy as np 

def work_increments(H: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    
    try:
        H_arr = np.asarray(H, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("H must be convertible to an array of real numbers.")

    if H_arr.ndim != 2:
        raise ValueError("H must be two-dimensional.")
    m, T = H_arr.shape
    if m < 2:
        raise ValueError("H must have at least 2 states (rows).")
    if T < 2:
        raise ValueError("H must have at least 2 time steps (columns).")
    if not np.all(np.isfinite(H_arr)):
        raise ValueError("H must not contain NaN or infinite values.")

    return H_arr[:, 1:] - H_arr[:, :-1]

import numpy as np 

def tilted_transition_matrix(
    energies_here: np.ndarray, energies_ahead: np.ndarray, alpha: float
) -> np.ndarray:
    """Reference implementation."""
    
    try:
        e_here = np.asarray(energies_here, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("energies_here must be convertible to an array of real numbers.")
    try:
        e_ahead = np.asarray(energies_ahead, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("energies_ahead must be convertible to an array of real numbers.")

    if e_here.ndim != 1 or e_ahead.ndim != 1:
        raise ValueError("energies_here and energies_ahead must be one-dimensional.")
    if e_here.shape[0] != e_ahead.shape[0]:
        raise ValueError("energies_here and energies_ahead must have the same length.")
    if e_here.shape[0] < 2:
        raise ValueError("energies_here and energies_ahead must contain at least 2 states.")
    if not np.all(np.isfinite(e_here)) or not np.all(np.isfinite(e_ahead)):
        raise ValueError("energies_here and energies_ahead must not contain NaN or infinite values.")

    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha must be convertible to a real scalar.")
    if not np.isfinite(alpha_val):
        raise ValueError("alpha must be a finite real number.")

    pi = glauber_transition_matrix(e_here)
    delta_w = e_ahead - e_here
    weight = np.exp(-alpha_val * delta_w)
    return pi * weight[np.newaxis, :]

import numpy as np

def exponential_work_average(H: np.ndarray, alpha: float) -> float:
    """Reference implementation."""

    
    try:
        H_arr = np.asarray(H, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("H must be convertible to an array of real numbers.")

    if H_arr.ndim != 2:
        raise ValueError("H must be two-dimensional.")
    m, T = H_arr.shape
    if m < 2:
        raise ValueError("H must have at least 2 states (rows).")
    if T < 2:
        raise ValueError("H must have at least 2 time steps (columns).")
    if not np.all(np.isfinite(H_arr)):
        raise ValueError("H must not contain NaN or infinite values.")

    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha must be convertible to a real scalar.")
    if not np.isfinite(alpha_val):
        raise ValueError("alpha must be a finite real number.")

    w0 = np.exp(-H_arr[:, 0])
    peq = w0 / np.sum(w0)
    dW = work_increments(H_arr)              # chains to Step 2
    vec = peq * np.exp(-alpha_val * dW[:, 0])

    n_intermediate = T - 2
    for tau in range(n_intermediate):
        pi_tilde = tilted_transition_matrix(  # chains to Step 3
            H_arr[:, tau + 1], H_arr[:, tau + 2], alpha_val
        )
        vec = vec @ pi_tilde

    return float(np.sum(vec))

import numpy as np 

def linear_interpolation_mse(Ei: float, Ef: float, N: int, alpha: float) -> float:
    """Reference implementation."""

    import numpy as np
    
    try:
        Ei_v = float(Ei)
        Ef_v = float(Ef)
        alpha_v = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("Ei, Ef, and alpha must be convertible to real scalars.")
    if not (np.isfinite(Ei_v) and np.isfinite(Ef_v) and np.isfinite(alpha_v)):
        raise ValueError("Ei, Ef, and alpha must be finite real numbers.")
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("N must be a positive integer.")

    lam_linear = Ei_v + (Ef_v - Ei_v) * np.arange(1, N + 1) / (N + 1)
    H2 = np.concatenate(([Ei_v], lam_linear, [Ef_v]))
    H = np.zeros((2, N + 2))
    H[1, :] = H2
    return exponential_work_average(H, alpha_v)

import numpy as np

def optimal_intermediate_energies(
    Ei: float, Ef: float, N: int, alpha: float, n_restarts: int, seed: int
) -> np.ndarray:
    """Reference implementation."""
    from scipy.optimize import minimize

    try:
        Ei_v = float(Ei)
        Ef_v = float(Ef)
        alpha_v = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("Ei, Ef, and alpha must be convertible to real scalars.")
    if not (np.isfinite(Ei_v) and np.isfinite(Ef_v) and np.isfinite(alpha_v)):
        raise ValueError("Ei, Ef, and alpha must be finite real numbers.")
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("N must be a positive integer.")
    if not isinstance(n_restarts, (int, np.integer)) or isinstance(n_restarts, bool) or n_restarts < 1:
        raise ValueError("n_restarts must be a positive integer.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    def _build_H(lam):
        H2 = np.concatenate(([Ei_v], lam, [Ef_v]))
        H = np.zeros((2, N + 2))
        H[1, :] = H2
        return H

    def _objective(lam):
        return exponential_work_average(_build_H(lam), alpha_v)

    lam_linear = Ei_v + (Ef_v - Ei_v) * np.arange(1, N + 1) / (N + 1)
    rng = np.random.default_rng(seed)
    starts = [lam_linear.copy()]
    span = max(abs(Ei_v), abs(Ef_v), 1.0) * 1.5
    for i in range(n_restarts - 1):
        if i % 2 == 0:
            starts.append(lam_linear + rng.normal(scale=3.0, size=N))
        else:
            starts.append(rng.uniform(-span, span, size=N))

    best_val = np.inf
    best_lam = lam_linear
    for x0 in starts:
        res = minimize(
            _objective, x0, method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12},
        )
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = res.fun
            best_lam = res.x

    return best_lam

def orchestrator(
    Ei: float, Ef: float, N: int, alpha: float, n_restarts: int, seed: int
) -> float:
    """Reference implementation."""

    import numpy as np
    from scipy.optimize import minimize
    
    try:
        Ei_v = float(Ei)
        Ef_v = float(Ef)
        alpha_v = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("Ei, Ef, and alpha must be convertible to real scalars.")
    if not (np.isfinite(Ei_v) and np.isfinite(Ef_v) and np.isfinite(alpha_v)):
        raise ValueError("Ei, Ef, and alpha must be finite real numbers.")
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("N must be a positive integer.")
    if not isinstance(n_restarts, (int, np.integer)) or isinstance(n_restarts, bool) or n_restarts < 1:
        raise ValueError("n_restarts must be a positive integer.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    # Step 6: recover the optimal intermediate energies
    lam_star = optimal_intermediate_energies(
        Ei_v, Ef_v, N, alpha_v, n_restarts, seed
    )

    # Step 4: evaluate the exact objective at the optimum
    H2 = np.concatenate(([Ei_v], lam_star, [Ef_v]))
    H = np.zeros((2, N + 2))
    H[1, :] = H2
    result = exponential_work_average(H, alpha_v)

    # Step 5: cross-check against the linear-interpolation baseline
    baseline = linear_interpolation_mse(Ei_v, Ef_v, N, alpha_v)
    if result > baseline + 1e-8:
        result = min(result, baseline)

    return float(result)
SCICODE_GOLD_EOF
