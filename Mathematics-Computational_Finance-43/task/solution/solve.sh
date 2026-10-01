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
def compute_double_heston_aes_coefficients(
    dt: float, rate: float, kappa: "np.ndarray", theta: "np.ndarray",
    gamma: "np.ndarray", rho: "np.ndarray",
) -> "np.ndarray":
    import numpy as np
    arrays = [np.asarray(x, dtype=float) for x in (kappa, theta, gamma, rho)]
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(rate):
        raise ValueError("dt must be positive and rate must be finite")
    if any(x.shape != (2,) or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("factor parameters must be finite length-two arrays")
    kappa, theta, gamma, rho = arrays
    if np.any(kappa <= 0.0) or np.any(theta <= 0.0) or np.any(gamma <= 0.0):
        raise ValueError("CIR scale parameters must be positive")
    if np.any(np.abs(rho) > 1.0):
        raise ValueError("correlations must lie in [-1, 1]")
    current = (rho * kappa / gamma - 0.5) * dt - rho / gamma
    endpoint = rho / gamma
    residual = (1.0 - rho ** 2) * dt
    drift = (rate - np.sum(rho * kappa * theta / gamma)) * dt
    return np.array([drift, *current, *endpoint, *residual], dtype=float)

import numpy as np
def compute_cir_transition_parameters(
    kappa: float, theta: float, gamma: float, dt: float
) -> "np.ndarray":
    values = np.asarray([kappa, theta, gamma, dt], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("all CIR inputs must be finite and positive")
    decay_loss = -np.expm1(-kappa * dt)
    decay = np.exp(-kappa * dt)
    scale = gamma ** 2 * decay_loss / (4.0 * kappa)
    degrees = 4.0 * kappa * theta / gamma ** 2
    multiplier = 4.0 * kappa * decay / (gamma ** 2 * decay_loss)
    return np.array([scale, degrees, multiplier], dtype=float)

import numpy as np
def simulate_cir_variance_paths(
    initial_variance: float, transition_parameters: "np.ndarray",
    n_steps: int, n_paths: int, seed: int,
) -> "np.ndarray":
    params = np.asarray(transition_parameters, dtype=float)
    if params.shape != (3,) or not np.all(np.isfinite(params)):
        raise ValueError("transition_parameters must be a finite length-three array")
    if params[0] <= 0.0 or params[1] <= 0.0 or params[2] < 0.0:
        raise ValueError("transition scale and degrees must be positive")
    if initial_variance < 0.0 or not np.isfinite(initial_variance):
        raise ValueError("initial_variance must be finite and nonnegative")
    for value in (n_steps, n_paths, seed):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts and seed must be integers")
    if n_steps < 1 or n_paths < 1 or seed < 0:
        raise ValueError("counts must be positive and seed nonnegative")
    if params[2] < 0.0:
        raise ValueError("noncentrality multiplier must be nonnegative")
    rng = np.random.default_rng(int(seed))
    paths = np.empty((n_steps + 1, n_paths), dtype=float)
    paths[0] = float(initial_variance)
    for step in range(n_steps):
        noncentrality = params[2] * paths[step]
        draw = rng.noncentral_chisquare(params[1], noncentrality)
        paths[step + 1] = params[0] * draw
    return paths

import numpy as np
def simulate_double_heston_aes_paths(
    initial_spot: float, coefficients: "np.ndarray", variance_one: "np.ndarray",
    variance_two: "np.ndarray", seed: int,
) -> "np.ndarray":
    coeff = np.asarray(coefficients, dtype=float)
    v1 = np.asarray(variance_one, dtype=float); v2 = np.asarray(variance_two, dtype=float)
    if coeff.shape != (7,) or not np.all(np.isfinite(coeff)):
        raise ValueError("coefficients must be a finite length-seven array")
    if v1.ndim != 2 or v1.shape != v2.shape or v1.shape[0] < 2:
        raise ValueError("variance arrays must have one equal two-dimensional shape")
    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        raise ValueError("variance paths must be finite")
    if np.any(v1 < 0.0) or np.any(v2 < 0.0) or np.any(coeff[5:] < 0.0):
        raise ValueError("variance paths and residual coefficients must be nonnegative")
    if initial_spot <= 0.0 or not np.isfinite(initial_spot):
        raise ValueError("initial_spot must be finite and positive")
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    log_spot = np.empty_like(v1); log_spot[0] = np.log(initial_spot)
    for step in range(v1.shape[0] - 1):
        z1 = rng.standard_normal(v1.shape[1]); z2 = rng.standard_normal(v1.shape[1])
        log_spot[step + 1] = (
            log_spot[step] + coeff[0] + coeff[1] * v1[step]
            + coeff[2] * v2[step] + coeff[3] * v1[step + 1]
            + coeff[4] * v2[step + 1] + np.sqrt(coeff[5] * v1[step]) * z1
            + np.sqrt(coeff[6] * v2[step]) * z2
        )
    return np.exp(log_spot)

import numpy as np
def apply_double_heston_lsm_step(
    spot: "np.ndarray", variance_one: "np.ndarray", variance_two: "np.ndarray",
    strike: float, current_step: int, cashflows: "np.ndarray",
    exercise_steps: "np.ndarray", rate: float, dt: float,
) -> "np.ndarray":
    arrays = [np.asarray(value, dtype=float) for value in
              (spot, variance_one, variance_two, cashflows, exercise_steps)]
    if (arrays[0].ndim != 1 or arrays[0].size == 0
            or any(value.shape != arrays[0].shape for value in arrays[1:])):
        raise ValueError("path inputs must be equal nonempty vectors")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("all path inputs must be finite")
    if np.any(arrays[1] < 0.0) or np.any(arrays[2] < 0.0):
        raise ValueError("variance states must be nonnegative")
    if not np.isfinite([strike, rate, dt]).all() or strike <= 0.0 or dt <= 0.0:
        raise ValueError("strike and dt must be positive finite values")
    spot, variance_one, variance_two, cashflows, exercise_steps = arrays
    if (isinstance(current_step, bool)
            or not isinstance(current_step, (int, np.integer))
            or current_step < 0 or np.any(exercise_steps < current_step)):
        raise ValueError("current_step and stored stopping dates are invalid")
    x = spot / strike
    design = np.column_stack([
        np.ones_like(x), x, variance_one, variance_two, x * x,
        x * variance_one, x * variance_two, variance_one * variance_one,
        variance_one * variance_two, variance_two * variance_two,
    ])
    payoff = np.maximum(strike - spot, 0.0); itm = payoff > 0.0
    continuation = np.zeros(spot.size, dtype=float)
    if np.any(itm):
        target = cashflows[itm] * np.exp(-rate * dt * (exercise_steps[itm] - current_step))
        beta = np.linalg.lstsq(design[itm], target, rcond=None)[0]; continuation[itm] = design[itm] @ beta
    exercise = itm & (payoff > continuation)
    return np.vstack([np.where(exercise, payoff, cashflows),
                      np.where(exercise, current_step, exercise_steps)]).astype(float)

import numpy as np
def price_discounted_cashflows(
    cashflows: "np.ndarray", exercise_steps: "np.ndarray", rate: float, dt: float,
) -> float:
    import math
    try:
        cashflows = [float(value) for value in cashflows]
        exercise_steps = [float(value) for value in exercise_steps]
    except (TypeError, ValueError) as exc:
        raise ValueError("cashflow state must contain real scalars") from exc
    if not cashflows or len(exercise_steps) != len(cashflows):
        raise ValueError("cashflows and exercise_steps must be equal nonempty vectors")
    values = cashflows + exercise_steps
    if not all(math.isfinite(value) for value in values):
        raise ValueError("cashflow state must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("cashflows and exercise steps must be nonnegative")
    if not math.isfinite(rate) or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("rate must be finite and dt must be positive")
    discounted = (cash * math.exp(-rate * dt * step)
                  for cash, step in zip(cashflows, exercise_steps))
    return float(sum(discounted) / len(cashflows))

import numpy as np
def summarize_bermudan_exercise_effect(
    bermudan_price: float, terminal_payoffs: "np.ndarray",
    exercise_steps: "np.ndarray", rate: float, dt: float, maturity_step: int,
) -> "np.ndarray":
    import numpy as np
    terminal = np.asarray(terminal_payoffs, dtype=float)
    stopping = np.asarray(exercise_steps, dtype=float)
    if terminal.ndim != 1 or terminal.size == 0 or stopping.shape != terminal.shape:
        raise ValueError("payoff and stopping arrays must be equal nonempty vectors")
    if not np.all(np.isfinite(terminal)) or not np.all(np.isfinite(stopping)):
        raise ValueError("payoff and stopping arrays must be finite")
    if np.any(terminal < 0.0) or np.any(stopping < 0.0):
        raise ValueError("payoffs and stopping indices must be nonnegative")
    if not np.isfinite(bermudan_price) or bermudan_price < 0.0:
        raise ValueError("bermudan_price must be finite and nonnegative")
    if not np.isfinite(rate) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("rate must be finite and dt must be positive")
    if isinstance(maturity_step, bool) or not isinstance(maturity_step, (int, np.integer)):
        raise ValueError("maturity_step must be an integer")
    if maturity_step < 1 or np.any(stopping > maturity_step):
        raise ValueError("stopping indices cannot exceed a positive maturity_step")
    maturity_steps = np.full(terminal.size, maturity_step, dtype=float)
    maturity_value = price_discounted_cashflows(
        terminal, maturity_steps, rate, dt
    )
    early_count = int(np.count_nonzero(stopping < maturity_step))
    premium = float(bermudan_price - maturity_value)
    mean_increment = 0.0 if early_count == 0 else premium * terminal.size / early_count
    return np.array([maturity_value, early_count, mean_increment], dtype=float)

import numpy as np
def run_double_heston_aes_bermudan(
    n_paths: int = 4096, n_steps: int = 6, seed_v1: int = 173,
    seed_v2: int = 271, seed_price: int = 389,
) -> float:
    import numpy as np
    for value in (n_paths, n_steps, seed_v1, seed_v2, seed_price):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts and seeds must be integers")
    if n_paths < 1 or n_steps < 1 or min(seed_v1, seed_v2, seed_price) < 0:
        raise ValueError("counts must be positive and seeds nonnegative")
    spot0 = strike = 61.9
    maturity, rate = 0.25, 0.03
    kappa = np.array([0.9, 1.2]); theta = np.array([0.1, 0.15])
    gamma = np.array([0.1, 0.2]); rho = np.array([-0.5, -0.5])
    initial = np.array([0.2, 0.49]); dt = maturity / n_steps
    coeff = compute_double_heston_aes_coefficients(dt, rate, kappa, theta, gamma, rho)
    trans1 = compute_cir_transition_parameters(kappa[0], theta[0], gamma[0], dt)
    trans2 = compute_cir_transition_parameters(kappa[1], theta[1], gamma[1], dt)
    v1 = simulate_cir_variance_paths(initial[0], trans1, n_steps, n_paths, seed_v1)
    v2 = simulate_cir_variance_paths(initial[1], trans2, n_steps, n_paths, seed_v2)
    spot = simulate_double_heston_aes_paths(spot0, coeff, v1, v2, seed_price)
    terminal_payoffs = np.maximum(strike - spot[-1], 0.0)
    cashflows = terminal_payoffs.copy()
    exercise_steps = np.full(n_paths, n_steps, dtype=float)
    for step in range(n_steps - 1, 0, -1):
        state = apply_double_heston_lsm_step(
            spot[step], v1[step], v2[step], strike, step, cashflows, exercise_steps, rate, dt)
        cashflows, exercise_steps = state[0], state[1]
    bermudan_price = price_discounted_cashflows(cashflows, exercise_steps, rate, dt)
    diagnostics = summarize_bermudan_exercise_effect(
        bermudan_price, terminal_payoffs, exercise_steps, rate, dt, n_steps)
    maturity_value, early_count, mean_increment = diagnostics
    return float(maturity_value + mean_increment * early_count / n_paths)

import numpy as np
def simulate_double_heston_truncated_euler_paths(
    n_paths: int, n_steps: int, seed_v1: int, seed_v2: int, seed_price: int,
) -> "np.ndarray":
    values = (n_paths, n_steps, seed_v1, seed_v2, seed_price)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError("counts and seeds must be integers")
    if n_paths < 1 or n_steps < 1 or min(seed_v1, seed_v2, seed_price) < 0:
        raise ValueError("counts must be positive and seeds nonnegative")
    kappa = np.array([0.9, 1.2]); theta = np.array([0.1, 0.15])
    gamma = np.array([0.1, 0.2]); rho = np.array([-0.5, -0.5])
    dt = 0.25 / n_steps
    variance_rngs = [np.random.default_rng(seed_v1), np.random.default_rng(seed_v2)]
    price_rng = np.random.default_rng(seed_price)
    paths = np.empty((3, n_steps + 1, n_paths), dtype=float)
    paths[:, 0] = np.array([61.9, 0.2, 0.49])[:, None]
    for step in range(n_steps):
        zv = [rng.standard_normal(n_paths) for rng in variance_rngs]
        zs = [price_rng.standard_normal(n_paths) for _ in range(2)]
        variances = paths[1:, step]
        correlated = [rho[j] * zv[j] + np.sqrt(1.0 - rho[j] ** 2) * zs[j]
                      for j in range(2)]
        return_shock = sum(np.sqrt(variances[j] * dt) * correlated[j]
                           for j in range(2))
        paths[0, step + 1] = paths[0, step] * (1.0 + 0.03 * dt + return_shock)
        for j in range(2):
            proposal = (variances[j] + kappa[j] * (theta[j] - variances[j]) * dt
                        + gamma[j] * np.sqrt(variances[j] * dt) * zv[j])
            paths[j + 1, step + 1] = np.maximum(proposal, 0.0)
    return paths

import numpy as np
def run_double_heston_accuracy_benchmark(
    n_paths: int = 16384, n_steps: int = 12, seed_v1: int = 173,
    seed_v2: int = 271, seed_price: int = 389, reference_price: float = 9.504,
) -> float:
    if (any(isinstance(x, bool) or not isinstance(x, (int, np.integer))
            for x in (n_paths, n_steps, seed_v1, seed_v2, seed_price))
            or n_paths < 1 or n_steps < 1 or min(seed_v1, seed_v2, seed_price) < 0):
        raise ValueError("counts must be positive integers and seeds nonnegative")
    if not np.isfinite(reference_price) or reference_price <= 0.0:
        raise ValueError("reference_price must be finite and positive")
    kappa = np.array([0.9, 1.2]); theta = np.array([0.1, 0.15])
    gamma = np.array([0.1, 0.2]); rho = np.array([-0.5, -0.5])
    initial = np.array([0.2, 0.49]); dt = 0.25 / n_steps
    coeff = compute_double_heston_aes_coefficients(dt, 0.03, kappa, theta, gamma, rho)
    trans1 = compute_cir_transition_parameters(kappa[0], theta[0], gamma[0], dt)
    trans2 = compute_cir_transition_parameters(kappa[1], theta[1], gamma[1], dt)
    v1 = simulate_cir_variance_paths(initial[0], trans1, n_steps, n_paths, seed_v1)
    v2 = simulate_cir_variance_paths(initial[1], trans2, n_steps, n_paths, seed_v2)
    spot = simulate_double_heston_aes_paths(61.9, coeff, v1, v2, seed_price)
    terminal = np.maximum(61.9 - spot[-1], 0.0)
    cashflows = terminal.copy(); exercise_steps = np.full(n_paths, n_steps, dtype=float)
    for step in range(n_steps - 1, 0, -1):
        state = apply_double_heston_lsm_step(
            spot[step], v1[step], v2[step], 61.9, step,
            cashflows, exercise_steps, 0.03, dt)
        cashflows, exercise_steps = state
    direct = price_discounted_cashflows(cashflows, exercise_steps, 0.03, dt)
    diagnostic = summarize_bermudan_exercise_effect(
        direct, terminal, exercise_steps, 0.03, dt, n_steps)
    reconstructed = diagnostic[0] + diagnostic[2] * diagnostic[1] / n_paths
    aes_price = run_double_heston_aes_bermudan(
        n_paths, n_steps, seed_v1, seed_v2, seed_price)
    if not np.isclose(aes_price, reconstructed, rtol=0.0, atol=1e-12):
        raise ValueError("long-step public-chain results are inconsistent")
    euler = simulate_double_heston_truncated_euler_paths(
        n_paths, n_steps, seed_v1, seed_v2, seed_price)
    strike, rate = 61.9, 0.03
    cashflows = np.maximum(strike - euler[0, -1], 0.0)
    exercise_steps = np.full(n_paths, n_steps, dtype=float)
    for step in range(n_steps - 1, 0, -1):
        state = apply_double_heston_lsm_step(
            euler[0, step], euler[1, step], euler[2, step], strike, step,
            cashflows, exercise_steps, rate, dt)
        cashflows, exercise_steps = state
    euler_price = price_discounted_cashflows(
        cashflows, exercise_steps, rate, dt)
    aes_error = abs(aes_price - reference_price)
    if aes_error <= np.finfo(float).eps * max(1.0, abs(reference_price)):
        raise ValueError("long-step reference error is numerically zero")
    return float(abs(euler_price - reference_price) / aes_error)
SCICODE_GOLD_EOF
