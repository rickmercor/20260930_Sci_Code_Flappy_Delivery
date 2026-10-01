#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_retained_kernel(H: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    matrix = np.asarray(H, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("H must be square with at least two states")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("H must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("H must be symmetric within atol=1e-12")
    try:
        threshold = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be finite and strictly positive") from exc
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("cutoff must be finite and strictly positive")

    magnitudes = np.abs(matrix)
    retained = (magnitudes >= threshold) & (~np.eye(matrix.shape[0], dtype=bool))
    W = np.where(retained, magnitudes, 0.0)
    row_mass = np.sum(W, axis=1, dtype=float)
    if np.any(row_mass <= 0.0):
        raise ValueError("every state must have at least one retained neighbor")
    Q = W / row_mass[:, None]
    return W.astype(float), Q.astype(float)

def compute_ir_log_fields(
    retained_magnitudes: np.ndarray,
    log_abs_psi: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    def lse(values: np.ndarray) -> float:
        maximum = float(np.max(values))
        return maximum + float(np.log(np.sum(np.exp(values - maximum), dtype=float)))

    W = np.asarray(retained_magnitudes, dtype=float)
    logs = np.asarray(log_abs_psi, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] < 2:
        raise ValueError("retained_magnitudes must be square with at least two states")
    if not np.all(np.isfinite(W)) or np.any(W < 0.0):
        raise ValueError("retained_magnitudes must be finite and nonnegative")
    if not np.allclose(W, W.T, rtol=0.0, atol=1e-12):
        raise ValueError("retained_magnitudes must be symmetric")
    if not np.allclose(np.diag(W), 0.0, rtol=0.0, atol=0.0):
        raise ValueError("retained_magnitudes must have zero diagonal")
    row_mass = np.sum(W, axis=1, dtype=float)
    if np.any(row_mass <= 0.0):
        raise ValueError("each retained row mass must be positive")
    if logs.ndim != 1 or logs.shape != (W.shape[0],) or not np.all(np.isfinite(logs)):
        raise ValueError("log_abs_psi must be a finite vector matching the state count")
    try:
        exponent = float(alpha)
        displacement = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha and beta must be finite scalars") from exc
    if not np.isfinite(exponent) or exponent < 0.0 or exponent > 2.0:
        raise ValueError("alpha must lie in [0, 2]")
    if not np.isfinite(displacement) or displacement < 0.0 or displacement >= 1.0:
        raise ValueError("beta must lie in [0, 1)")

    log_auxiliary = np.log(row_mass) + exponent * logs
    n_states = W.shape[0]
    log_evaluation = np.empty(n_states, dtype=float)
    score = np.empty(n_states, dtype=float)
    second_ratio = np.empty(n_states, dtype=float)
    for y in range(n_states):
        term_logs: list[float] = []
        term_amplitude_logs: list[float] = []
        term_logs.append(float(np.log1p(-displacement) + np.log(row_mass[y]) + exponent * logs[y]))
        term_amplitude_logs.append(float(logs[y]))
        if displacement > 0.0:
            for x in np.flatnonzero(W[:, y] > 0.0):
                term_logs.append(float(np.log(displacement) + np.log(W[x, y]) + exponent * logs[x]))
                term_amplitude_logs.append(float(logs[x]))
        values = np.asarray(term_logs, dtype=float)
        amplitudes = np.asarray(term_amplitude_logs, dtype=float)
        log_total = lse(values)
        mixture_weights = np.exp(values - log_total)
        log_evaluation[y] = log_total
        score[y] = float(np.dot(mixture_weights, amplitudes))
        second_ratio[y] = float(np.dot(mixture_weights, amplitudes * amplitudes))
    return (
        log_auxiliary.astype(float),
        log_evaluation.astype(float),
        score.astype(float),
        second_ratio.astype(float),
    )

def run_log_hamiltonian_chain(
    proposal: np.ndarray,
    log_auxiliary: np.ndarray,
    initial_state: int,
    proposal_draws: np.ndarray,
    accept_draws: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Integral
    import numpy as np

    Q = np.asarray(proposal, dtype=float)
    log_mass = np.asarray(log_auxiliary, dtype=float)
    p_draws = np.asarray(proposal_draws, dtype=float)
    a_draws = np.asarray(accept_draws, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] < 2:
        raise ValueError("proposal must be square with at least two states")
    if not np.all(np.isfinite(Q)) or np.any(Q < 0.0):
        raise ValueError("proposal must be finite and nonnegative")
    if not np.allclose(np.sum(Q, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("proposal rows must sum to one")
    if np.any((Q > 0.0) & (Q.T <= 0.0)):
        raise ValueError("proposal support must be reversible")
    if log_mass.ndim != 1 or log_mass.shape != (Q.shape[0],) or not np.all(np.isfinite(log_mass)):
        raise ValueError("log_auxiliary must be a finite vector matching proposal")
    if isinstance(initial_state, bool) or not isinstance(initial_state, Integral):
        raise ValueError("initial_state must be an integer")
    current = int(initial_state)
    if current < 0 or current >= Q.shape[0]:
        raise ValueError("initial_state is out of range")
    if p_draws.ndim != 1 or a_draws.ndim != 1 or p_draws.size < 1 or p_draws.shape != a_draws.shape:
        raise ValueError("proposal_draws and accept_draws must be equal nonempty vectors")
    if not np.all(np.isfinite(p_draws)) or np.any((p_draws < 0.0) | (p_draws >= 1.0)):
        raise ValueError("proposal_draws must lie in [0, 1)")
    if not np.all(np.isfinite(a_draws)) or np.any((a_draws < 0.0) | (a_draws >= 1.0)):
        raise ValueError("accept_draws must lie in [0, 1)")

    trace = np.empty((p_draws.size, 5), dtype=float)
    for move, (proposal_draw, accept_draw) in enumerate(zip(p_draws, a_draws)):
        cumulative = 0.0
        trial = -1
        last_positive = -1
        for state, probability in enumerate(Q[current]):
            if probability > 0.0:
                last_positive = state
            cumulative += float(probability)
            if cumulative > float(proposal_draw):
                trial = state
                break
        if trial < 0:
            trial = last_positive
        log_ratio = (
            log_mass[trial]
            - log_mass[current]
            + np.log(Q[trial, current])
            - np.log(Q[current, trial])
        )
        log_acceptance = min(0.0, float(log_ratio))
        acceptance_probability = float(np.exp(log_acceptance))
        accepted = bool(accept_draw == 0.0 or np.log(accept_draw) < log_acceptance)
        pre_state = current
        if accepted:
            current = int(trial)
        trace[move] = (
            float(pre_state),
            float(trial),
            acceptance_probability,
            1.0 if accepted else 0.0,
            float(current),
        )
    return trace

def select_evaluation_histogram(
    chain_trace: np.ndarray,
    beta: float,
    kernel_draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    trace = np.asarray(chain_trace, dtype=float)
    draws = np.asarray(kernel_draws, dtype=float)
    if trace.ndim != 2 or trace.shape[1] != 5 or trace.shape[0] < 1 or not np.all(np.isfinite(trace)):
        raise ValueError("chain_trace must be a finite nonempty array with five columns")
    integer_columns = trace[:, [0, 1, 3, 4]]
    if not np.all(integer_columns == np.floor(integer_columns)) or np.any(integer_columns < 0.0):
        raise ValueError("state and accepted columns must contain nonnegative integers")
    accepted = trace[:, 3].astype(np.int64)
    if np.any((accepted != 0) & (accepted != 1)):
        raise ValueError("accepted_flag must contain only zero or one")
    probabilities = trace[:, 2]
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("acceptance probabilities must lie in [0, 1]")
    pre = trace[:, 0].astype(np.int64)
    trial = trace[:, 1].astype(np.int64)
    post = trace[:, 4].astype(np.int64)
    expected_post = np.where(accepted == 1, trial, pre)
    if not np.array_equal(post, expected_post):
        raise ValueError("post_state is inconsistent with accepted_flag")
    try:
        displacement = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be finite and in [0, 1)") from exc
    if not np.isfinite(displacement) or displacement < 0.0 or displacement >= 1.0:
        raise ValueError("beta must be finite and in [0, 1)")
    if draws.ndim != 1 or draws.shape != (trace.shape[0],):
        raise ValueError("kernel_draws must match the number of moves")
    if not np.all(np.isfinite(draws)) or np.any((draws < 0.0) | (draws >= 1.0)):
        raise ValueError("kernel_draws must lie in [0, 1)")

    evaluation_states = np.where(draws < displacement, trial, pre).astype(np.int64)
    unique_states, counts = np.unique(evaluation_states, return_counts=True)
    return evaluation_states, unique_states.astype(np.int64), counts.astype(np.int64)

def compute_adaptive_reweighting(
    H: np.ndarray,
    amplitude_signs: np.ndarray,
    log_abs_psi: np.ndarray,
    log_evaluation: np.ndarray,
    exponent_score: np.ndarray,
    exponent_second_ratio: np.ndarray,
    unique_states: np.ndarray,
    counts: np.ndarray,
    alpha: float,
    damping: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    def lse(values: np.ndarray) -> float:
        maximum = float(np.max(values))
        return maximum + float(np.log(np.sum(np.exp(values - maximum), dtype=float)))

    def signed_sum(log_magnitudes: np.ndarray, term_signs: np.ndarray) -> float:
        positive = log_magnitudes[term_signs > 0.0]
        negative = log_magnitudes[term_signs < 0.0]
        lp = lse(positive) if positive.size else -np.inf
        ln = lse(negative) if negative.size else -np.inf
        if not np.isfinite(ln):
            return float(np.exp(lp))
        if not np.isfinite(lp):
            return -float(np.exp(ln))
        if lp == ln:
            return 0.0
        if lp > ln:
            log_abs = lp + float(np.log(-np.expm1(ln - lp)))
            sign = 1.0
        else:
            log_abs = ln + float(np.log(-np.expm1(lp - ln)))
            sign = -1.0
        if log_abs > np.log(np.finfo(float).max):
            raise ValueError("local energy magnitude exceeds binary64 range")
        return sign * float(np.exp(log_abs))

    matrix = np.asarray(H, dtype=float)
    signs = np.asarray(amplitude_signs, dtype=float)
    logs = np.asarray(log_abs_psi, dtype=float)
    log_eval = np.asarray(log_evaluation, dtype=float)
    score = np.asarray(exponent_score, dtype=float)
    second = np.asarray(exponent_second_ratio, dtype=float)
    states_raw = np.asarray(unique_states)
    multiplicities_raw = np.asarray(counts)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("H must be square with at least two states")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("H must be finite and symmetric")
    n_states = matrix.shape[0]
    for name, array in (
        ("amplitude_signs", signs),
        ("log_abs_psi", logs),
        ("log_evaluation", log_eval),
        ("exponent_score", score),
        ("exponent_second_ratio", second),
    ):
        if array.ndim != 1 or array.shape != (n_states,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite state vector")
    if np.any((signs != -1.0) & (signs != 1.0)):
        raise ValueError("amplitude_signs must contain only -1 and +1")
    if states_raw.ndim != 1 or states_raw.size < 1 or not np.all(states_raw == np.floor(states_raw)):
        raise ValueError("unique_states must be a nonempty integer vector")
    states = states_raw.astype(np.int64)
    if np.any(states < 0) or np.any(states >= n_states) or np.any(np.diff(states) <= 0):
        raise ValueError("unique_states must be strictly increasing and in range")
    if multiplicities_raw.ndim != 1 or multiplicities_raw.shape != states.shape or not np.all(multiplicities_raw == np.floor(multiplicities_raw)):
        raise ValueError("counts must be an integer vector matching unique_states")
    multiplicities = multiplicities_raw.astype(np.int64)
    if np.any(multiplicities <= 0):
        raise ValueError("counts must be strictly positive")
    try:
        exponent = float(alpha)
        damp = float(damping)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha and damping must be finite scalars") from exc
    if not np.isfinite(exponent) or exponent < 0.0 or exponent > 2.0:
        raise ValueError("alpha must lie in [0, 2]")
    if not np.isfinite(damp) or damp <= 0.0 or damp > 1.0:
        raise ValueError("damping must lie in (0, 1]")

    local_energies = np.empty(states.size, dtype=float)
    for position, state in enumerate(states):
        nonzero = np.flatnonzero(matrix[state] != 0.0)
        term_logs = np.log(np.abs(matrix[state, nonzero])) + logs[nonzero] - logs[state]
        term_signs = np.sign(matrix[state, nonzero]) * signs[nonzero] * signs[state]
        local_energies[position] = signed_sum(term_logs.astype(float), term_signs.astype(float))
    if not np.all(np.isfinite(local_energies)):
        raise ValueError("local energies must be finite")

    grouped_log_weights = np.log(multiplicities.astype(float)) + 2.0 * logs[states] - log_eval[states]
    normalized_log_weights = grouped_log_weights - lse(grouped_log_weights)
    grouped_weights = np.exp(normalized_log_weights)
    energy = float(np.dot(grouped_weights, local_energies))
    total_count = float(np.sum(multiplicities))
    mean_score = float(np.dot(multiplicities, score[states]) / total_count)
    mean_second = float(np.dot(multiplicities, second[states]) / total_count)
    curvature = mean_second - mean_score * mean_score
    residual_terms = grouped_weights * np.abs(local_energies - energy)
    residual_normalization = float(np.sum(residual_terms))
    projected_alpha = exponent
    next_alpha = exponent
    if (
        np.isfinite(curvature)
        and curvature > 0.0
        and np.isfinite(residual_normalization)
        and residual_normalization > 0.0
    ):
        residual_weights = residual_terms / residual_normalization
        correction = (float(np.dot(residual_weights, score[states])) - mean_score) / curvature
        proposed = exponent + correction
        projected_alpha = float(np.clip(proposed, 0.0, 2.0))
        next_alpha = exponent + damp * (projected_alpha - exponent)
    diagnostics = np.array([energy, curvature, projected_alpha, next_alpha], dtype=float)
    return grouped_weights.astype(float), local_energies.astype(float), diagnostics

def solve_predictive_sample_space_sr(
    log_derivatives: np.ndarray,
    unique_states: np.ndarray,
    grouped_weights: np.ndarray,
    local_energies: np.ndarray,
    previous_direction: np.ndarray,
    predictor_coefficient: float,
    diagonal_shift: float,
) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    operators = np.asarray(log_derivatives, dtype=float)
    states_raw = np.asarray(unique_states)
    weights = np.asarray(grouped_weights, dtype=float)
    energies = np.asarray(local_energies, dtype=float)
    previous = np.asarray(previous_direction, dtype=float)
    if operators.ndim != 2 or operators.shape[0] < 2 or operators.shape[1] < 1 or not np.all(np.isfinite(operators)):
        raise ValueError("log_derivatives must be a finite state-by-parameter matrix")
    if states_raw.ndim != 1 or states_raw.size < 1 or not np.all(states_raw == np.floor(states_raw)):
        raise ValueError("unique_states must be a nonempty integer vector")
    states = states_raw.astype(np.int64)
    if np.any(states < 0) or np.any(states >= operators.shape[0]) or np.any(np.diff(states) <= 0):
        raise ValueError("unique_states must be strictly increasing and in range")
    if weights.ndim != 1 or weights.shape != states.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("grouped_weights must be finite, positive, and match unique_states")
    if not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("grouped_weights must sum to one")
    if energies.ndim != 1 or energies.shape != states.shape or not np.all(np.isfinite(energies)):
        raise ValueError("local_energies must be finite and match unique_states")
    if previous.ndim != 1 or previous.shape != (operators.shape[1],) or not np.all(np.isfinite(previous)):
        raise ValueError("previous_direction must match the parameter count")
    try:
        predictor_scale = float(predictor_coefficient)
        shift = float(diagonal_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError("predictor_coefficient and diagonal_shift must be finite scalars") from exc
    if not np.isfinite(predictor_scale) or predictor_scale < 0.0:
        raise ValueError("predictor_coefficient must be nonnegative")
    if not np.isfinite(shift) or shift <= 0.0:
        raise ValueError("diagonal_shift must be strictly positive")

    sampled_operators = operators[states]
    operator_mean = np.sum(weights[:, None] * sampled_operators, axis=0)
    energy_mean = float(np.dot(weights, energies))
    centered = np.sqrt(weights)[:, None] * (sampled_operators - operator_mean)
    residual = 2.0 * np.sqrt(weights) * (energies - energy_mean)
    predictor = predictor_scale * previous
    system = centered @ centered.T + shift * np.eye(states.size, dtype=float)
    right_hand_side = residual - centered @ predictor
    try:
        sample_coefficients = np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError as exc:
        raise ValueError("sample-space SR system is singular") from exc
    direction = predictor + centered.T @ sample_coefficients
    if not np.all(np.isfinite(direction)):
        raise ValueError("SR direction must be finite")
    return direction.astype(float)

def run_two_block_ir_sr_update(
    H: np.ndarray,
    base_psi: np.ndarray,
    log_derivatives: np.ndarray,
    initial_parameters: np.ndarray,
    cutoff: float,
    initial_alpha: float,
    beta: float,
    initial_state: int,
    proposal_draws: np.ndarray,
    accept_draws: np.ndarray,
    kernel_draws: np.ndarray,
    diagonal_shift: float,
    predictor_coefficient: float,
    learning_rate: float,
    alpha_damping: float = 0.02,
) -> float:
    """Reference two-block orchestrator."""
    from numbers import Integral
    import numpy as np

    matrix = np.asarray(H, dtype=float)
    reference = np.asarray(base_psi, dtype=float)
    operators = np.asarray(log_derivatives, dtype=float)
    parameters = np.asarray(initial_parameters, dtype=float).copy()
    p_draws = np.asarray(proposal_draws, dtype=float)
    a_draws = np.asarray(accept_draws, dtype=float)
    k_draws = np.asarray(kernel_draws, dtype=float)
    if reference.ndim != 1 or reference.size < 2 or not np.all(np.isfinite(reference)) or np.any(reference == 0.0):
        raise ValueError("base_psi must be a finite nonzero state vector")
    if operators.ndim != 2 or operators.shape[0] != reference.size or operators.shape[1] < 1 or not np.all(np.isfinite(operators)):
        raise ValueError("log_derivatives must be finite and match base_psi")
    if parameters.ndim != 1 or parameters.shape != (operators.shape[1],) or not np.all(np.isfinite(parameters)):
        raise ValueError("initial_parameters must match the parameter count")
    if matrix.shape != (reference.size, reference.size):
        raise ValueError("H must match the state count")
    if isinstance(initial_state, bool) or not isinstance(initial_state, Integral):
        raise ValueError("initial_state must be an integer")
    if p_draws.ndim != 2 or p_draws.shape[0] != 2 or p_draws.shape[1] < 1:
        raise ValueError("proposal_draws must have shape (2, n_moves)")
    if a_draws.shape != p_draws.shape or k_draws.shape != p_draws.shape:
        raise ValueError("all draw arrays must share shape (2, n_moves)")
    try:
        exponent = float(initial_alpha)
        step_size = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_alpha and learning_rate must be finite scalars") from exc
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("learning_rate must be finite and strictly positive")

    retained_magnitudes, proposal = build_retained_kernel(matrix, cutoff)
    amplitude_signs = np.sign(reference).astype(float)
    base_logs = np.log(np.abs(reference))
    current_state = int(initial_state)
    previous_direction = np.zeros(operators.shape[1], dtype=float)
    for block in range(2):
        log_abs_psi = base_logs + operators @ parameters
        log_auxiliary, log_evaluation, score, second_ratio = compute_ir_log_fields(
            retained_magnitudes, log_abs_psi, exponent, beta
        )
        trace = run_log_hamiltonian_chain(
            proposal, log_auxiliary, current_state, p_draws[block], a_draws[block]
        )
        _, unique_states, counts = select_evaluation_histogram(
            trace, beta, k_draws[block]
        )
        grouped_weights, local_energies, diagnostics = compute_adaptive_reweighting(
            matrix,
            amplitude_signs,
            log_abs_psi,
            log_evaluation,
            score,
            second_ratio,
            unique_states,
            counts,
            exponent,
            alpha_damping,
        )
        direction = solve_predictive_sample_space_sr(
            operators,
            unique_states,
            grouped_weights,
            local_energies,
            previous_direction,
            predictor_coefficient,
            diagonal_shift,
        )
        parameters = parameters - step_size * direction
        current_state = int(trace[-1, 4])
        exponent = float(diagnostics[3])
        previous_direction = direction
    if not np.all(np.isfinite(parameters)):
        raise ValueError("updated parameters must be finite")
    return float(parameters[0])
SCICODE_GOLD_EOF
