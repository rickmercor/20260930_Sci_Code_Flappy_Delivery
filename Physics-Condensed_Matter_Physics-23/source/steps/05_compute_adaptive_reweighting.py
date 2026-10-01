"""
Compute complete-Hamiltonian local energies, stable grouped Born-to-evaluation weights, and the damped projected update of the auxiliary amplitude exponent from residual-weighted evaluation-mass derivatives.

Exact reweighting depends only on the estimator configuration after marginalization, and repeated states contribute through their multiplicities. The exponent adaptation compares a residual-weighted evaluation-mass score with its sample mean and scales that correction by a positive curvature estimate; projection and damping affect only the next block. Local energies still use the complete Hamiltonian rather than the retained proposal graph.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray], containing grouped normalized weights, complete-Hamiltonian local energies for the unique states, and diagnostics [energy, curvature, projected_alpha, next_alpha].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def _logsumexp(values: np.ndarray) -> float:
    """Return a stable log-sum-exp."""
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum), dtype=float)))

def _signed_exponential_sum(log_magnitudes: np.ndarray, signs: np.ndarray) -> float:
    """Evaluate a finite signed exponential sum with cancellation-aware log arithmetic."""
    log_magnitudes = np.asarray(log_magnitudes, dtype=float)
    signs = np.asarray(signs, dtype=float)
    positive = log_magnitudes[signs > 0.0]
    negative = log_magnitudes[signs < 0.0]
    log_positive = _logsumexp(positive) if positive.size else -np.inf
    log_negative = _logsumexp(negative) if negative.size else -np.inf
    if not np.isfinite(log_negative):
        return float(np.exp(log_positive))
    if not np.isfinite(log_positive):
        return -float(np.exp(log_negative))
    if log_positive == log_negative:
        return 0.0
    if log_positive > log_negative:
        log_abs = log_positive + float(np.log(-np.expm1(log_negative - log_positive)))
        output_sign = 1.0
    else:
        log_abs = log_negative + float(np.log(-np.expm1(log_positive - log_negative)))
        output_sign = -1.0
    return output_sign * float(np.exp(log_abs))

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
    """Compute grouped weights, local energies, and the next exponent.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric complete Hamiltonian.
    amplitude_signs : np.ndarray
        Vector containing exactly ``-1`` or ``+1`` for every state.
    log_abs_psi : np.ndarray
        Finite log magnitudes for every state.
    log_evaluation : np.ndarray
        Finite unnormalized evaluation log masses.
    exponent_score : np.ndarray
        First exponent derivative of the evaluation log mass.
    exponent_second_ratio : np.ndarray
        Second derivative of the evaluation mass divided by that mass.
    unique_states : np.ndarray
        Strictly increasing sampled state indices.
    counts : np.ndarray
        Positive integer multiplicities for the sampled states.
    alpha : float
        Current exponent in ``[0, 2]``.
    damping : float, default=0.02
        Finite exponent damping factor in ``(0, 1]``.

    Returns
    -------
    grouped_weights : np.ndarray
        Normalized grouped importance weights.
    local_energies : np.ndarray
        Complete-Hamiltonian local energies for ``unique_states``.
    diagnostics : np.ndarray
        ``[weighted_energy, curvature, projected_alpha, next_alpha]``.

    Raises
    ------
    ValueError
        If inputs are invalid, incompatible, or a local energy is not finite.
    """
    return (
        np.empty_like(np.asarray(unique_states), dtype=float),
        np.empty_like(np.asarray(unique_states), dtype=float),
        np.empty(4, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_adaptive_reweighting(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return deterministic unit-test specifications."""
    return [
        {
            "setup": """import numpy as np
H = np.array([[-1.0,-0.8,0.35],[-0.8,0.2,-0.55],[0.35,-0.55,0.7]], dtype=float)
amplitude_signs = np.array([1.0,-1.0,1.0])
log_abs_psi = np.log(np.array([0.82,0.37,0.145]))
log_evaluation = np.array([0.1,-0.7,-1.5])
exponent_score = np.array([-0.5,-0.9,-1.8])
exponent_second_ratio = np.array([0.4,1.0,3.5])
unique_states = np.array([0,1,2])
counts = np.array([2,1,3])
alpha = 1.35
""",
            "call": "np.concatenate([_part.ravel() for _part in compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
            "gold_call": "np.concatenate([_part.ravel() for _part in _oracle_compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
        },
        {
            "setup": """import numpy as np
H = np.diag(np.array([0.0,4.0], dtype=float))
amplitude_signs = np.ones(2)
log_abs_psi = np.zeros(2)
log_evaluation = np.zeros(2)
exponent_score = np.array([0.0,1.0])
exponent_second_ratio = np.array([0.5,2.5])
unique_states = np.array([0,1])
counts = np.array([1,1])
alpha = 1.9
""",
            "call": "np.concatenate([_part.ravel() for _part in compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
            "gold_call": "np.concatenate([_part.ravel() for _part in _oracle_compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
        },
        {
            "setup": """import numpy as np
H = np.diag(np.array([4.0,0.0], dtype=float))
amplitude_signs = np.ones(2)
log_abs_psi = np.zeros(2)
log_evaluation = np.zeros(2)
exponent_score = np.array([0.0,1.0])
exponent_second_ratio = np.array([0.5,2.5])
unique_states = np.array([0,1])
counts = np.array([1,1])
alpha = 0.1
""",
            "call": "np.concatenate([_part.ravel() for _part in compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
            "gold_call": "np.concatenate([_part.ravel() for _part in _oracle_compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
        },
        {
            "setup": """import numpy as np
H = np.eye(3, dtype=float) * 1.6
amplitude_signs = np.array([1.0,-1.0,1.0])
log_abs_psi = np.array([-5.0,0.0,5.0])
log_evaluation = np.array([-10.0,0.0,10.0])
exponent_score = np.array([-2.0,0.0,2.0])
exponent_second_ratio = np.array([4.0,0.0,4.0])
unique_states = np.array([0,1,2])
counts = np.array([1,2,1])
alpha = 1.2
""",
            "call": "np.concatenate([_part.ravel() for _part in compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
            "gold_call": "np.concatenate([_part.ravel() for _part in _oracle_compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
        },
        {
            "setup": """import numpy as np
H = np.array([[1.0,-1.0],[-1.0,2.0]], dtype=float)
amplitude_signs = np.ones(2)
log_abs_psi = np.array([-700.0,0.0])
log_evaluation = np.array([-1400.0,0.0])
exponent_score = np.array([-700.0,0.0])
exponent_second_ratio = np.array([490000.0,0.0])
unique_states = np.array([0,1])
counts = np.array([1,1000000])
alpha = 1.0
""",
            "call": "np.concatenate([_part.ravel() for _part in compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
            "gold_call": "np.concatenate([_part.ravel() for _part in _oracle_compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, alpha)])",
        },
        {
            "setup": """import numpy as np
H = np.eye(3)
amplitude_signs = np.ones(3)
log_abs_psi = np.zeros(3)
log_evaluation = np.zeros(3)
exponent_score = np.zeros(3)
exponent_second_ratio = np.ones(3)
unique_states = np.array([0,0])
counts = np.array([1,1])
def run_model():
    try:
        compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
H = np.eye(2)
amplitude_signs = np.ones(2)
log_abs_psi = np.zeros(2)
log_evaluation = np.zeros(2)
exponent_score = np.zeros(2)
exponent_second_ratio = np.ones(2)
unique_states = np.array([0,1])
counts = np.array([1.0,1.5])
def run_model():
    try:
        compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
H = np.eye(2)
amplitude_signs = np.ones(2)
log_abs_psi = np.zeros(2)
log_evaluation = np.zeros(2)
exponent_score = np.zeros(2)
exponent_second_ratio = np.ones(2)
unique_states = np.array([0,1])
counts = np.array([1,1])
def run_model():
    try:
        compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, 2.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_adaptive_reweighting(H, amplitude_signs, log_abs_psi, log_evaluation, exponent_score, exponent_second_ratio, unique_states, counts, 2.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
