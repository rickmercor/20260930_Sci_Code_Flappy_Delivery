"""
Run two sequential importance-reweighted sampling-and-optimization blocks by composing every preceding public operation, initializing block 1 with a zero preceding unscaled SR direction, carrying all mutated state into block 2, and returning the first final parameter.

There is no preceding optimization direction before block 1, so its predictive SR term starts from the zero vector. The second block is not an independent repeat: its amplitudes are reconstructed from the first parameter update, its chain begins from the first final state, its auxiliary exponent uses the first damped update, and its predictor uses the first unscaled direction. The final function must orchestrate the six public scientific operations rather than reproduce their internals.

Returns
-------
float, the first component of the parameter vector after two sequential updates as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral

import numpy as np

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
    """Run two coupled sampling, reweighting, adaptation, and predictive-SR blocks.

    The implementation must call ``build_retained_kernel``, ``compute_ir_log_fields``,
    ``run_log_hamiltonian_chain``, ``select_evaluation_histogram``,
    ``compute_adaptive_reweighting``, and ``solve_predictive_sample_space_sr``. Block 1
    uses a zero preceding unscaled direction; block 2 uses the direction from block 1.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric complete Hamiltonian.
    base_psi : np.ndarray
        Finite nonzero signed reference amplitudes.
    log_derivatives : np.ndarray
        Finite state-by-parameter matrix used by the log-linear ansatz and SR.
    initial_parameters : np.ndarray
        Finite initial parameter vector.
    cutoff : float
        Positive inclusive retained-connection threshold.
    initial_alpha : float
        Initial auxiliary exponent in ``[0, 2]``.
    beta : float
        Evaluation displacement probability in ``[0, 1)``.
    initial_state : int
        Zero-based chain state before block 1.
    proposal_draws : np.ndarray
        Array with shape ``(2, n_moves)`` containing categorical draws in ``[0, 1)``.
    accept_draws : np.ndarray
        Array with shape ``(2, n_moves)`` containing strict-acceptance draws in ``[0, 1)``.
    kernel_draws : np.ndarray
        Array with shape ``(2, n_moves)`` containing estimator-kernel draws in ``[0, 1)``.
    diagonal_shift : float
        Positive shifted-SR regularization.
    predictor_coefficient : float
        Nonnegative multiplier of the preceding unscaled direction.
    learning_rate : float
        Positive parameter-update scale applied after each SR solve.
    alpha_damping : float, default=0.02
        Exponent damping factor in ``(0, 1]``.

    Returns
    -------
    final_component : float
        First parameter component after block 2.

    Raises
    ------
    ValueError
        If any input violates this or a preceding step's contract.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_two_block_ir_sr_update(
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

    retained_magnitudes, proposal = _oracle_build_retained_kernel(matrix, cutoff)
    amplitude_signs = np.sign(reference).astype(float)
    base_logs = np.log(np.abs(reference))
    current_state = int(initial_state)
    previous_direction = np.zeros(operators.shape[1], dtype=float)
    for block in range(2):
        log_abs_psi = base_logs + operators @ parameters
        log_auxiliary, log_evaluation, score, second_ratio = _oracle_compute_ir_log_fields(
            retained_magnitudes, log_abs_psi, exponent, beta
        )
        trace = _oracle_run_log_hamiltonian_chain(
            proposal, log_auxiliary, current_state, p_draws[block], a_draws[block]
        )
        _, unique_states, counts = _oracle_select_evaluation_histogram(
            trace, beta, k_draws[block]
        )
        grouped_weights, local_energies, diagnostics = _oracle_compute_adaptive_reweighting(
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
        direction = _oracle_solve_predictive_sample_space_sr(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return deterministic unit-test specifications."""
    prompt_setup = 'import numpy as np\nH = np.array([\n    [-1.25, -0.80,  0.35,  0.00,  0.00,  0.00, -0.25],\n    [-0.80, -0.70, -0.55,  0.22,  0.00, -0.18,  0.00],\n    [ 0.35, -0.55,  0.15, -0.72,  0.28,  0.00,  0.16],\n    [ 0.00,  0.22, -0.72,  0.65, -0.40,  0.31,  0.00],\n    [ 0.00,  0.00,  0.28, -0.40, -0.35, -0.63,  0.27],\n    [ 0.00, -0.18,  0.00,  0.31, -0.63,  1.10, -0.48],\n    [-0.25,  0.00,  0.16,  0.00,  0.27, -0.48,  0.25],\n], dtype=float)\nbase_psi = np.array([0.82, -0.37, 0.145, -0.061, 0.29, -0.013, 0.51], dtype=float)\nlog_derivatives = np.array([\n    [ 0.15, -0.40,  0.64,  0.89,  0.99,  0.92,  0.62, -1.00, 1.00],\n    [ 0.75,  0.10,  0.99,  0.81, -0.26,  0.70, -0.23, -0.67, 0.44],\n    [-0.30,  0.65,  0.86, -0.16, -0.93,  0.36, -0.90, -0.33, 0.11],\n    [ 1.20, -0.55,  0.33, -0.95,  0.49, -0.03, -0.90,  0.00, 0.00],\n    [-0.80,  0.35, -0.35, -0.71,  0.80, -0.42, -0.21,  0.33, 0.11],\n    [ 0.45,  1.10, -0.87,  0.31, -0.70, -0.74,  0.63,  0.67, 0.44],\n    [-0.10, -0.85, -0.98,  0.99, -0.62, -0.94,  1.00,  1.00, 1.00],\n], dtype=float)\ninitial_parameters = np.array([0.30498276274429814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)\ncutoff = 0.20\ninitial_alpha = 1.35\nbeta = 0.47\ninitial_state = 3\nproposal_draws = np.array([\n    [0.91,0.05,0.95,0.65,0.73,0.93,0.70,0.10,0.96,0.80,0.40,0.35],\n    [0.18,0.84,0.33,0.99,0.58,0.07,0.76,0.44,0.61,0.28,0.92,0.15],\n], dtype=float)\naccept_draws = np.array([\n    [0.13,0.90,0.09,0.27,0.32,0.88,0.02,0.42,0.99,0.006,0.75,0.11],\n    [0.61,0.04,0.72,0.15,0.49,0.95,0.21,0.36,0.08,0.77,0.12,0.54],\n], dtype=float)\nkernel_draws = np.array([\n    [0.20,0.75,0.46,0.47,0.10,0.90,0.30,0.65,0.05,0.49,0.40,0.80],\n    [0.47,0.12,0.88,0.39,0.50,0.03,0.71,0.22,0.95,0.46,0.31,0.66],\n], dtype=float)\ndiagonal_shift = 0.22649441768635536\npredictor_coefficient = 0.13361074072574586\nlearning_rate = 0.22292696898140568\nalpha_damping = 0.02\n'
    call = "run_two_block_ir_sr_update(H, base_psi, log_derivatives, initial_parameters, cutoff, initial_alpha, beta, initial_state, proposal_draws, accept_draws, kernel_draws, diagonal_shift, predictor_coefficient, learning_rate, alpha_damping)"
    gold = "_oracle_run_two_block_ir_sr_update(H, base_psi, log_derivatives, initial_parameters, cutoff, initial_alpha, beta, initial_state, proposal_draws, accept_draws, kernel_draws, diagonal_shift, predictor_coefficient, learning_rate, alpha_damping)"
    return [
        {"setup": prompt_setup, "call": call, "gold_call": gold},
        {
            "setup": prompt_setup + "\naccept_draws = accept_draws.copy()\naccept_draws[1, 0] = 0.0\n",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": prompt_setup + "\nkernel_draws = kernel_draws.copy()\nkernel_draws[0, 3] = np.nextafter(beta, 0.0)\n",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": prompt_setup + "\nbeta = 0.0\n",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": prompt_setup + "\naccept_draws = accept_draws.copy()\naccept_draws[0, 0] = 0.085\n",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": prompt_setup + "\npredictor_coefficient = 0.0\n",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": prompt_setup + """
proposal_draws = proposal_draws[0]
def run_model():
    try:
        run_two_block_ir_sr_update(H, base_psi, log_derivatives, initial_parameters, cutoff, initial_alpha, beta, initial_state, proposal_draws, accept_draws, kernel_draws, diagonal_shift, predictor_coefficient, learning_rate, alpha_damping)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_two_block_ir_sr_update(H, base_psi, log_derivatives, initial_parameters, cutoff, initial_alpha, beta, initial_state, proposal_draws, accept_draws, kernel_draws, diagonal_shift, predictor_coefficient, learning_rate, alpha_damping)
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
            "setup": prompt_setup + """
def run_model():
    try:
        run_two_block_ir_sr_update(H, base_psi, log_derivatives, initial_parameters, cutoff, initial_alpha, beta, initial_state, proposal_draws, accept_draws, kernel_draws, diagonal_shift, predictor_coefficient, 0.0, alpha_damping)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_two_block_ir_sr_update(H, base_psi, log_derivatives, initial_parameters, cutoff, initial_alpha, beta, initial_state, proposal_draws, accept_draws, kernel_draws, diagonal_shift, predictor_coefficient, 0.0, alpha_damping)
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
