"""
Select one estimator configuration from every pre-transition chain state by either staying or reusing that move's Hamiltonian trial, then combine the realized configurations into an ascending unique-state histogram.

The estimator kernel is separate from the accept/reject transition. Conditional on a pre-transition chain state, the already generated Hamiltonian trial is distributed according to the required displacement proposal and can therefore be reused when the kernel displaces. Grouping repeated estimator states before later wave-function work is algebraically equivalent to assigning each occurrence the same importance weight.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray], containing the evaluation-state sequence, ascending unique states, and integer multiplicities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_evaluation_histogram(
    chain_trace: np.ndarray,
    beta: float,
    kernel_draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the strict stay-or-displace estimator kernel and group repeats.

    Parameters
    ----------
    chain_trace : np.ndarray
        Binary64 trace with shape ``(n_moves, 5)`` and columns ``pre_state``,
        ``trial_state``, ``acceptance_probability``, ``accepted_flag``, ``post_state``.
    beta : float
        Finite displacement probability in ``[0, 1)``.
    kernel_draws : np.ndarray
        One-dimensional draws in ``[0, 1)`` with one entry per move.

    Returns
    -------
    evaluation_states : np.ndarray
        Selected estimator states in move order.
    unique_states : np.ndarray
        Distinct estimator states in ascending order.
    counts : np.ndarray
        Positive integer multiplicities corresponding to ``unique_states``.

    Raises
    ------
    ValueError
        If the trace, probability, or draws are invalid or inconsistent.
    """
    return (
        np.empty(np.asarray(chain_trace).shape[0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_evaluation_histogram(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return deterministic unit-test specifications."""
    base = """import numpy as np
trace = np.array([[0,2,0.4,0,0],[0,1,1.0,1,1],[1,2,0.3,1,2],[2,0,0.2,0,2]], dtype=float)
"""
    return [
        {
            "setup": base + "beta = 0.47\nkernel_draws = np.array([0.2,0.8,0.47,0.1], dtype=float)\n",
            "call": "np.concatenate(select_evaluation_histogram(trace, beta, kernel_draws))",
            "gold_call": "np.concatenate(_oracle_select_evaluation_histogram(trace, beta, kernel_draws))",
        },
        {
            "setup": base + "beta = 0.0\nkernel_draws = np.array([0.0,0.2,0.7,0.9], dtype=float)\n",
            "call": "np.concatenate(select_evaluation_histogram(trace, beta, kernel_draws))",
            "gold_call": "np.concatenate(_oracle_select_evaluation_histogram(trace, beta, kernel_draws))",
        },
        {
            "setup": base + "beta = 0.999\nkernel_draws = np.array([0.0,0.2,0.7,0.998], dtype=float)\n",
            "call": "np.concatenate(select_evaluation_histogram(trace, beta, kernel_draws))",
            "gold_call": "np.concatenate(_oracle_select_evaluation_histogram(trace, beta, kernel_draws))",
        },
        {
            "setup": base + "beta = 0.47\nkernel_draws = np.array([0.47,0.47,0.47,0.47], dtype=float)\n",
            "call": "np.concatenate(select_evaluation_histogram(trace, beta, kernel_draws))",
            "gold_call": "np.concatenate(_oracle_select_evaluation_histogram(trace, beta, kernel_draws))",
        },
        {
            "setup": """import numpy as np
trace = np.array([[0,1,0.5,0,1]], dtype=float)
kernel_draws = np.array([0.2], dtype=float)
def run_model():
    try:
        select_evaluation_histogram(trace, 0.4, kernel_draws)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_evaluation_histogram(trace, 0.4, kernel_draws)
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
            "setup": base + """
def run_model():
    try:
        select_evaluation_histogram(trace, 0.4, np.array([0.2,0.3]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_evaluation_histogram(trace, 0.4, np.array([0.2,0.3]))
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
