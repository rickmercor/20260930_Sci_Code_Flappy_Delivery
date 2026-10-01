"""
Propagate a deterministic Hamiltonian-guided Metropolis-Hastings chain in the log domain, recording each pre-state, categorical trial, acceptance probability, strict acceptance flag, and immediately mutated post-state.

A state-dependent Hamiltonian proposal requires the full forward/reverse Hastings correction unless its cancellation has already been established algebraically. Log-domain comparison prevents probability-ratio underflow, categorical boundaries use strict cumulative selection, and each accepted move changes the proposal row used by the next transition.

Returns
-------
np.ndarray, a binary64 trace with shape (n_moves, 5) and columns [pre_state, trial_state, acceptance_probability, accepted_flag, post_state].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral

import numpy as np

def _draw_categorical_strict(probabilities: np.ndarray, draw: float) -> int:
    """Select the first index whose cumulative probability is strictly greater than draw."""
    cumulative = 0.0
    last_positive = -1
    for index, probability in enumerate(np.asarray(probabilities, dtype=float)):
        if probability > 0.0:
            last_positive = index
        cumulative += float(probability)
        if cumulative > draw:
            return int(index)
    return int(last_positive)

def run_log_hamiltonian_chain(
    proposal: np.ndarray,
    log_auxiliary: np.ndarray,
    initial_state: int,
    proposal_draws: np.ndarray,
    accept_draws: np.ndarray,
) -> np.ndarray:
    """Run a sequential log-domain Metropolis-Hastings chain.

    Parameters
    ----------
    proposal : np.ndarray
        Row-stochastic proposal matrix with reversible positive support.
    log_auxiliary : np.ndarray
        Finite unnormalized auxiliary log masses.
    initial_state : int
        Zero-based state before the first supplied move.
    proposal_draws : np.ndarray
        One-dimensional categorical draws in ``[0, 1)``.
    accept_draws : np.ndarray
        One-dimensional strict-acceptance draws in ``[0, 1)``.

    Returns
    -------
    trace : np.ndarray
        Array with columns ``pre_state``, ``trial_state``, ``acceptance_probability``,
        ``accepted_flag``, and ``post_state``.

    Raises
    ------
    ValueError
        If the proposal, log masses, state, or draws are invalid or incompatible.
    """
    return np.empty((np.asarray(proposal_draws).size, 5), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_log_hamiltonian_chain(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return deterministic unit-test specifications."""
    return [
        {
            "setup": """import numpy as np
Q = np.array([[0.0,0.6,0.4],[0.3,0.0,0.7],[0.2,0.8,0.0]], dtype=float)
log_auxiliary = np.log(np.array([1.0,2.0,0.5], dtype=float))
initial_state = 0
proposal_draws = np.array([0.61,0.10,0.95,0.20], dtype=float)
accept_draws = np.array([0.10,0.90,0.01,0.50], dtype=float)
""",
            "call": "run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
            "gold_call": "_oracle_run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
        },
        {
            "setup": """import numpy as np
Q = np.array([[0.0,0.25,0.75],[0.5,0.0,0.5],[0.5,0.5,0.0]], dtype=float)
log_auxiliary = np.zeros(3, dtype=float)
initial_state = 0
proposal_draws = np.array([0.25], dtype=float)
accept_draws = np.array([0.0], dtype=float)
""",
            "call": "run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
            "gold_call": "_oracle_run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
        },
        {
            "setup": """import numpy as np
Q = np.array([[0.0,1.0],[1.0,0.0]], dtype=float)
log_auxiliary = np.array([0.0, np.log(0.25)], dtype=float)
initial_state = 0
proposal_draws = np.array([0.3], dtype=float)
accept_draws = np.array([0.25], dtype=float)
""",
            "call": "run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
            "gold_call": "_oracle_run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
        },
        {
            "setup": """import numpy as np
Q = np.array([[0.0,0.9,0.1],[0.2,0.0,0.8],[0.7,0.3,0.0]], dtype=float)
log_auxiliary = np.log(np.array([1.0,1.5,0.4], dtype=float))
initial_state = 0
proposal_draws = np.array([0.95,0.25], dtype=float)
accept_draws = np.array([0.5,0.5], dtype=float)
""",
            "call": "run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
            "gold_call": "_oracle_run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
        },
        {
            "setup": """import numpy as np
Q = np.array([[0.0,1.0],[1.0,0.0]], dtype=float)
log_auxiliary = np.array([0.0,-1000.0], dtype=float)
initial_state = 0
proposal_draws = np.array([0.2], dtype=float)
accept_draws = np.array([0.0], dtype=float)
""",
            "call": "run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
            "gold_call": "_oracle_run_log_hamiltonian_chain(Q, log_auxiliary, initial_state, proposal_draws, accept_draws).ravel()",
        },
        {
            "setup": """import numpy as np
Q = np.array([[0.0,1.0],[1.0,0.0]], dtype=float)
log_auxiliary = np.array([0.0,0.0], dtype=float)
def run_model():
    try:
        run_log_hamiltonian_chain(Q, log_auxiliary, 0, np.array([0.2,0.3]), np.array([0.1]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_log_hamiltonian_chain(Q, log_auxiliary, 0, np.array([0.2,0.3]), np.array([0.1]))
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
Q = np.array([[0.0,1.0],[0.0,1.0]], dtype=float)
log_auxiliary = np.array([0.0,0.0], dtype=float)
def run_model():
    try:
        run_log_hamiltonian_chain(Q, log_auxiliary, 0, np.array([0.2]), np.array([0.1]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_log_hamiltonian_chain(Q, log_auxiliary, 0, np.array([0.2]), np.array([0.1]))
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
