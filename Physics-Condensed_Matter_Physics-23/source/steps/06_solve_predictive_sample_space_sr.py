"""
Solve the shifted predictive stochastic-reconfiguration system in sampled-configuration space, adding the projection of the preceding unscaled direction before any learning-rate scaling and returning the new unscaled parameter direction.

Centered logarithmic derivatives factor the SR covariance as a configuration-by-parameter matrix product. The push-through identity reduces the shifted solve to sampled-configuration space, and a predictor formed from the preceding unscaled direction enters the residual before that solve. The learning rate is deliberately absent from this operation.

Returns
-------
np.ndarray, the finite unscaled predictive SR direction with shape (n_parameters,) in binary64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_predictive_sample_space_sr(
    log_derivatives: np.ndarray,
    unique_states: np.ndarray,
    grouped_weights: np.ndarray,
    local_energies: np.ndarray,
    previous_direction: np.ndarray,
    predictor_coefficient: float,
    diagonal_shift: float,
) -> np.ndarray:
    """Compute a predictive shifted SR direction in sample space.

    Parameters
    ----------
    log_derivatives : np.ndarray
        Finite state-by-parameter logarithmic derivative matrix.
    unique_states : np.ndarray
        Strictly increasing sampled state indices.
    grouped_weights : np.ndarray
        Strictly positive normalized weights for ``unique_states``.
    local_energies : np.ndarray
        Finite local energies for ``unique_states``.
    previous_direction : np.ndarray
        Finite preceding unscaled direction with one entry per parameter.
    predictor_coefficient : float
        Finite nonnegative multiplier of the preceding direction.
    diagonal_shift : float
        Finite strictly positive SR shift.

    Returns
    -------
    direction : np.ndarray
        New unscaled predictive SR direction.

    Raises
    ------
    ValueError
        If inputs are invalid, incompatible, or the sample-space solve is singular or non-finite.
    """
    return np.empty(np.asarray(log_derivatives).shape[1], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_predictive_sample_space_sr(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return deterministic unit-test specifications."""
    return [
        {
            "setup": """import numpy as np
O = np.array([[0.1,-0.4,0.7],[0.8,0.2,-0.3],[-0.5,1.0,0.4],[0.9,-0.6,-0.2]], dtype=float)
unique_states = np.array([0,1,3])
grouped_weights = np.array([0.2,0.5,0.3])
local_energies = np.array([-1.2,0.7,2.1])
previous_direction = np.zeros(3)
predictor_coefficient = 0.3
diagonal_shift = 0.2
""",
            "call": "solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
            "gold_call": "_oracle_solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
        },
        {
            "setup": """import numpy as np
O = np.array([[0.1,-0.4,0.7,0.3],[0.8,0.2,-0.3,-0.1],[-0.5,1.0,0.4,0.2],[0.9,-0.6,-0.2,0.8]], dtype=float)
unique_states = np.array([0,2,3])
grouped_weights = np.array([0.55,0.15,0.30])
local_energies = np.array([-2.0,1.0,0.5])
previous_direction = np.array([0.3,-0.2,0.4,0.1])
predictor_coefficient = 0.8
diagonal_shift = 0.07
""",
            "call": "solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
            "gold_call": "_oracle_solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
        },
        {
            "setup": """import numpy as np
O = np.array([[1.0,0.0,1.0],[0.0,1.0,1.0],[1.0,1.0,0.0]], dtype=float)
unique_states = np.array([0,1,2])
grouped_weights = np.array([0.2,0.3,0.5])
local_energies = np.array([-1.0,0.2,1.4])
previous_direction = np.array([0.4,-0.5,0.1])
predictor_coefficient = 0.0
diagonal_shift = 0.3
""",
            "call": "solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
            "gold_call": "_oracle_solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
        },
        {
            "setup": """import numpy as np
O = np.array([[0.2,-0.1,0.7],[0.4,0.3,-0.2]], dtype=float)
unique_states = np.array([1])
grouped_weights = np.array([1.0])
local_energies = np.array([2.5])
previous_direction = np.array([0.3,-0.6,0.9])
predictor_coefficient = 0.4
diagonal_shift = 1e-8
""",
            "call": "solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
            "gold_call": "_oracle_solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
        },
        {
            "setup": """import numpy as np
O = np.array([[1.0,1.0,1.0,0.0],[1.0+1e-12,1.0,1.0,1e-12],[1.0,1.0+1e-12,1.0,-1e-12]], dtype=float)
unique_states = np.array([0,1,2])
grouped_weights = np.array([0.3,0.4,0.3])
local_energies = np.array([1.0,1.0+1e-8,1.0-1e-8])
previous_direction = np.zeros(4)
predictor_coefficient = 0.5
diagonal_shift = 1e-10
""",
            "call": "solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
            "gold_call": "_oracle_solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, predictor_coefficient, diagonal_shift)",
        },
        {
            "setup": """import numpy as np
O = np.eye(3)
unique_states = np.array([0,1,2])
grouped_weights = np.array([0.2,0.2,0.2])
local_energies = np.array([0.0,1.0,2.0])
previous_direction = np.zeros(3)
def run_model():
    try:
        solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, 0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, 0.0, 0.1)
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
O = np.eye(3)
unique_states = np.array([0,1,2])
grouped_weights = np.array([0.2,0.3,0.5])
local_energies = np.array([0.0,1.0,2.0])
previous_direction = np.zeros(2)
def run_model():
    try:
        solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, 0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_predictive_sample_space_sr(O, unique_states, grouped_weights, local_energies, previous_direction, 0.0, 0.1)
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
