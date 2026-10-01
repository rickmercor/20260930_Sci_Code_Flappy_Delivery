"""
Orchestrate the complete deterministic protein-acquisition pipeline. Call the preceding sequence-encoding, surrogate, potential, reflective-leapfrog, discretization, Metropolis, and ensemble-UCB steps in order, then return the maximum UCB among the accepted candidates.

Orchestrate the complete deterministic protein-acquisition pipeline. Call the preceding sequence-encoding, surrogate, potential, reflective-leapfrog, discretization, Metropolis, and ensemble-UCB steps, then return the maximum UCB among the accepted discrete candidates. For each surrogate ensemble member, begin from the same one-hot reference sequence. Initialize one independent NumPy random generator from that member's supplied seed and draw one standard-normal initial momentum matrix. Maintain the continuous position and momentum throughout the member's complete trajectory. At each of the specified updates, retain the pre-update position and momentum, perform one reflective leapfrog update, discretize the updated continuous position, and evaluate the resulting one-hot proposal with the same surrogate. Draw the next uniform variate from that member's generator and apply the discretization-aware Metropolis comparison between the pre-update continuous Hamiltonian and the discretized post-update Hamiltonian. Collect the discrete proposal only when it is accepted, while continuing the continuous trajectory regardless of that decision. After processing every update of every surrogate, treat identical accepted sequences as one candidate and preserve their first-occurrence order. Evaluate each distinct accepted candidate with the complete surrogate ensemble, calculate its mean-plus-population-standard-deviation UCB, and return the largest UCB. Raise ValueError if no proposal is accepted. The reflective-leapfrog function retains its supplied starting position as the fixed reference for the surrogate penalty. Therefore, obtain the state after update t by running the reflective-leapfrog function from the original one-hot state and original momentum for t steps. Use the preceding prefix endpoint as the pre-update state for that update's Metropolis comparison.

Returns
-------
float, the maximum ensemble UCB score across all distinct accepted discrete protein candidates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from importlib import import_module

import numpy as np


def optimize_protein_candidate(
    initial_sequence: np.ndarray,
    alphabet_size: int,
    weights_ensemble: np.ndarray,
    contact_matrix: np.ndarray,
    biases: np.ndarray,
    contact_scales: np.ndarray,
    penalty_scales: np.ndarray,
    seeds: np.ndarray,
    epsilon: float,
    n_steps: int,
) -> float:
    """Run all acquisition stages and return the maximum UCB.

    Parameters
    ----------
    initial_sequence : np.ndarray
        One-dimensional array of zero-based residue-class indices.
    alphabet_size : int
        Number of permitted residue classes.
    weights_ensemble : np.ndarray
        Surrogate weight matrices of shape ``(M, L, A)``.
    contact_matrix : np.ndarray
        Residue-contact matrix of shape ``(A, A)``.
    biases : np.ndarray
        Surrogate biases of shape ``(M,)``.
    contact_scales : np.ndarray
        Surrogate contact multipliers of shape ``(M,)``.
    penalty_scales : np.ndarray
        Surrogate penalty multipliers of shape ``(M,)``.
    seeds : np.ndarray
        One non-negative integer seed per ensemble member.
    epsilon : float
        Positive leapfrog step size.
    n_steps : int
        Positive number of leapfrog steps per trajectory.

    Raises
    ------
    ValueError
        If ensemble dimensions or seeds are invalid, a downstream
        input fails validation, or no candidate is accepted.

    Returns
    -------
    maximum_ucb : float
        Maximum UCB across the accepted discrete candidates.
    """
    return maximum_ucb

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_optimize_protein_candidate(
    initial_sequence: np.ndarray,
    alphabet_size: int,
    weights_ensemble: np.ndarray,
    contact_matrix: np.ndarray,
    biases: np.ndarray,
    contact_scales: np.ndarray,
    penalty_scales: np.ndarray,
    seeds: np.ndarray,
    epsilon: float,
    n_steps: int,
) -> float:
    """Return the source-style per-update acquisition result."""

    weights_array = np.asarray(
        weights_ensemble,
        dtype=np.float64,
    )
    biases_array = np.asarray(
        biases,
        dtype=np.float64,
    )
    contact_scales_array = np.asarray(
        contact_scales,
        dtype=np.float64,
    )
    penalty_scales_array = np.asarray(
        penalty_scales,
        dtype=np.float64,
    )
    seeds_array = np.asarray(seeds)

    if (
        weights_array.ndim != 3
        or weights_array.shape[0] < 1
    ):
        raise ValueError(
            "weights_ensemble must have shape (M, L, A)"
        )

    model_count = weights_array.shape[0]

    if not all(
        array.shape == (model_count,)
        for array in (
            biases_array,
            contact_scales_array,
            penalty_scales_array,
            seeds_array,
        )
    ):
        raise ValueError(
            "ensemble vectors and seeds must have shape (M,)"
        )

    if not np.issubdtype(
        seeds_array.dtype,
        np.integer,
    ):
        raise ValueError(
            "seeds must contain integers"
        )

    if np.any(seeds_array < 0):
        raise ValueError(
            "seeds must be non-negative"
        )

    if (
        isinstance(n_steps, (bool, np.bool_))
        or not isinstance(n_steps, (int, np.integer))
        or int(n_steps) < 1
    ):
        raise ValueError(
            "n_steps must be a positive integer"
        )

    q_init = _oracle_encode_sequence(
        initial_sequence,
        alphabet_size,
    )

    if weights_array.shape[1:] != q_init.shape:
        raise ValueError(
            "weights_ensemble must match the encoded sequence shape"
        )

    accepted_candidates: list[np.ndarray] = []
    seen_candidates: set[tuple[int, ...]] = set()

    for model_index in range(model_count):
        rng = np.random.default_rng(
            int(seeds_array[model_index])
        )

        momentum_initial = rng.normal(
            size=q_init.shape
        )

        position_before = q_init.copy()
        momentum_before = momentum_initial.copy()

        for update_number in range(
            1,
            int(n_steps) + 1,
        ):
            (
                score_before,
                score_gradient_before,
            ) = _oracle_surrogate_score_gradient(
                position_before,
                q_init,
                weights_array[model_index],
                contact_matrix,
                biases_array[model_index],
                contact_scales_array[model_index],
                penalty_scales_array[model_index],
            )

            (
                potential_before,
                _,
            ) = _oracle_potential_energy_gradient(
                score_before,
                score_gradient_before,
            )

            (
                position_after,
                momentum_after,
            ) = _oracle_reflective_leapfrog(
                q_init,
                momentum_initial,
                weights_array[model_index],
                contact_matrix,
                biases_array[model_index],
                contact_scales_array[model_index],
                penalty_scales_array[model_index],
                epsilon,
                update_number,
            )

            (
                residue_indices,
                discrete_candidate,
            ) = _oracle_discretize_positions(
                position_after
            )

            (
                score_proposed,
                score_gradient_proposed,
            ) = _oracle_surrogate_score_gradient(
                discrete_candidate,
                q_init,
                weights_array[model_index],
                contact_matrix,
                biases_array[model_index],
                contact_scales_array[model_index],
                penalty_scales_array[model_index],
            )

            (
                potential_proposed,
                _,
            ) = _oracle_potential_energy_gradient(
                score_proposed,
                score_gradient_proposed,
            )

            uniform_draw = float(rng.random())

            (
                _,
                accepted,
            ) = _oracle_metropolis_acceptance(
                potential_before,
                potential_proposed,
                momentum_before,
                momentum_after,
                uniform_draw,
            )

            if accepted == 1:
                candidate_key = tuple(
                    int(value)
                    for value in residue_indices.tolist()
                )

                if candidate_key not in seen_candidates:
                    seen_candidates.add(candidate_key)
                    accepted_candidates.append(
                        discrete_candidate
                    )

            position_before = position_after
            momentum_before = momentum_after

    if not accepted_candidates:
        raise ValueError(
            "no candidate was accepted"
        )

    ucb_scores = _oracle_ensemble_ucb(
        np.stack(
            accepted_candidates,
            axis=0,
        ),
        q_init,
        weights_array,
        contact_matrix,
        biases_array,
        contact_scales_array,
        penalty_scales_array,
    )

    return float(np.max(ucb_scores))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, minimum-ensemble, and invalid-seed tests."""
    return [
        {
            "setup": """import numpy as np
initial_sequence = np.array([0, 1, 2, 3, 0], dtype=int)
alphabet_size = 4
parameter_rng = np.random.default_rng(7049)
weights_ensemble = parameter_rng.normal(
    0.0, 0.85, size=(5, 5, 4)
).astype(np.float64)
raw_contact = parameter_rng.normal(
    0.0, 0.35, size=(4, 4)
).astype(np.float64)
contact_matrix = (raw_contact + raw_contact.T) / 2.0
biases = parameter_rng.normal(
    -0.15, 0.30, size=5
).astype(np.float64)
contact_scales = parameter_rng.uniform(
    0.35, 0.95, size=5
).astype(np.float64)
penalty_scales = parameter_rng.uniform(
    0.12, 0.42, size=5
).astype(np.float64)
seeds = parameter_rng.integers(
    1000, 100000, size=5, dtype=np.int64
)
epsilon = float(parameter_rng.uniform(0.48, 0.82))
n_steps = 12
""",
            "call": (
                "optimize_protein_candidate("
                "initial_sequence, alphabet_size, "
                "weights_ensemble, contact_matrix, biases, "
                "contact_scales, penalty_scales, seeds, "
                "epsilon, n_steps)"
            ),
            "gold_call": (
                "_oracle_optimize_protein_candidate("
                "initial_sequence, alphabet_size, "
                "weights_ensemble, contact_matrix, biases, "
                "contact_scales, penalty_scales, seeds, "
                "epsilon, n_steps)"
            ),
        },
        {
            "setup": """import numpy as np
initial_sequence = np.array([0, 1], dtype=int)
alphabet_size = 2
weights_ensemble = np.zeros((1, 2, 2), dtype=float)
contact_matrix = np.zeros((2, 2), dtype=float)
biases = np.array([-0.25])
contact_scales = np.array([0.0])
penalty_scales = np.array([0.0])
seeds = np.array([17], dtype=int)
epsilon = 0.2
n_steps = 1
""",
            "call": (
                "optimize_protein_candidate("
                "initial_sequence, alphabet_size, "
                "weights_ensemble, contact_matrix, biases, "
                "contact_scales, penalty_scales, seeds, "
                "epsilon, n_steps)"
            ),
            "gold_call": (
                "_oracle_optimize_protein_candidate("
                "initial_sequence, alphabet_size, "
                "weights_ensemble, contact_matrix, biases, "
                "contact_scales, penalty_scales, seeds, "
                "epsilon, n_steps)"
            ),
        },
        {
            "setup": """import numpy as np
initial_sequence = np.array([0, 1], dtype=int)
alphabet_size = 2
weights_ensemble = np.zeros((1, 2, 2), dtype=float)
contact_matrix = np.zeros((2, 2), dtype=float)
biases = np.array([0.0])
contact_scales = np.array([0.0])
penalty_scales = np.array([0.0])
seeds = np.array([-1], dtype=int)
epsilon = 0.2
n_steps = 1

def _model_error_code():
    try:
        optimize_protein_candidate(
            initial_sequence,
            alphabet_size,
            weights_ensemble,
            contact_matrix,
            biases,
            contact_scales,
            penalty_scales,
            seeds,
            epsilon,
            n_steps,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _oracle_error_code():
    try:
        _oracle_optimize_protein_candidate(
            initial_sequence,
            alphabet_size,
            weights_ensemble,
            contact_matrix,
            biases,
            contact_scales,
            penalty_scales,
            seeds,
            epsilon,
            n_steps,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "_model_error_code()",
            "gold_call": "_oracle_error_code()",
        },
    ]
