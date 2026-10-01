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


def encode_sequence(
    sequence: np.ndarray,
    alphabet_size: int,
) -> np.ndarray:
    """Return the deterministic one-hot encoding."""

    sequence_array = np.asarray(sequence)

    if (
        sequence_array.ndim != 1
        or sequence_array.size == 0
    ):
        raise ValueError(
            "sequence must be a non-empty one-dimensional array"
        )

    if (
        isinstance(alphabet_size, (bool, np.bool_))
        or not isinstance(
            alphabet_size,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "alphabet_size must be a positive integer"
        )

    if int(alphabet_size) < 1:
        raise ValueError(
            "alphabet_size must be a positive integer"
        )

    if not np.issubdtype(
        sequence_array.dtype,
        np.integer,
    ):
        raise ValueError(
            "sequence entries must be integers"
        )

    sequence_int = sequence_array.astype(
        np.int64,
        copy=False,
    )

    if (
        np.any(sequence_int < 0)
        or np.any(
            sequence_int >= int(alphabet_size)
        )
    ):
        raise ValueError(
            "sequence entries must lie in [0, alphabet_size)"
        )

    return np.eye(
        int(alphabet_size),
        dtype=np.float64,
    )[sequence_int]

import numpy as np


def surrogate_score_gradient(
    q: np.ndarray,
    q_init: np.ndarray,
    weights: np.ndarray,
    contact_matrix: np.ndarray,
    bias: float,
    contact_scale: float,
    penalty_scale: float,
) -> tuple[float, np.ndarray]:
    """Return the surrogate score and analytic gradient."""

    q_array = np.asarray(q, dtype=np.float64)
    q_init_array = np.asarray(q_init, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    contact_array = np.asarray(
        contact_matrix,
        dtype=np.float64,
    )

    if (
        q_array.ndim != 2
        or q_array.shape[0] < 2
        or q_array.shape[1] < 1
    ):
        raise ValueError(
            "q must have shape (L, A) with L >= 2 and A >= 1"
        )

    if (
        q_init_array.shape != q_array.shape
        or weights_array.shape != q_array.shape
    ):
        raise ValueError(
            "q_init and weights must have the same shape as q"
        )

    alphabet_size = q_array.shape[1]

    if contact_array.shape != (
        alphabet_size,
        alphabet_size,
    ):
        raise ValueError(
            "contact_matrix must have shape (A, A)"
        )

    if not all(
        np.all(np.isfinite(array))
        for array in (
            q_array,
            q_init_array,
            weights_array,
            contact_array,
        )
    ):
        raise ValueError(
            "all array inputs must contain finite values"
        )

    scalar_values = (
        bias,
        contact_scale,
        penalty_scale,
    )

    if not all(
        np.isscalar(value) and np.isfinite(value)
        for value in scalar_values
    ):
        raise ValueError(
            "bias, contact_scale, and penalty_scale must be finite"
        )

    if float(penalty_scale) < 0.0:
        raise ValueError(
            "penalty_scale must be non-negative"
        )

    displacement = q_array - q_init_array

    contact_term = (
        q_array[0]
        @ contact_array
        @ q_array[-1]
    )

    score = (
        float(bias)
        + np.sum(weights_array * q_array)
        + float(contact_scale) * contact_term
        - 0.5
        * float(penalty_scale)
        * np.sum(displacement**2)
    )

    gradient = (
        weights_array
        - float(penalty_scale) * displacement
    )

    gradient = gradient.astype(
        np.float64,
        copy=True,
    )

    gradient[0] += (
        float(contact_scale)
        * (contact_array @ q_array[-1])
    )

    gradient[-1] += (
        float(contact_scale)
        * (contact_array.T @ q_array[0])
    )

    return float(score), gradient

import numpy as np


def potential_energy_gradient(
    score: float,
    score_gradient: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return a numerically stable potential and gradient."""

    gradient_array = np.asarray(
        score_gradient,
        dtype=np.float64,
    )

    if (
        not np.isscalar(score)
        or not np.isfinite(score)
    ):
        raise ValueError(
            "score must be a finite scalar"
        )

    if (
        gradient_array.size == 0
        or not np.all(np.isfinite(gradient_array))
    ):
        raise ValueError(
            "score_gradient must be non-empty and finite"
        )

    score_float = float(score)

    potential = np.logaddexp(
        0.0,
        -score_float,
    )

    derivative_factor = -np.exp(
        -np.logaddexp(0.0, score_float)
    )

    potential_gradient = (
        derivative_factor * gradient_array
    )

    return (
        float(potential),
        potential_gradient.astype(
            np.float64,
            copy=False,
        ),
    )

import numpy as np


def _reflect_unit_interval(
    position: np.ndarray,
    half_momentum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeatedly reflect finite coordinates into the closed unit interval."""

    reflected_position = position.copy()
    reflected_momentum = half_momentum.copy()
    reflection_passes = 0

    while np.any(
        (reflected_position < 0.0)
        | (reflected_position > 1.0)
    ):
        above = reflected_position > 1.0
        below = reflected_position < 0.0

        reflected_position[above] = (
            2.0 - reflected_position[above]
        )
        reflected_momentum[above] *= -1.0

        reflected_position[below] *= -1.0
        reflected_momentum[below] *= -1.0

        reflection_passes += 1
        if reflection_passes > 100_000:
            raise ValueError(
                "virtual-barrier reflection did not converge"
            )

    return reflected_position, reflected_momentum


def reflective_leapfrog(
    q_init: np.ndarray,
    momentum: np.ndarray,
    weights: np.ndarray,
    contact_matrix: np.ndarray,
    bias: float,
    contact_scale: float,
    penalty_scale: float,
    epsilon: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic reflected leapfrog endpoint."""

    q_reference = np.asarray(q_init, dtype=np.float64)
    p_current = np.asarray(
        momentum,
        dtype=np.float64,
    ).copy()

    if q_reference.ndim != 2 or q_reference.shape[0] < 2:
        raise ValueError(
            "q_init must have shape (L, A) with L >= 2"
        )

    if p_current.shape != q_reference.shape:
        raise ValueError(
            "momentum must have the same shape as q_init"
        )

    if (
        not np.all(np.isfinite(q_reference))
        or not np.all(np.isfinite(p_current))
    ):
        raise ValueError(
            "q_init and momentum must contain finite values"
        )

    if np.any(
        (q_reference < 0.0)
        | (q_reference > 1.0)
    ):
        raise ValueError(
            "q_init must lie within [0, 1]"
        )

    if (
        not np.isscalar(epsilon)
        or not np.isfinite(epsilon)
        or float(epsilon) <= 0.0
    ):
        raise ValueError(
            "epsilon must be a positive finite scalar"
        )

    if (
        isinstance(n_steps, (bool, np.bool_))
        or not isinstance(n_steps, (int, np.integer))
    ):
        raise ValueError(
            "n_steps must be a positive integer"
        )

    if int(n_steps) < 1:
        raise ValueError(
            "n_steps must be a positive integer"
        )

    q_current = q_reference.copy()
    step_size = float(epsilon)

    for _ in range(int(n_steps)):
        score, score_gradient = (
            surrogate_score_gradient(
                q_current,
                q_reference,
                weights,
                contact_matrix,
                bias,
                contact_scale,
                penalty_scale,
            )
        )

        _, potential_gradient = (
            potential_energy_gradient(
                score,
                score_gradient,
            )
        )

        p_half = (
            p_current
            - 0.5 * step_size * potential_gradient
        )

        proposed_position = (
            q_current + step_size * p_half
        )

        q_current, p_half = _reflect_unit_interval(
            proposed_position,
            p_half,
        )

        score_new, score_gradient_new = (
            surrogate_score_gradient(
                q_current,
                q_reference,
                weights,
                contact_matrix,
                bias,
                contact_scale,
                penalty_scale,
            )
        )

        _, potential_gradient_new = (
            potential_energy_gradient(
                score_new,
                score_gradient_new,
            )
        )

        p_current = (
            p_half
            - 0.5 * step_size * potential_gradient_new
        )

    return (
        q_current.astype(np.float64),
        p_current.astype(np.float64),
    )

import numpy as np


def discretize_positions(
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sitewise argmax indices and one-hot positions."""

    q_array = np.asarray(
        q,
        dtype=np.float64,
    )

    if (
        q_array.ndim != 2
        or q_array.shape[0] < 1
        or q_array.shape[1] < 1
    ):
        raise ValueError(
            "q must be a non-empty two-dimensional array"
        )

    if not np.all(np.isfinite(q_array)):
        raise ValueError(
            "q must contain finite values"
        )

    residue_indices = np.argmax(
        q_array,
        axis=1,
    ).astype(np.int64)

    one_hot = np.eye(
        q_array.shape[1],
        dtype=np.float64,
    )[residue_indices]

    return residue_indices, one_hot

import numpy as np


def metropolis_acceptance(
    initial_potential: float,
    proposed_potential: float,
    initial_momentum: np.ndarray,
    final_momentum: np.ndarray,
    uniform_draw: float,
) -> tuple[float, int]:
    """Return the stable Metropolis probability and decision."""

    initial_momentum_array = np.asarray(
        initial_momentum,
        dtype=np.float64,
    )

    final_momentum_array = np.asarray(
        final_momentum,
        dtype=np.float64,
    )

    scalar_values = (
        initial_potential,
        proposed_potential,
        uniform_draw,
    )

    if not all(
        np.isscalar(value) and np.isfinite(value)
        for value in scalar_values
    ):
        raise ValueError(
            "potential energies and uniform_draw must be finite scalars"
        )

    if (
        initial_momentum_array.shape
        != final_momentum_array.shape
    ):
        raise ValueError(
            "initial_momentum and final_momentum must have matching shapes"
        )

    if (
        initial_momentum_array.size == 0
        or not np.all(
            np.isfinite(initial_momentum_array)
        )
    ):
        raise ValueError(
            "initial_momentum must be non-empty and finite"
        )

    if not np.all(
        np.isfinite(final_momentum_array)
    ):
        raise ValueError(
            "final_momentum must contain finite values"
        )

    if not 0.0 <= float(uniform_draw) < 1.0:
        raise ValueError(
            "uniform_draw must lie in [0, 1)"
        )

    initial_kinetic = (
        0.5
        * np.sum(initial_momentum_array**2)
    )

    final_kinetic = (
        0.5
        * np.sum(final_momentum_array**2)
    )

    log_ratio = (
        float(initial_potential)
        + initial_kinetic
        - float(proposed_potential)
        - final_kinetic
    )

    probability = (
        1.0
        if log_ratio >= 0.0
        else float(np.exp(log_ratio))
    )

    accepted = int(
        float(uniform_draw) < probability
    )

    return probability, accepted

import numpy as np


def ensemble_ucb(
    candidates: np.ndarray,
    q_init: np.ndarray,
    weights_ensemble: np.ndarray,
    contact_matrix: np.ndarray,
    biases: np.ndarray,
    contact_scales: np.ndarray,
    penalty_scales: np.ndarray,
) -> np.ndarray:
    """Return ensemble mean-plus-population-SD scores."""

    candidates_array = np.asarray(
        candidates,
        dtype=np.float64,
    )
    q_init_array = np.asarray(
        q_init,
        dtype=np.float64,
    )
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

    if (
        candidates_array.ndim != 3
        or candidates_array.shape[0] < 1
    ):
        raise ValueError(
            "candidates must have shape (K, L, A) with K >= 1"
        )

    if q_init_array.shape != candidates_array.shape[1:]:
        raise ValueError(
            "q_init must match each candidate shape"
        )

    if (
        weights_array.ndim != 3
        or weights_array.shape[1:] != q_init_array.shape
    ):
        raise ValueError(
            "weights_ensemble must have shape (M, L, A)"
        )

    model_count = weights_array.shape[0]

    if model_count < 1:
        raise ValueError(
            "the ensemble must contain at least one model"
        )

    if not all(
        array.shape == (model_count,)
        for array in (
            biases_array,
            contact_scales_array,
            penalty_scales_array,
        )
    ):
        raise ValueError(
            "ensemble parameter vectors must have shape (M,)"
        )

    score_matrix = np.empty(
        (candidates_array.shape[0], model_count),
        dtype=np.float64,
    )

    for candidate_index, candidate in enumerate(
        candidates_array
    ):
        for model_index in range(model_count):
            score, _ = surrogate_score_gradient(
                candidate,
                q_init_array,
                weights_array[model_index],
                contact_matrix,
                biases_array[model_index],
                contact_scales_array[model_index],
                penalty_scales_array[model_index],
            )

            score_matrix[
                candidate_index,
                model_index,
            ] = score

    return (
        score_matrix.mean(axis=1)
        + score_matrix.std(axis=1, ddof=0)
    )

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

    q_init = encode_sequence(
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
            ) = surrogate_score_gradient(
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
            ) = potential_energy_gradient(
                score_before,
                score_gradient_before,
            )

            (
                position_after,
                momentum_after,
            ) = reflective_leapfrog(
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
            ) = discretize_positions(
                position_after
            )

            (
                score_proposed,
                score_gradient_proposed,
            ) = surrogate_score_gradient(
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
            ) = potential_energy_gradient(
                score_proposed,
                score_gradient_proposed,
            )

            uniform_draw = float(rng.random())

            (
                _,
                accepted,
            ) = metropolis_acceptance(
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

    ucb_scores = ensemble_ucb(
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
SCICODE_GOLD_EOF
