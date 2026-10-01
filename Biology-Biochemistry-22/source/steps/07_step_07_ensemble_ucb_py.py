"""
Evaluate every accepted discrete protein candidate using the complete surrogate ensemble and calculate its upper-confidence-bound score from the ensemble mean and population standard deviation.

An ensemble of surrogate models provides both a predicted protein-fitness value and an estimate of predictive uncertainty. For a candidate $\bar{q}$ evaluated by $M$ surrogate members, first calculate the ensemble mean:

$$
\mu(\bar{q})=\frac{1}{M}\sum_{j=1}^{M}f_j(\bar{q}).
$$

Calculate the population standard deviation using

$$
\sigma(\bar{q})=
\sqrt{\frac{1}{M}\sum_{j=1}^{M}
\left(f_j(\bar{q})-\mu(\bar{q})\right)^2}.
$$

The upper-confidence-bound score is

$$
\operatorname{UCB}(\bar{q})=
\mu(\bar{q})+\sigma(\bar{q}).
$$

The mean rewards candidates with high predicted fitness, while the population standard deviation rewards candidates located in uncertain regions. The use of the population standard deviation corresponds to NumPy’s `std` calculation with $ddof=0$.

Returns
-------
np.ndarray, a float64 vector of UCB scores with shape (number_of_candidates,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from importlib import import_module

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
    """Calculate one UCB score per accepted candidate.

    Parameters
    ----------
    candidates : np.ndarray
        Accepted one-hot candidates of shape ``(K, L, A)``.
    q_init : np.ndarray
        Initial one-hot state of shape ``(L, A)``.
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

    Raises
    ------
    ValueError
        If the candidate or ensemble shapes are invalid, the ensemble
        is empty, parameter-vector lengths do not match, or a supplied
        surrogate parameter is invalid.

    Returns
    -------
    ucb_scores : np.ndarray
        Float64 vector of shape ``(K,)`` containing the UCB
        score of each accepted candidate.
    """
    return ucb_scores

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ensemble_ucb(
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
            score, _ = _oracle_surrogate_score_gradient(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, single-member, and invalid-empty tests."""

    return [
        {
            "setup": """import numpy as np

q_init = np.eye(3, dtype=float)

candidates = np.array([
    np.eye(3)[[1, 1, 1]],
    np.eye(3)[[1, 0, 0]],
])

weights_ensemble = np.array([
    [
        [1.2, -0.3, 0.5],
        [0.1, 1.0, -0.4],
        [-0.2, 0.3, 0.9],
    ],
    [
        [0.8, 0.4, -0.2],
        [-0.3, 1.1, 0.2],
        [0.5, -0.4, 0.7],
    ],
    [
        [1.0, -0.2, 0.3],
        [0.2, 0.7, 0.6],
        [-0.1, 0.8, 0.5],
    ],
])

contact_matrix = np.array([
    [0.2, -0.1, 0.3],
    [-0.1, 0.4, 0.0],
    [0.3, 0.0, 0.5],
])

biases = np.array([-0.5, -0.3, -0.4])
contact_scales = np.array([0.7, 0.5, 0.6])
penalty_scales = np.array([0.25, 0.30, 0.20])
""",
            "call": (
                "ensemble_ucb("
                "candidates, q_init, weights_ensemble, contact_matrix, "
                "biases, contact_scales, penalty_scales).tolist()"
            ),
            "gold_call": (
                "_oracle_ensemble_ucb("
                "candidates, q_init, weights_ensemble, contact_matrix, "
                "biases, contact_scales, penalty_scales).tolist()"
            ),
        },
        {
            "setup": """import numpy as np

q_init = np.eye(2, dtype=float)
candidates = q_init[None, :, :]
weights_ensemble = np.zeros((1, 2, 2), dtype=float)
contact_matrix = np.zeros((2, 2), dtype=float)
biases = np.array([0.25])
contact_scales = np.array([0.0])
penalty_scales = np.array([0.0])
""",
            "call": (
                "ensemble_ucb("
                "candidates, q_init, weights_ensemble, contact_matrix, "
                "biases, contact_scales, penalty_scales).tolist()"
            ),
            "gold_call": (
                "_oracle_ensemble_ucb("
                "candidates, q_init, weights_ensemble, contact_matrix, "
                "biases, contact_scales, penalty_scales).tolist()"
            ),
        },
        {
            "setup": """import numpy as np

q_init = np.eye(2, dtype=float)
candidates = np.empty((0, 2, 2), dtype=float)
weights_ensemble = np.zeros((1, 2, 2), dtype=float)
contact_matrix = np.zeros((2, 2), dtype=float)
biases = np.array([0.0])
contact_scales = np.array([0.0])
penalty_scales = np.array([0.0])

def _model_error_code():
    try:
        ensemble_ucb(
            candidates,
            q_init,
            weights_ensemble,
            contact_matrix,
            biases,
            contact_scales,
            penalty_scales,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _oracle_error_code():
    try:
        _oracle_ensemble_ucb(
            candidates,
            q_init,
            weights_ensemble,
            contact_matrix,
            biases,
            contact_scales,
            penalty_scales,
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
