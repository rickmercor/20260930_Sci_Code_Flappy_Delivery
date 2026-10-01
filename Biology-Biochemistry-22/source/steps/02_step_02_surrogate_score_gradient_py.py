"""
Evaluate one fixed structure-aware surrogate model and its analytic gradient. The surrogate combines position-specific residue scores, an interaction between the first and last sequence positions, and a quadratic penalty for movement away from the initial protein sequence.

Each trained surrogate combines a position-specific linear score, a contact interaction between the first and last sequence positions, and a quadratic penalty relative to $q^{init}$. Its gradient contains the position-specific weight matrix, the penalty contribution from displacement away from $q^{init}$, and contact contributions in the first and last rows. Returning both the scalar score and its analytic gradient supplies the information needed to construct the potential gradient used by Hamiltonian dynamics.

Returns
-------
tuple[float, np.ndarray], the scalar surrogate score and its float64 gradient matrix of shape (L, A)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Evaluate one surrogate score and its gradient.

    Parameters
    ----------
    q : np.ndarray
        Continuous sequence state of shape ``(L, A)``, with ``L >= 2``.
    q_init : np.ndarray
        Initial one-hot state with the same shape as ``q``.
    weights : np.ndarray
        Position-specific weights with the same shape as ``q``.
    contact_matrix : np.ndarray
        Residue-contact matrix of shape ``(A, A)``.
    bias : float
        Scalar surrogate intercept.
    contact_scale : float
        Multiplier for the first-to-last position interaction.
    penalty_scale : float
        Non-negative quadratic penalty multiplier.

    Raises
    ------
    ValueError
        If the array shapes are incompatible, an input is non-finite,
        or ``penalty_scale`` is negative.

    Returns
    -------
    result : tuple[float, np.ndarray]
        Surrogate score and float64 gradient of shape ``(L, A)``.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_surrogate_score_gradient(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, boundary, and invalid-input test cases."""

    return [
        {
            "setup": """import numpy as np

q = np.eye(3, dtype=float)
q_init = q.copy()
weights = np.array([
    [1.2, -0.3, 0.5],
    [0.1, 1.0, -0.4],
    [-0.2, 0.3, 0.9],
])
contact_matrix = np.array([
    [0.2, -0.1, 0.3],
    [-0.1, 0.4, 0.0],
    [0.3, 0.0, 0.5],
])
bias = -0.5
contact_scale = 0.7
penalty_scale = 0.25
""",
            "call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "surrogate_score_gradient("
                "q, q_init, weights, contact_matrix, bias, "
                "contact_scale, penalty_scale))"
            ),
            "gold_call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "_oracle_surrogate_score_gradient("
                "q, q_init, weights, contact_matrix, bias, "
                "contact_scale, penalty_scale))"
            ),
        },
        {
            "setup": """import numpy as np

q = np.zeros((2, 1), dtype=float)
q_init = np.zeros((2, 1), dtype=float)
weights = np.zeros((2, 1), dtype=float)
contact_matrix = np.zeros((1, 1), dtype=float)
bias = 0.0
contact_scale = 0.0
penalty_scale = 0.0
""",
            "call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "surrogate_score_gradient("
                "q, q_init, weights, contact_matrix, bias, "
                "contact_scale, penalty_scale))"
            ),
            "gold_call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "_oracle_surrogate_score_gradient("
                "q, q_init, weights, contact_matrix, bias, "
                "contact_scale, penalty_scale))"
            ),
        },
        {
            "setup": """import numpy as np

q = np.zeros((3, 2), dtype=float)
q_init = np.zeros((3, 2), dtype=float)
weights = np.zeros((2, 2), dtype=float)
contact_matrix = np.eye(2, dtype=float)

def _model_error_code():
    try:
        surrogate_score_gradient(
            q,
            q_init,
            weights,
            contact_matrix,
            0.0,
            1.0,
            0.1,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _oracle_error_code():
    try:
        _oracle_surrogate_score_gradient(
            q,
            q_init,
            weights,
            contact_matrix,
            0.0,
            1.0,
            0.1,
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
