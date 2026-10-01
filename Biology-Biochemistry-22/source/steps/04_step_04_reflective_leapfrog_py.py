"""
Evolve one continuously relaxed protein sequence through leapfrog Hamiltonian dynamics. Whenever a coordinate leaves the permitted interval, apply repeated virtual-barrier reflections and reverse the corresponding momentum component.

Hamiltonian dynamics uses the potential gradient to update momentum and then uses momentum to update the continuous protein-sequence state. With unit mass and leapfrog step size $\epsilon$, one step begins with the half-momentum update

$$
p_{t+1/2}=p_t-\frac{\epsilon}{2}\nabla U(q_t).
$$

The position is then updated using

$$
q_{t+1}=q_t+\epsilon p_{t+1/2}.
$$

The relaxed coordinates must remain between $0$ and $1$. If a coordinate exceeds $1$, reflect it using $q_i\leftarrow2-q_i$. If a coordinate falls below $0$, reflect it using $q_i\leftarrow-q_i$. In either case, reverse its corresponding momentum using $p_i\leftarrow-p_i$. Repeat these transformations until every coordinate satisfies $0\leq q_i\leq1$.

After reflection, complete the leapfrog step with

$$
p_{t+1}=p_{t+1/2}-\frac{\epsilon}{2}\nabla U(q_{t+1}).
$$

Unlike clipping, reflection preserves the method’s bouncing movement at the virtual barriers. Repeated reflection is necessary when a large update crosses the bounded interval more than once.

Returns
-------
tuple[np.ndarray, np.ndarray], the final float64 position and momentum matrices, each with the same shape as q_init
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from importlib import import_module

import numpy as np


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
    """Run a reflective leapfrog trajectory for one surrogate member.
 
    Parameters
    ----------
    q_init : np.ndarray
        Initial continuous state of shape ``(L, A)`` with ``L >= 2`` and
        all entries in ``[0, 1]``.
    momentum : np.ndarray
        Finite initial momentum with the same shape as ``q_init``.
    weights : np.ndarray
        Finite surrogate weight matrix with the same shape as ``q_init``.
    contact_matrix : np.ndarray
        Finite residue-contact matrix of shape ``(A, A)``.
    bias : float
        Finite surrogate intercept.
    contact_scale : float
        Finite contact-interaction multiplier.
    penalty_scale : float
        Finite non-negative penalty relative to ``q_init``.
    epsilon : float
        Positive finite leapfrog step size.
    n_steps : int
        Positive integer number of leapfrog steps.
 
    Returns
    -------
    result : tuple[np.ndarray, np.ndarray]
        Final float64 position and momentum matrices.
 
    Raises
    ------
    ValueError
        If an array has an invalid or incompatible shape; an input contains
        a non-finite value; ``q_init`` lies outside ``[0, 1]``;
        ``penalty_scale`` is negative; ``epsilon`` is not positive and
        finite; ``n_steps`` is not a positive integer; or reflective
        correction cannot return a coordinate to the unit interval.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_reflective_leapfrog(
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
            _oracle_surrogate_score_gradient(
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
            _oracle_potential_energy_gradient(
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
            _oracle_surrogate_score_gradient(
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
            _oracle_potential_energy_gradient(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, stationary-boundary, and multi-reflection tests."""

    return [
        {
            "setup": """import numpy as np

q_init = np.eye(3, dtype=float)
rng = np.random.default_rng(101)
momentum = rng.normal(size=(3, 3))
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
""",
            "call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "reflective_leapfrog("
                "q_init, momentum, weights, contact_matrix, "
                "-0.5, 0.7, 0.25, 0.6, 4))"
            ),
            "gold_call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "_oracle_reflective_leapfrog("
                "q_init, momentum, weights, contact_matrix, "
                "-0.5, 0.7, 0.25, 0.6, 4))"
            ),
        },
        {
            "setup": """import numpy as np

q_init = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])
momentum = np.zeros((2, 2), dtype=float)
weights = np.zeros((2, 2), dtype=float)
contact_matrix = np.zeros((2, 2), dtype=float)
""",
            "call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "reflective_leapfrog("
                "q_init, momentum, weights, contact_matrix, "
                "0.0, 0.0, 0.0, 0.5, 1))"
            ),
            "gold_call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "_oracle_reflective_leapfrog("
                "q_init, momentum, weights, contact_matrix, "
                "0.0, 0.0, 0.0, 0.5, 1))"
            ),
        },
        {
            "setup": """import numpy as np

q_init = np.array([
    [0.2, 0.8],
    [0.7, 0.3],
])
momentum = np.array([
    [8.0, -7.0],
    [6.0, -9.0],
])
weights = np.zeros((2, 2), dtype=float)
contact_matrix = np.zeros((2, 2), dtype=float)
""",
            "call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "reflective_leapfrog("
                "q_init, momentum, weights, contact_matrix, "
                "0.0, 0.0, 0.0, 0.75, 2))"
            ),
            "gold_call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "_oracle_reflective_leapfrog("
                "q_init, momentum, weights, contact_matrix, "
                "0.0, 0.0, 0.0, 0.75, 2))"
            ),
        },
    ]
