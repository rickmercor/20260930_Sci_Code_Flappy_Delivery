"""
Convert a surrogate protein-fitness score into the probability-derived potential energy used by Hamiltonian dynamics, and calculate the corresponding analytic potential gradient.

The Hamiltonian acquisition procedure must assign lower potential energy to protein states with higher predicted fitness. It first converts the surrogate score $f$ into a probability using the sigmoid function:

$$
P(f)=\sigma(f)=\frac{1}{1+\exp(-f)}.
$$

The potential energy is then defined as

$$
U(q)=-\log\bigl(\sigma(f(q))\bigr).
$$

Applying the chain rule gives the potential gradient:

$$
\nabla_q U(q)=\bigl(\sigma(f(q))-1\bigr)\nabla_q f(q).
$$

Because $\sigma(f)-1$ is negative, movement in the direction of increasing predicted fitness generally decreases the potential energy. A numerically stable implementation should avoid directly evaluating expressions that may overflow when the surrogate score has a very large magnitude.

Returns
-------
tuple[float, np.ndarray], the scalar potential energy and its float64 gradient array with the same shape as score_gradient
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def potential_energy_gradient(
    score: float,
    score_gradient: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Calculate the probability-derived potential and gradient.
 
    Parameters
    ----------
    score : float
        Finite scalar surrogate fitness score.
    score_gradient : np.ndarray
        Non-empty finite numerical gradient of the score with respect to
        position.
 
    Returns
    -------
    result : tuple[float, np.ndarray]
        Scalar potential energy and float64 potential-gradient array.
 
    Raises
    ------
    ValueError
        If ``score`` is not a finite scalar, or if ``score_gradient`` is
        empty or contains a non-finite value.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_potential_energy_gradient(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, boundary, and numerical-stability test cases."""

    return [
        {
            "setup": """import numpy as np

score = 2.81
score_gradient = np.array([
    [1.0, -0.5],
    [0.25, 2.0],
], dtype=float)
""",
            "call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "potential_energy_gradient(score, score_gradient))"
            ),
            "gold_call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "_oracle_potential_energy_gradient("
                "score, score_gradient))"
            ),
        },
        {
            "setup": """import numpy as np

score = 0.0
score_gradient = np.zeros((1, 1), dtype=float)
""",
            "call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "potential_energy_gradient(score, score_gradient))"
            ),
            "gold_call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "_oracle_potential_energy_gradient("
                "score, score_gradient))"
            ),
        },
        {
            "setup": """import numpy as np

score = 1000.0
score_gradient = np.array([1.0, -1.0], dtype=float)
""",
            "call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "potential_energy_gradient(score, score_gradient))"
            ),
            "gold_call": (
                "(lambda r: [float(r[0]), r[1].tolist()])("
                "_oracle_potential_energy_gradient("
                "score, score_gradient))"
            ),
        },
    ]
