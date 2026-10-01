"""
Evaluate a parametrically translated and rotated implicit ellipse.



The state may contain several generalized coordinates. Each coordinate has

its own center-translation direction and angular rate, so the returned field

table contains the spatial gradient, complete state gradient, and the packed

upper triangle of the state Hessian.

An ellipse is specified by its reference center $c$, positive semiaxes $a$ and

$b$, and counterclockwise reference angle $\phi$. A generalized state $q$

changes the center to $c+D^{\mathsf T}q$ and the angle to

$\phi+q^{\mathsf T}\omega$. Here $D$ has one translation direction per row;

$D$ and $\omega$ are constant with respect to $q$.



Let $R(\alpha)$ be the standard counterclockwise two-dimensional rotation matrix.

The dimensionless implicit field is



$$

g(x;q)=

\left\|

\operatorname{diag}\!\left(a^{-1},b^{-1}\right)

R\!\left(\phi+q^{\mathsf T}\omega\right)^{\mathsf T}

\left(x-c-D^{\mathsf T}q\right)

\right\|_2^2-1.

$$



The nonpositive sublevel set is the ellipse interior. Evaluate this field,

its gradient with respect to the spatial point, its gradient with respect to

the generalized state, and its state Hessian. Spatial points are held fixed

when taking state derivatives; reference geometry and motion coefficients

are held fixed for every derivative. These are partial derivatives of the

field, not derivatives along a moving boundary.



Both translation and orientation depend on the state. Mixed derivatives must

therefore reflect that dependence. The signature specifies the column order

and upper-triangle packing; it also specifies the invalid-input behavior.

Returns
-------
float64 np.ndarray of shape (n_points, 3 + p + p*(p+1)//2), containing [g, dg_dx, dg_dy, dg_dstate..., packed upper-triangular d2g_dstate2...]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_ellipse_field(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
) -> np.ndarray:
    """Evaluate an implicit ellipse and its first- and second-order derivatives.

    Parameters
    ----------
    points : np.ndarray
        Finite array of shape ``(n_points, 2)`` in metres.
    center : np.ndarray
        Undeformed ellipse center with shape ``(2,)`` in metres.
    axes : np.ndarray
        Strictly positive semiaxes ``(a, b)`` with shape ``(2,)`` in metres.
    angle : float
        Counterclockwise reference rotation in radians.
    state : np.ndarray
        Finite generalized-coordinate vector with shape ``(p,)``.
    translation_directions : np.ndarray
        Shape ``(p, 2)``. Each row is either zero or unit length and maps one
        state coordinate to center translation.
    rotation_rates : np.ndarray
        Finite shape ``(p,)`` rates mapping the state to additional rotation.
        The deformed center and angle are ``center + state @ directions`` and
        ``angle + state @ rotation_rates``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_points, 3 + p + p*(p+1)//2)``. Columns
        contain ``g``, the two spatial-gradient entries, the ``p`` state-
        gradient entries, then the upper triangle of the symmetric state
        Hessian in NumPy ``triu_indices(p)`` order.

    Raises
    ------
    ValueError
        If an array has an incompatible shape, any input is non-finite, a
        semiaxis is non-positive, or a translation-direction row is neither
        zero nor unit length.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_evaluate_ellipse_field(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
) -> np.ndarray:
    """Reference evaluation of the quadratic field and derivatives."""
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    directions = np.asarray(translation_directions, dtype=np.float64)
    rotation_rates = np.asarray(rotation_rates, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if center.shape != (2,) or axes.shape != (2,):
        raise ValueError("center and axes must have shape (2,)")
    if state.ndim != 1 or state.size < 1:
        raise ValueError("state must have shape (p,) with p >= 1")
    if directions.shape != (state.size, 2) or rotation_rates.shape != state.shape:
        raise ValueError("motion arrays must have shapes (p, 2) and (p,)")
    if not all(
        np.all(np.isfinite(value))
        for value in (points, center, axes, state, directions, rotation_rates, [angle])
    ):
        raise ValueError("all inputs must be finite")
    if np.any(axes <= 0.0):
        raise ValueError("ellipse semiaxes must be positive")
    direction_norms = np.linalg.norm(directions, axis=1)
    valid_directions = (direction_norms <= 1e-14) | np.isclose(
        direction_norms, 1.0, rtol=0.0, atol=1e-10
    )
    if not np.all(valid_directions):
        raise ValueError("translation directions must be zero or unit length")

    deformed_angle = float(angle + state @ rotation_rates)
    deformed_center = center + state @ directions
    cosine = np.cos(deformed_angle)
    sine = np.sin(deformed_angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotation_rate = np.array([[-sine, -cosine], [cosine, -sine]])
    diagonal = np.diag(axes ** -2)
    metric = rotation @ diagonal @ rotation.T
    metric_rate = (
        rotation_rate @ diagonal @ rotation.T
        + rotation @ diagonal @ rotation_rate.T
    )
    metric_second = 2.0 * (
        rotation_rate @ diagonal @ rotation_rate.T - metric
    )
    offset = points - deformed_center
    gradient = 2.0 * offset @ metric
    field = np.einsum("ni,ij,nj->n", offset, metric, offset) - 1.0
    angle_derivative = np.einsum(
        "ni,ij,nj->n", offset, metric_rate, offset
    )
    state_derivative = (
        -(gradient @ directions.T)
        + angle_derivative[:, None] * rotation_rates[None, :]
    )
    direction_metric_direction = 2.0 * directions @ metric @ directions.T
    direction_metric_rate_offset = directions @ metric_rate @ offset.T
    angle_second_derivative = np.einsum(
        "ni,ij,nj->n", offset, metric_second, offset
    )
    state_hessian = np.empty(
        (points.shape[0], state.size, state.size), dtype=np.float64
    )
    for row in range(state.size):
        for column in range(state.size):
            state_hessian[:, row, column] = (
                direction_metric_direction[row, column]
                - 2.0
                * rotation_rates[column]
                * direction_metric_rate_offset[row]
                - 2.0
                * rotation_rates[row]
                * direction_metric_rate_offset[column]
                + rotation_rates[row]
                * rotation_rates[column]
                * angle_second_derivative
            )
    upper = np.triu_indices(state.size)
    packed_hessian = state_hessian[:, upper[0], upper[1]]
    return np.column_stack(
        [field, gradient, state_derivative, packed_hessian]
    ).astype(
        np.float64, copy=False
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return coupled-Hessian, isotropic, mixed-motion, and invalid cases."""
    return [
        {
            "setup": """
import numpy as np
points = np.array([[0.2, -0.1], [0.8, 0.4], [-0.3, 0.6]])
center = np.array([-0.1, 0.05])
axes = np.array([0.9, 0.45])
angle = 0.31
state = np.array([-0.08, 0.12])
directions = np.array([[1.0, 0.0], [0.0, 0.0]])
rotation_rates = np.array([0.0, 1.0])
""",
            "call": "evaluate_ellipse_field(points, center, axes, angle, state, directions, rotation_rates)",
            "gold_call": "_oracle_evaluate_ellipse_field(points, center, axes, angle, state, directions, rotation_rates)",
        },
        {
            "setup": """
import numpy as np
points = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
center = np.zeros(2)
axes = np.ones(2)
angle = 0.0
state = np.zeros(1)
directions = np.zeros((1, 2))
rotation_rates = np.zeros(1)
""",
            "call": "evaluate_ellipse_field(points, center, axes, angle, state, directions, rotation_rates)",
            "gold_call": "_oracle_evaluate_ellipse_field(points, center, axes, angle, state, directions, rotation_rates)",
        },
        {
            "setup": """
import numpy as np
points = np.array([
    [-0.75, 0.90], [0.15, -0.35], [1.20, 0.45], [-0.40, -1.10]
])
center = np.array([0.18, -0.27])
axes = np.array([1.15, 0.38])
angle = -0.73
state = np.array([0.41, -0.17, 0.09])
directions = np.array([[0.6, 0.8], [0.0, 0.0], [-0.8, 0.6]])
rotation_rates = np.array([0.0, 0.7, -0.35])
""",
            "call": "evaluate_ellipse_field(points, center, axes, angle, state, directions, rotation_rates)",
            "gold_call": "_oracle_evaluate_ellipse_field(points, center, axes, angle, state, directions, rotation_rates)",
        },
        {
            "setup": """
import numpy as np
points = np.array([[0.0, 0.0]])
center = np.zeros(2)
axes = np.array([1.0, 0.0])
angle = 0.0
state = np.zeros(1)
directions = np.zeros((1, 2))
rotation_rates = np.zeros(1)
def run_model():
    try:
        evaluate_ellipse_field(points, center, axes, angle, state, directions, rotation_rates)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_ellipse_field(points, center, axes, angle, state, directions, rotation_rates)
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
