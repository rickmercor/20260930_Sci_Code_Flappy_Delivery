"""
Differentiate moving fiber-boundary roots with respect to every state coordinate.



Translation and rotation may be coupled through a vector state. At each root,

apply implicit differentiation twice to obtain the full state gradient and

Hessian. The common transversality denominator vanishes at a tangent crossing,

where both derivatives are undefined and the input is rejected.

For a fixed fiber with endpoints $s$ and $e$, write $x(h)=s+h(e-s)$.

Each supplied finite root $h(q)$ lies on the moving ellipse boundary:



$$

g\!\left(x(h(q));q\right)=0.

$$



Use the same field and state-dependent center and orientation as the

field-evaluation step. The endpoints, reference geometry, translation

directions, and angular rates remain fixed while the state varies. Follow

each root locally on its existing branch, before clipping to the finite

fiber.



Return the first and second derivatives of $h$ with respect to the state,

in the layout specified by the signature. The Hessian is the total second

derivative of the boundary root: it must account for the boundary point's

motion as well as the explicit state dependence of the ellipse.



The transversality measure is the absolute derivative of $g(x(h);q)$ with

respect to $h$, holding $q$ fixed. Reject a finite root when that measure is

strictly below transversality_tolerance. Paired missing roots retain missing

derivatives in all channels. The signature defines their representation and

the remaining error behavior.

Returns
-------
float64 np.ndarray of shape (n_fibers, 2, p + 1, p): row 0 is the root gradient, rows 1..p are the root Hessian, with all-NaN noncrossing entries
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def differentiate_boundary_roots(
    starts: np.ndarray,
    ends: np.ndarray,
    roots: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
    transversality_tolerance: float = 1e-10,
) -> np.ndarray:
    """Return the state gradient and Hessian of every finite ellipse root.

    Parameters
    ----------
    starts, ends : np.ndarray
        Matching arrays of shape ``(n_fibers, 2)`` in metres.
    roots : np.ndarray
        Shape ``(n_fibers, 2)`` with finite ordered roots or paired ``NaN``.
    center, axes : np.ndarray
        Shape ``(2,)`` reference ellipse data.
    angle : float
        Reference rotation in radians.
    state : np.ndarray
        Generalized coordinates with shape ``(p,)``.
    translation_directions : np.ndarray
        Shape ``(p, 2)`` with zero or unit rows.
    rotation_rates : np.ndarray
        Finite angular rates with shape ``(p,)``.
    transversality_tolerance : float, optional
        Positive lower bound for the absolute implicit-root denominator.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_fibers, 2, p + 1, p)``. Slice
        ``result[:, :, 0, :]`` contains each root gradient and
        ``result[:, :, 1:, :]`` contains its symmetric Hessian. Noncrossing
        lines retain ``NaN`` in every derivative entry.

    Raises
    ------
    ValueError
        If a root row is unpaired or a finite root is non-transversal at the
        stated tolerance.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_differentiate_boundary_roots(
    starts: np.ndarray,
    ends: np.ndarray,
    roots: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
    transversality_tolerance: float = 1e-10,
) -> np.ndarray:
    """Reference second-order implicit differentiation of boundary roots."""
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    roots = np.asarray(roots, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    directions = np.asarray(translation_directions, dtype=np.float64)
    rotation_rates = np.asarray(rotation_rates, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[0] < 1 or starts.shape[1] != 2:
        raise ValueError("starts must have shape (n_fibers, 2)")
    if ends.shape != starts.shape or roots.shape != starts.shape:
        raise ValueError("ends and roots must match starts")
    if center.shape != (2,) or axes.shape != (2,):
        raise ValueError("center and axes must have shape (2,)")
    if state.ndim != 1 or state.size < 1:
        raise ValueError("state must have shape (p,) with p >= 1")
    if directions.shape != (state.size, 2) or rotation_rates.shape != state.shape:
        raise ValueError("motion arrays must have shapes (p, 2) and (p,)")
    numeric = (
        starts,
        ends,
        center,
        axes,
        state,
        directions,
        rotation_rates,
        [angle, transversality_tolerance],
    )
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("finite geometry and parameters are required")
    if np.any(axes <= 0.0) or transversality_tolerance <= 0.0:
        raise ValueError("axes and transversality_tolerance must be positive")
    segments = ends - starts
    if np.any(np.linalg.norm(segments, axis=1) <= 0.0):
        raise ValueError("fibers must have positive length")
    paired_nan = np.isnan(roots[:, 0]) & np.isnan(roots[:, 1])
    paired_finite = np.isfinite(roots[:, 0]) & np.isfinite(roots[:, 1])
    if not np.all(paired_nan | paired_finite):
        raise ValueError("root rows must be paired finite values or paired NaNs")
    direction_norms = np.linalg.norm(directions, axis=1)
    valid_directions = (direction_norms <= 1e-14) | np.isclose(
        direction_norms, 1.0, rtol=0.0, atol=1e-10
    )
    if not np.all(valid_directions):
        raise ValueError("translation directions must be zero or unit length")

    deformed_angle = float(angle + state @ rotation_rates)
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
    deformed_center = center + state @ directions
    sensitivities = np.full(
        roots.shape + (state.size + 1, state.size), np.nan, dtype=np.float64
    )
    for column in range(2):
        finite = np.isfinite(roots[:, column])
        boundary = starts[finite] + roots[finite, column, None] * segments[finite]
        offset = boundary - deformed_center
        gradient = 2.0 * offset @ metric
        angle_derivative = np.einsum(
            "ni,ij,nj->n", offset, metric_rate, offset
        )
        state_derivative = (
            -(gradient @ directions.T)
            + angle_derivative[:, None] * rotation_rates[None, :]
        )
        denominator = np.einsum("ni,ni->n", gradient, segments[finite])
        if np.any(np.abs(denominator) < transversality_tolerance):
            raise ValueError("boundary root is not transversal")
        root_gradient = -state_derivative / denominator[:, None]
        sensitivities[finite, column, 0, :] = root_gradient

        g_hh = 2.0 * np.einsum(
            "ni,ij,nj->n", segments[finite], metric, segments[finite]
        )
        q_metric_rate_x = np.einsum(
            "ni,ij,nj->n", segments[finite], metric_rate, offset
        )
        q_metric_directions = segments[finite] @ metric @ directions.T
        g_hi = 2.0 * (
            q_metric_rate_x[:, None] * rotation_rates[None, :]
            - q_metric_directions
        )
        direction_metric_direction = 2.0 * directions @ metric @ directions.T
        direction_metric_rate_offset = directions @ metric_rate @ offset.T
        angle_second_derivative = np.einsum(
            "ni,ij,nj->n", offset, metric_second, offset
        )
        partial_hessian = np.empty(
            (offset.shape[0], state.size, state.size), dtype=np.float64
        )
        for row in range(state.size):
            for state_column in range(state.size):
                partial_hessian[:, row, state_column] = (
                    direction_metric_direction[row, state_column]
                    - 2.0
                    * rotation_rates[state_column]
                    * direction_metric_rate_offset[row]
                    - 2.0
                    * rotation_rates[row]
                    * direction_metric_rate_offset[state_column]
                    + rotation_rates[row]
                    * rotation_rates[state_column]
                    * angle_second_derivative
                )
        root_hessian = np.empty_like(partial_hessian)
        for row in range(state.size):
            for state_column in range(state.size):
                numerator = (
                    partial_hessian[:, row, state_column]
                    + g_hi[:, row] * root_gradient[:, state_column]
                    + g_hi[:, state_column] * root_gradient[:, row]
                    + g_hh
                    * root_gradient[:, row]
                    * root_gradient[:, state_column]
                )
                root_hessian[:, row, state_column] = -numerator / denominator
        sensitivities[finite, column, 1:, :] = root_hessian
    return sensitivities

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return multi-state Hessian, fixed, coupled-motion, and tangent cases."""
    return [
        {
            "setup": """
import numpy as np
starts = np.array([[-2.0, 0.0], [-2.0, 0.5]])
ends = np.array([[2.0, 0.0], [2.0, 0.5]])
roots = np.array([[0.25, 0.75], [0.28349364905389035, 0.7165063509461097]])
center = np.zeros(2)
axes = np.ones(2)
angle = 0.0
state = np.zeros(2)
directions = np.array([[1.0, 0.0], [0.0, 1.0]])
rotation_rates = np.zeros(2)
""",
            "call": "differentiate_boundary_roots(starts, ends, roots, center, axes, angle, state, directions, rotation_rates)",
            "gold_call": "_oracle_differentiate_boundary_roots(starts, ends, roots, center, axes, angle, state, directions, rotation_rates)",
        },
        {
            "setup": """
import numpy as np
starts = np.array([[-2.0, 0.0], [-2.0, 2.0]])
ends = np.array([[2.0, 0.0], [2.0, 2.0]])
roots = np.array([[0.25, 0.75], [np.nan, np.nan]])
center = np.zeros(2)
axes = np.ones(2)
angle = 0.0
state = np.array([0.2, -0.3])
directions = np.zeros((2, 2))
rotation_rates = np.zeros(2)
def encode_array(value):
    array = np.asarray(value, dtype=float)
    return np.concatenate((
        np.nan_to_num(array, nan=0.0).ravel(),
        np.isnan(array).astype(float).ravel(),
    ))
def run_model():
    return encode_array(
        differentiate_boundary_roots(
            starts, ends, roots, center, axes, angle, state, directions, rotation_rates
        )
    )
def run_gold():
    return encode_array(
        _oracle_differentiate_boundary_roots(
            starts, ends, roots, center, axes, angle, state, directions, rotation_rates
        )
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np
starts = np.array([
    [-1.30, -0.70], [-1.00, 0.40], [1.60, -0.20], [-1.40, 1.30]
])
ends = np.array([
    [1.20, 0.90], [1.40, -0.50], [2.40, 0.30], [1.10, 1.10]
])
roots = np.array([
    [0.29641573886704015, 0.72915523353279610],
    [0.30760906023461443, 0.74018737370529260],
    [np.nan, np.nan],
    [np.nan, np.nan],
])
center = np.array([0.15, -0.20])
axes = np.array([0.90, 0.45])
angle = 0.47
state = np.array([0.13, 0.0, 0.0])
directions = np.array([[0.6, 0.8], [0.0, 0.0], [-0.8, 0.6]])
rotation_rates = np.array([0.0, 0.5, -0.2])
def encode_array(value):
    array = np.asarray(value, dtype=float)
    return np.concatenate((
        np.nan_to_num(array, nan=0.0).ravel(),
        np.isnan(array).astype(float).ravel(),
    ))
def run_model():
    return encode_array(
        differentiate_boundary_roots(
            starts, ends, roots, center, axes, angle, state, directions, rotation_rates
        )
    )
def run_gold():
    return encode_array(
        _oracle_differentiate_boundary_roots(
            starts, ends, roots, center, axes, angle, state, directions, rotation_rates
        )
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np
starts = np.array([[-2.0, 1.0]])
ends = np.array([[2.0, 1.0]])
near_tangent_root = np.nextafter(0.5, np.inf)
roots = np.array([[near_tangent_root, near_tangent_root]])
center = np.zeros(2)
axes = np.ones(2)
angle = 0.0
state = np.zeros(1)
directions = np.array([[1.0, 0.0]])
rotation_rates = np.zeros(1)
def run_model():
    try:
        differentiate_boundary_roots(starts, ends, roots, center, axes, angle, state, directions, rotation_rates)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_differentiate_boundary_roots(starts, ends, roots, center, axes, angle, state, directions, rotation_rates)
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
