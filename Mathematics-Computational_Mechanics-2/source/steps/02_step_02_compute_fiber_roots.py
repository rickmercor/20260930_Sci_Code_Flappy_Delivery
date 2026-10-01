"""
Compute the two boundary parameters of an implicit ellipse on each fiber.



Apply the same multi-coordinate translation and rotation map used by the

field-evaluation step before intersecting the ellipse.

Roots are returned on the infinite supporting line; clipping to

`$0 <= h <= 1$` belongs to the contact-interval step.  Preserve distinct

crossings when the fiber coordinates are much larger than the ellipse:

forming the polynomial discriminant by subtracting nearly equal terms is

not sufficiently accurate.  A line with no real crossing receives two

``NaN`` values.

Apply the generalized motion map $c(q)=c+q^TD$ and $\phi(q)=\phi+q^T\omega$ before intersecting a fiber. For a line parameterized by $x(h)=s+hu$, transforming $s-c(q)$ and $u$ to ellipse-normalized coordinates reduces the boundary equation to a line-circle intersection. The closest point of the normalized supporting line separates the mean root location from the half-span between crossings. This geometric form avoids losing a small discriminant through cancellation when $s$ and $u$ are large compared with the ellipse axes.



Algebraically, substitution into the implicit field produces



$$

\alpha h^2+\beta h+\gamma=0,

$$



where $\alpha=u^TAu$, $\beta=2u^TA(s-c(q))$, and $\gamma=(s-c(q))^TA(s-c(q))-1$. Directly evaluating $\beta^2-4\alpha\gamma$ can subtract nearly equal large numbers. Detect that cancellation relative to the two terms and use the normalized closest-point margin in that regime. A positive discriminant yields two ordered crossings, a zero discriminant is tangential contact, and a negative discriminant has no real crossing. The roots describe the infinite line, so clipping is deferred.

Returns
-------
float64 np.ndarray of shape (n_fibers, 2), ascending roots or [NaN, NaN]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_fiber_roots(
    starts: np.ndarray,
    ends: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
    discriminant_tolerance: float = 1e-12,
) -> np.ndarray:
    """Return ordered ellipse crossings for each fiber-supporting line.

    Parameters
    ----------
    starts, ends : np.ndarray
        Matching finite arrays of shape ``(n_fibers, 2)`` in metres. Every
        fiber must have positive length.
    center, axes : np.ndarray
        Shape ``(2,)`` reference ellipse parameters; axes are positive.
    angle : float
        Reference rotation in radians.
    state : np.ndarray
        Finite generalized-coordinate vector with shape ``(p,)``.
    translation_directions : np.ndarray
        Shape ``(p, 2)`` with zero or unit rows.
    rotation_rates : np.ndarray
        Finite shape ``(p,)`` angular rates. Motion is applied using the same
        center and angle maps as in ``evaluate_ellipse_field``.
    discriminant_tolerance : float, optional
        Positive absolute tolerance used to recognize a repeated root.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_fibers, 2)`` containing ascending roots;
        a nonintersecting line has ``[NaN, NaN]``. Root separation must be
        retained for finite inputs whose coordinate and ellipse scales differ
        by up to eight orders of magnitude.

    Raises
    ------
    ValueError
        If a fiber has zero length or ``discriminant_tolerance`` is not
        positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_compute_fiber_roots(
    starts: np.ndarray,
    ends: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
    discriminant_tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference quadratic-line intersection."""
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    directions = np.asarray(translation_directions, dtype=np.float64)
    rotation_rates = np.asarray(rotation_rates, dtype=np.float64)
    if (
        starts.ndim != 2
        or starts.shape[0] < 1
        or starts.shape[1] != 2
        or ends.shape != starts.shape
    ):
        raise ValueError("starts and ends must have matching shape (n_fibers, 2)")
    if center.shape != (2,) or axes.shape != (2,):
        raise ValueError("center and axes must have shape (2,)")
    if state.ndim != 1 or state.size < 1:
        raise ValueError("state must have shape (p,) with p >= 1")
    if directions.shape != (state.size, 2) or rotation_rates.shape != state.shape:
        raise ValueError("motion arrays must have shapes (p, 2) and (p,)")
    numeric = (
        starts, ends, center, axes, state, directions, rotation_rates,
        [angle, discriminant_tolerance],
    )
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("all inputs must be finite")
    if np.any(axes <= 0.0) or discriminant_tolerance <= 0.0:
        raise ValueError("axes and discriminant_tolerance must be positive")
    segments = ends - starts
    if np.any(np.linalg.norm(segments, axis=1) <= 0.0):
        raise ValueError("fibers must have positive length")
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
    metric = rotation @ np.diag(axes ** -2) @ rotation.T
    offsets = starts - deformed_center
    quadratic = np.einsum("ni,ij,nj->n", segments, metric, segments)
    linear = 2.0 * np.einsum("ni,ij,nj->n", offsets, metric, segments)
    constant = np.einsum("ni,ij,nj->n", offsets, metric, offsets) - 1.0
    direct_discriminant = linear**2 - 4.0 * quadratic * constant

    normalized_offsets = (offsets @ rotation) / axes
    normalized_segments = (segments @ rotation) / axes
    closest_parameter = -np.einsum(
        "ni,ni->n", normalized_offsets, normalized_segments
    ) / quadratic
    closest_offsets = (
        normalized_offsets
        + closest_parameter[:, None] * normalized_segments
    )
    radial_margin = 1.0 - np.einsum(
        "ni,ni->n", closest_offsets, closest_offsets
    )
    geometric_discriminant = 4.0 * quadratic * radial_margin
    cancellation_scale = linear**2 + np.abs(4.0 * quadratic * constant)
    use_geometric = np.abs(direct_discriminant) <= (
        16.0 * np.finfo(np.float64).eps * cancellation_scale
    )
    discriminant = np.where(
        use_geometric, geometric_discriminant, direct_discriminant
    )
    roots = np.full((starts.shape[0], 2), np.nan, dtype=np.float64)
    crossing = discriminant >= -float(discriminant_tolerance)
    direct_crossing = crossing & ~use_geometric
    square_root = np.sqrt(np.maximum(discriminant[direct_crossing], 0.0))
    denominator = 2.0 * quadratic[direct_crossing]
    roots[direct_crossing, 0] = (
        -linear[direct_crossing] - square_root
    ) / denominator
    roots[direct_crossing, 1] = (
        -linear[direct_crossing] + square_root
    ) / denominator
    geometric_crossing = crossing & use_geometric
    half_span = np.sqrt(
        np.maximum(radial_margin[geometric_crossing], 0.0)
        / quadratic[geometric_crossing]
    )
    roots[geometric_crossing, 0] = (
        closest_parameter[geometric_crossing] - half_span
    )
    roots[geometric_crossing, 1] = (
        closest_parameter[geometric_crossing] + half_span
    )
    return roots

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return crossing, tolerance, scale-separation, and invalid cases."""
    return [
        {
            "setup": """
import numpy as np
starts = np.array([[-2.0, 0.0], [-2.0, 0.5]])
ends = np.array([[2.0, 0.0], [2.0, 0.5]])
center = np.zeros(2)
axes = np.array([1.0, 0.75])
angle = 0.0
state = np.zeros(1)
directions = np.zeros((1, 2))
rotation_rates = np.zeros(1)
""",
            "call": "compute_fiber_roots(starts, ends, center, axes, angle, state, directions, rotation_rates)",
            "gold_call": "_oracle_compute_fiber_roots(starts, ends, center, axes, angle, state, directions, rotation_rates)",
        },
        {
            "setup": """
import numpy as np
near_tangent = np.nextafter(1.0, np.inf)
starts = np.array([[-2.0, near_tangent], [-2.0, 1.2]])
ends = np.array([[2.0, near_tangent], [2.0, 1.2]])
center = np.zeros(2)
axes = np.ones(2)
angle = 0.0
state = np.zeros(1)
directions = np.zeros((1, 2))
rotation_rates = np.zeros(1)
def encode_array(value):
    array = np.asarray(value, dtype=float)
    return np.concatenate((
        np.nan_to_num(array, nan=0.0).ravel(),
        np.isnan(array).astype(float).ravel(),
    ))
def run_model():
    return encode_array(
        compute_fiber_roots(starts, ends, center, axes, angle, state, directions, rotation_rates)
    )
def run_gold():
    return encode_array(
        _oracle_compute_fiber_roots(
            starts, ends, center, axes, angle, state, directions, rotation_rates
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
    [1.0e8, 0.0], [2.0, 0.0], [-3.0, 0.0], [-2.0, 2.0]
])
ends = np.array([
    [0.0, 0.0], [3.0, 0.0], [-2.0, 0.0], [2.0, 2.0]
])
center = np.zeros(2)
axes = np.ones(2)
angle = 0.37
state = np.zeros(2)
directions = np.zeros((2, 2))
rotation_rates = np.array([0.4, -0.2])
def encode_array(value):
    array = np.asarray(value, dtype=float)
    return np.concatenate((
        np.nan_to_num(array, nan=0.0).ravel(),
        np.isnan(array).astype(float).ravel(),
    ))
def run_model():
    return encode_array(
        compute_fiber_roots(starts, ends, center, axes, angle, state, directions, rotation_rates)
    )
def run_gold():
    return encode_array(
        _oracle_compute_fiber_roots(
            starts, ends, center, axes, angle, state, directions, rotation_rates
        )
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np
starts = np.array([[0.0, 0.0]])
ends = np.array([[0.0, 0.0]])
center = np.zeros(2)
axes = np.ones(2)
angle = 0.0
state = np.zeros(1)
directions = np.zeros((1, 2))
rotation_rates = np.zeros(1)
def run_model():
    try:
        compute_fiber_roots(starts, ends, center, axes, angle, state, directions, rotation_rates)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_fiber_roots(starts, ends, center, axes, angle, state, directions, rotation_rates)
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
