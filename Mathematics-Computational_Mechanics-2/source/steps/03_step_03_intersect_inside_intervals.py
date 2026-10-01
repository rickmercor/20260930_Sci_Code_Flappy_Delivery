"""
Intersect all body-interior intervals with the finite fiber.



The input stacks the ordered root pairs for two or more convex bodies.

Contact is their common intersection with ``[0, 1]``. Owner code 0 denotes

a fiber boundary and code `$j + 1$` denotes body index `$j$`.

Along a line crossing a convex body, its interior lies between the two ordered boundary roots. For $B\ge2$ bodies, common contact requires simultaneous membership in every body and in the sampled segment, so



$$

h_{\mathrm{in}}=\max(0,h_{1,-},\ldots,h_{B,-}),\qquad

h_{\mathrm{out}}=\min(1,h_{1,+},\ldots,h_{B,+}).

$$



Its length is positive only when $h_{\mathrm{out}}>h_{\mathrm{in}}$. Owner code 0 denotes the segment constraint and code $j$ denotes body $j$. Recording the unique owner of each active endpoint is essential because that owner selects a complete root-Jacobian row downstream. Equal active candidates within the prescribed tolerance are nondifferentiable switching events.

Returns
-------
float64 np.ndarray of shape (n_fibers, 4), interval bounds and numeric owner codes in 0..n_bodies, or an all-NaN row when no positive-length common contact interval exists
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def intersect_inside_intervals(
    root_sets: np.ndarray, tie_tolerance: float = 1e-12
) -> np.ndarray:
    """Return contact-interval bounds and active endpoint owners.

    Parameters
    ----------
    root_sets : np.ndarray
        Float array of shape ``(n_bodies, n_fibers, 2)`` for at least two
        bodies. Each row contains ascending roots or paired ``NaN`` values.
    tie_tolerance : float, optional
        Positive tolerance for detecting nondifferentiable owner ties.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_fibers, 4)`` with columns
        ``[h_lower, h_upper, lower_owner, upper_owner]``. Owner codes range
        from 0 through ``n_bodies``. A fiber with no positive-length common
        contact interval has four ``NaN`` values.

    Raises
    ------
    ValueError
        If fewer than two bodies are supplied, a root row is unpaired or
        descending, or an active endpoint has a nondifferentiable owner tie.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_intersect_inside_intervals(
    root_sets: np.ndarray, tie_tolerance: float = 1e-12
) -> np.ndarray:
    """Reference interval intersection with unique active owners."""
    roots = np.asarray(root_sets, dtype=np.float64)
    if roots.ndim != 3 or roots.shape[0] < 2 or roots.shape[1] < 1 or roots.shape[2] != 2:
        raise ValueError("root_sets must have shape (n_bodies, n_fibers, 2)")
    if not np.isfinite(tie_tolerance) or tie_tolerance <= 0.0:
        raise ValueError("tie_tolerance must be positive and finite")
    paired_nan = np.isnan(roots[..., 0]) & np.isnan(roots[..., 1])
    paired_finite = np.isfinite(roots[..., 0]) & np.isfinite(roots[..., 1])
    if not np.all(paired_nan | paired_finite):
        raise ValueError("each root row must be paired finite values or paired NaNs")
    if np.any(paired_finite & (roots[..., 0] > roots[..., 1])):
        raise ValueError("finite roots must be ascending")

    result = np.full((roots.shape[1], 4), np.nan, dtype=np.float64)
    for fiber in range(roots.shape[1]):
        if np.any(np.isnan(roots[:, fiber])):
            continue
        lower_candidates = np.concatenate(([0.0], roots[:, fiber, 0]))
        upper_candidates = np.concatenate(([1.0], roots[:, fiber, 1]))
        lower = float(np.max(lower_candidates))
        upper = float(np.min(upper_candidates))
        if upper <= lower + tie_tolerance:
            continue
        lower_matches = np.flatnonzero(
            np.abs(lower_candidates - lower) <= tie_tolerance
        )
        upper_matches = np.flatnonzero(
            np.abs(upper_candidates - upper) <= tie_tolerance
        )
        if lower_matches.size != 1 or upper_matches.size != 1:
            raise ValueError("active contact endpoint has a nondifferentiable tie")
        result[fiber] = [lower, upper, lower_matches[0], upper_matches[0]]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return multi-body, clipped, no-contact, and owner-tie cases."""
    return [
        {
            "setup": """
import numpy as np
roots1 = np.array([
    [0.10, 0.80], [0.20, 0.90], [-0.40, 0.65],
    [-0.40, 1.30], [0.15, 1.40], [-0.40, 1.30],
])
roots2 = np.array([
    [0.25, 0.70], [0.35, 1.20], [-0.20, 0.85],
    [-0.20, 0.85], [-0.30, 1.20], [-0.20, 1.20],
])
roots3 = np.array([
    [0.20, 0.75], [0.30, 0.75], [-0.10, 0.90],
    [0.05, 1.10], [-0.10, 1.10], [-0.10, 1.10],
])
root_sets = np.stack([roots1, roots2, roots3])
""",
            "call": "intersect_inside_intervals(root_sets)",
            "gold_call": "_oracle_intersect_inside_intervals(root_sets)",
        },
        {
            "setup": """
import numpy as np
roots1 = np.array([[-0.30, 1.40], [0.10, 0.20]])
roots2 = np.array([[-0.10, 1.10], [0.30, 0.40]])
root_sets = np.stack([roots1, roots2])
def encode_array(value):
    array = np.asarray(value, dtype=float)
    return np.concatenate((
        np.nan_to_num(array, nan=0.0).ravel(),
        np.isnan(array).astype(float).ravel(),
    ))
def run_model():
    return encode_array(intersect_inside_intervals(root_sets))
def run_gold():
    return encode_array(_oracle_intersect_inside_intervals(root_sets))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np
roots1 = np.array([[0.10, 0.80], [np.nan, np.nan]])
roots2 = np.array([[0.80, 0.95], [0.20, 0.70]])
roots3 = np.array([[-0.20, 1.20], [0.25, 0.65]])
root_sets = np.stack([roots1, roots2, roots3])
def encode_array(value):
    array = np.asarray(value, dtype=float)
    return np.concatenate((
        np.nan_to_num(array, nan=0.0).ravel(),
        np.isnan(array).astype(float).ravel(),
    ))
def run_model():
    return encode_array(intersect_inside_intervals(root_sets))
def run_gold():
    return encode_array(_oracle_intersect_inside_intervals(root_sets))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np
roots1 = np.array([[0.20, 0.80]])
roots2 = np.array([[0.20 + 5.0e-13, 0.70]])
roots3 = np.array([[0.10, 0.90]])
root_sets = np.stack([roots1, roots2, roots3])
def run_model():
    try:
        intersect_inside_intervals(root_sets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_intersect_inside_intervals(root_sets)
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
