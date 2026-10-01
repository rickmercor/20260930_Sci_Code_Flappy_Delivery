"""
Aggregate contact-interval lengths into a weighted overlap area.



If a fiber has parameter interval `$[h_lower, h_upper]$`, its contact length

is ``||end - start|| * (h_upper - h_lower)``.  Multiplication by a line weight

with units of length gives an area contribution.  Rows containing no contact

are represented by ``NaN`` interval bounds and contribute zero. Compute the

two-dimensional line norm without overflowing for finite, widely scaled

coordinates.

A line-conditioned area estimator converts each active parameter interval into physical length using the line norm. For deterministic lines with prescribed weights $w_i$, the overlap estimate is



$$

V_c=\sum_i w_i\,\lVert q_i\rVert\max(0,h_{\mathrm{out},i}-h_{\mathrm{in},i}).

$$



The weights have units of length, so each product contributes an area. Inactive intervals contribute exactly zero. Because finite segment components may differ greatly in scale, the two-dimensional norm should be evaluated with a scaled hypotenuse operation rather than by forming $q_x^2+q_y^2$, which can overflow even when the weighted area contribution is finite.

Returns
-------
one finite nonnegative float, the weighted overlap area in square metres
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def estimate_overlap_area(
    interval_state: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Estimate contact area from weighted fiber-intersection lengths.

    Parameters
    ----------
    interval_state : np.ndarray
        Shape ``(n_fibers, 4)`` from the contact-interval step.
    starts, ends : np.ndarray
        Matching finite arrays of shape ``(n_fibers, 2)`` in metres.
    weights : np.ndarray
        Finite nonnegative line weights of shape ``(n_fibers,)`` in metres,
        with at least one positive value.

    Returns
    -------
    float
        Finite nonnegative overlap estimate in square metres. Finite products
        must be retained when a segment component is near ``1e200`` and its
        corresponding line weight is near ``1e-200``.

    Raises
    ------
    ValueError
        If any weight is negative, all weights are zero, or an active interval
        has a bound outside ``[0, 1]``.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_estimate_overlap_area(
    interval_state: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Reference weighted line-length quadrature."""
    state = np.asarray(interval_state, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] < 1 or state.shape[1] != 4:
        raise ValueError("interval_state must have shape (n_fibers, 4)")
    if starts.shape != (state.shape[0], 2) or ends.shape != starts.shape:
        raise ValueError("starts and ends must match the number of intervals")
    if weights.shape != (state.shape[0],):
        raise ValueError("weights must have shape (n_fibers,)")
    if not all(np.all(np.isfinite(value)) for value in (starts, ends, weights)):
        raise ValueError("fiber data and weights must be finite")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("weights must be nonnegative with at least one positive value")
    segments = ends - starts
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    if np.any(lengths <= 0.0):
        raise ValueError("fibers must have positive length")
    active = np.isfinite(state[:, 0])
    if np.any(active & ~np.all(np.isfinite(state), axis=1)):
        raise ValueError("active interval rows must be finite")
    if np.any((~active) & ~np.all(np.isnan(state), axis=1)):
        raise ValueError("inactive interval rows must contain four NaNs")
    if np.any(active & ((state[:, 0] < 0.0) | (state[:, 1] > 1.0))):
        raise ValueError("active bounds must lie in [0, 1]")
    if np.any(active & (state[:, 1] <= state[:, 0])):
        raise ValueError("active intervals must have positive length")
    fractions = np.zeros(state.shape[0], dtype=np.float64)
    fractions[active] = state[active, 1] - state[active, 0]
    area = float(np.sum(weights * lengths * fractions))
    if not np.isfinite(area) or area < 0.0:
        raise ValueError("overlap estimate must be finite and nonnegative")
    return area

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, zero-contact, scaled-mask, and invalid-weight cases."""
    return [
        {
            "setup": """
import numpy as np
state = np.array([[0.2, 0.7, 1.0, 2.0], [0.1, 0.4, 2.0, 1.0]])
starts = np.array([[0.0, 0.0], [0.0, 0.0]])
ends = np.array([[2.0, 0.0], [0.0, 3.0]])
weights = np.array([0.25, 0.5])
""",
            "call": "estimate_overlap_area(state, starts, ends, weights)",
            "gold_call": "_oracle_estimate_overlap_area(state, starts, ends, weights)",
        },
        {
            "setup": """
import numpy as np
state = np.full((2, 4), np.nan)
starts = np.array([[0.0, 0.0], [1.0, 0.0]])
ends = np.array([[1.0, 0.0], [1.0, 2.0]])
weights = np.array([0.2, 0.3])
""",
            "call": "estimate_overlap_area(state, starts, ends, weights)",
            "gold_call": "_oracle_estimate_overlap_area(state, starts, ends, weights)",
        },
        {
            "setup": """
import numpy as np
state = np.array([
    [0.15, 0.85, 1.0, 2.0],
    [np.nan, np.nan, np.nan, np.nan],
    [0.05, 0.40, 0.0, 1.0],
    [0.60, 0.95, 2.0, 0.0],
])
starts = np.array([
    [0.0, 0.0], [0.0, 0.0], [0.3, -1.2], [-0.8, 0.9]
])
ends = np.array([
    [3.0e200, 4.0e200], [0.0, 2.0], [1.1, 1.5], [1.6, -0.3]
])
weights = np.array([2.0e-201, 0.17, 0.31, 0.23])
""",
            "call": "estimate_overlap_area(state, starts, ends, weights)",
            "gold_call": "_oracle_estimate_overlap_area(state, starts, ends, weights)",
        },
        {
            "setup": """
import numpy as np
state = np.array([[0.2, 0.7, 1.0, 2.0]])
starts = np.array([[0.0, 0.0]])
ends = np.array([[1.0, 0.0]])
weights = np.array([-0.1])
def run_model():
    try:
        estimate_overlap_area(state, starts, ends, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_estimate_overlap_area(state, starts, ends, weights)
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
