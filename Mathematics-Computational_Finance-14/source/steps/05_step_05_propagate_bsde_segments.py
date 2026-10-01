"""
Propagate every BSDE component forward on its own interval.

Every backward component is rewritten as a forward recurrence on its own

interval. The driver depends on both the value and the control, so the

discrete drift contributes one term proportional to the current value and a

second proportional to the control contracted with the driver coefficient.

The control also enters through the Brownian contraction at each local step.

Every segment begins from its separately parameterized value head.

Inputs

------

value_starts: Segment-start values of shape (n_segments, n_paths).

controls: Grid controls of shape (n_steps, n_paths, dim_w).

increments: Brownian increments with the same shape as controls.

segment_steps: Positive grid counts for the successive segments.

r: Risk-free rate in the driver f=-rY+theta.Z.

h: Positive uniform time step.

theta: Driver control coefficient of shape (dim_w,).

Returns

-------

terminals: Float array of segment right-end values.

Returns
-------
np.ndarray of shape (m, n_paths), segment endpoint values as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def propagate_bsde_segments(
    value_starts: np.ndarray,
    controls: np.ndarray,
    increments: np.ndarray,
    segment_steps: np.ndarray,
    r: float,
    h: float,
    theta: np.ndarray,
) -> np.ndarray:
    """Propagate every BSDE segment forward to its right endpoint.

    Parameters
    ----------
    value_starts : np.ndarray
        Segment-start values of shape ``(m, n_paths)``.
    controls : np.ndarray
        Martingale controls of shape ``(n_steps, n_paths, dim_w)``.
    increments : np.ndarray
        Brownian increments with the same shape as controls.
    segment_steps : np.ndarray
        Positive step counts of shape ``(m,)`` summing to n_steps.
    r : float
        Finite risk-free rate in the driver ``f=-rY+theta.Z``.
    h : float
        Positive finite time step.
    theta : np.ndarray
        Finite driver control coefficient of shape ``(dim_w,)``.

    Raises
    ------
    ValueError
        If shapes disagree, segment counts are not positive integers or do not
        cover the grid, h is not positive, or any numeric input is nonfinite.

    Returns
    -------
    terminals : np.ndarray
        Forward-propagated segment endpoint values of shape ``(m, n_paths)``.
    """
    return terminals  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_propagate_bsde_segments(
    value_starts: np.ndarray,
    controls: np.ndarray,
    increments: np.ndarray,
    segment_steps: np.ndarray,
    r: float,
    h: float,
    theta: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    value_starts = np.asarray(value_starts, dtype=float)
    controls = np.asarray(controls, dtype=float)
    increments = np.asarray(increments, dtype=float)
    segment_steps = np.asarray(segment_steps)
    theta = np.asarray(theta, dtype=float)
    if value_starts.ndim != 2 or min(value_starts.shape) <= 0:
        raise ValueError("value_starts must be a nonempty matrix")
    if controls.ndim != 3 or increments.shape != controls.shape or controls.shape[1] != value_starts.shape[1]:
        raise ValueError("controls and increments must match paths and each other")
    if segment_steps.shape != (value_starts.shape[0],) or not np.issubdtype(segment_steps.dtype, np.integer):
        raise ValueError("segment_steps must be one integer count per segment")
    if np.any(segment_steps <= 0) or int(np.sum(segment_steps)) != controls.shape[0]:
        raise ValueError("segment_steps must be positive and cover the full grid")
    if theta.shape != (controls.shape[2],):
        raise ValueError("theta must have one coefficient per Brownian dimension")
    if not np.isfinite(r) or not np.isfinite(h) or h <= 0:
        raise ValueError("r must be finite and h must be finite and positive")
    if not all(np.all(np.isfinite(array)) for array in (value_starts, controls, increments, theta)):
        raise ValueError("all arrays must be finite")
    terminals = np.empty_like(value_starts, dtype=float)
    offset = 0
    for j, count in enumerate(segment_steps.astype(int)):
        y = value_starts[j].copy()
        for i in range(offset, offset + count):
            drift = -(-float(r) * y + controls[i] @ theta)
            y = y + drift * float(h) + np.sum(controls[i] * increments[i], axis=1)
        terminals[j] = y
        offset += count
    return terminals

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """value_starts = np.array([[2.0, 3.0], [1.0, 1.5]])
controls = np.array([[[0.5], [0.2]], [[0.4], [0.1]], [[-0.3], [0.6]]])
increments = np.array([[[0.1], [-0.2]], [[0.05], [0.3]], [[-0.1], [0.2]]])
segment_steps = np.array([2, 1])
r = 0.06
h = 0.125
theta = np.array([0.45])
""",
            "call": "np.round(propagate_bsde_segments(value_starts, controls, increments, segment_steps, r, h, theta), 12).tolist()",
            "gold_call": "np.round(_oracle_propagate_bsde_segments(value_starts, controls, increments, segment_steps, r, h, theta), 12).tolist()",
        },
        {
            "setup": """value_starts = np.array([[2.0]])
controls = np.zeros((1, 1, 1))
increments = np.zeros((1, 1, 1))
segment_steps = np.array([1])
r = 0.0
h = 1.0
theta = np.array([0.0])
""",
            "call": "np.round(propagate_bsde_segments(value_starts, controls, increments, segment_steps, r, h, theta), 12).tolist()",
            "gold_call": "np.round(_oracle_propagate_bsde_segments(value_starts, controls, increments, segment_steps, r, h, theta), 12).tolist()",
        },
        {
            "setup": """value_starts = np.ones((2, 1))
controls = np.zeros((2, 1, 1))
increments = np.zeros((2, 1, 1))
segment_steps = np.array([1, 2])
theta = np.array([0.45])
def run_model():
    try:
        propagate_bsde_segments(value_starts, controls, increments, segment_steps, 0.0, 0.1, theta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_propagate_bsde_segments(value_starts, controls, increments, segment_steps, 0.0, 0.1, theta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
