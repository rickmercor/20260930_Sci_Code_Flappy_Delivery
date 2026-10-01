"""
Construct the intermediate reflection and final payoff targets.

At each intermediate Bermudan date, the paper's compounding condition couples

the ending component to the next segment's continuation value and the immediate

exercise value. The final boundary follows the terminal convention rather than

an intermediate coupling. Exercise depends on the geometric basket formed from

the asset state at the corresponding segment endpoint.

Inputs

------

paths: Positive asset paths of shape (n_steps + 1, n_paths, d).

value_starts: Continuation values at the segment starts.

segment_steps: Positive grid counts for the successive segments.

strikes: Put strikes for the compounding and terminal dates.

Returns

-------

targets: Float array of shape (n_segments, n_paths).

Returns
-------
np.ndarray of shape (m, n_paths), Bermudan targets as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_bermudan_targets(
    paths: np.ndarray,
    value_starts: np.ndarray,
    segment_steps: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    """Construct intermediate reflection targets and the terminal payoff.

    Parameters
    ----------
    paths : np.ndarray
        Positive paths of shape ``(n_steps + 1, n_paths, d)``.
    value_starts : np.ndarray
        Segment-start continuation values of shape ``(m, n_paths)``.
    segment_steps : np.ndarray
        Positive integer grid counts of shape ``(m,)``.
    strikes : np.ndarray
        Finite strikes of shape ``(m,)``.

    Raises
    ------
    ValueError
        If shapes disagree, paths are not positive and finite, segment counts
        are invalid, or values and strikes are nonfinite.

    Returns
    -------
    targets : np.ndarray
        Compounding and terminal targets of shape ``(m, n_paths)``.
    """
    return targets  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_construct_bermudan_targets(
    paths: np.ndarray,
    value_starts: np.ndarray,
    segment_steps: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    paths = np.asarray(paths, dtype=float)
    value_starts = np.asarray(value_starts, dtype=float)
    segment_steps = np.asarray(segment_steps)
    strikes = np.asarray(strikes, dtype=float)
    if paths.ndim != 3 or min(paths.shape) <= 0 or np.any(paths <= 0) or not np.all(np.isfinite(paths)):
        raise ValueError("paths must be a nonempty positive finite array")
    if value_starts.ndim != 2 or value_starts.shape[1] != paths.shape[1]:
        raise ValueError("value_starts must match the number of paths")
    m = value_starts.shape[0]
    if segment_steps.shape != (m,) or not np.issubdtype(segment_steps.dtype, np.integer):
        raise ValueError("segment_steps must be one integer count per segment")
    if np.any(segment_steps <= 0) or int(np.sum(segment_steps)) != paths.shape[0] - 1:
        raise ValueError("segment_steps must be positive and cover the path grid")
    if strikes.shape != (m,) or not np.all(np.isfinite(strikes)) or not np.all(np.isfinite(value_starts)):
        raise ValueError("strikes and value_starts must be finite and shape compatible")
    targets = np.empty_like(value_starts, dtype=float)
    for j, end in enumerate(np.cumsum(segment_steps.astype(int))):
        basket = np.exp(np.mean(np.log(paths[end]), axis=1))
        exercise = np.maximum(strikes[j] - basket, 0.0)
        targets[j] = np.maximum(value_starts[j + 1], exercise) if j < m - 1 else exercise
    return targets

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """paths = np.array([[[4.0, 9.0], [9.0, 16.0]], [[1.0, 4.0], [16.0, 25.0]], [[4.0, 4.0], [25.0, 25.0]]])
value_starts = np.array([[1.0, 1.0], [2.0, 3.0]])
segment_steps = np.array([1, 1])
strikes = np.array([5.0, 6.0])
""",
            "call": "np.round(construct_bermudan_targets(paths, value_starts, segment_steps, strikes), 12).tolist()",
            "gold_call": "np.round(_oracle_construct_bermudan_targets(paths, value_starts, segment_steps, strikes), 12).tolist()",
        },
        {
            "setup": """paths = np.array([[[5.0]], [[5.0]]])
value_starts = np.array([[1.0]])
segment_steps = np.array([1])
strikes = np.array([5.0])
""",
            "call": "np.round(construct_bermudan_targets(paths, value_starts, segment_steps, strikes), 12).tolist()",
            "gold_call": "np.round(_oracle_construct_bermudan_targets(paths, value_starts, segment_steps, strikes), 12).tolist()",
        },
        {
            "setup": """paths = np.array([[[5.0]], [[-1.0]]])
value_starts = np.array([[1.0]])
segment_steps = np.array([1])
strikes = np.array([5.0])
def run_model():
    try:
        construct_bermudan_targets(paths, value_starts, segment_steps, strikes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_construct_bermudan_targets(paths, value_starts, segment_steps, strikes)
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
