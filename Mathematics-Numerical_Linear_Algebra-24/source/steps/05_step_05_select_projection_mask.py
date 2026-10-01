"""
Select the next set of elements to project.



An element that has not yet been processed becomes eligible when its score

stands in the required relation to the current threshold. The boundary case, a

score exactly equal to the threshold, is decided by the established convention

rather than by an arbitrary choice, and that convention is not restated here.

When the caller signals the fallback regime, every unprocessed element is

selected regardless of score.

Returns
-------
integer np.ndarray of shape (n_elements,), the 0/1 next-round mask
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_projection_mask(
    scores: np.ndarray,
    delta: float,
    projected_mask: np.ndarray,
    force_all: bool = False,
) -> np.ndarray:
    """Select unprocessed elements for the next projection round.

    Raises ``ValueError`` unless every one of the following holds: ``scores``
    is a nonempty one-dimensional array whose entries are all finite and
    nonnegative; ``projected_mask`` has ``dtype=bool`` and exactly the shape
    of ``scores``; ``delta`` is a scalar that is neither NaN nor negative
    (``inf`` is permitted); and ``force_all`` is a Boolean.

    Parameters
    ----------
    scores : np.ndarray
        Nonnegative element scores of shape ``(n_elements,)``.
    delta : float
        Current nonnegative selection threshold.
    projected_mask : np.ndarray
        Record of elements already processed, with ``dtype=bool`` and the
        shape of ``scores``.
    force_all : bool, optional
        When true, select every unprocessed element and ignore ``delta``.

    Returns
    -------
    np.ndarray
        Integer 0/1 mask of shape ``(n_elements,)``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_select_projection_mask(scores, delta, projected_mask, force_all=False):
    """Reference implementation of the strict gate and fallback."""
    
    scores = np.asarray(scores, dtype=float)
    mask = np.asarray(projected_mask)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("scores must be finite and nonnegative")
    if mask.shape != scores.shape or mask.dtype != np.bool_:
        raise ValueError("projected_mask must be Boolean and match scores")
    if not np.isscalar(delta) or np.isnan(float(delta)) or float(delta) < 0.0:
        raise ValueError("delta must be a nonnegative scalar")
    if not isinstance(force_all, (bool, np.bool_)):
        raise ValueError("force_all must be Boolean")  # noqa: TRY004
    remaining = ~mask
    chosen = remaining if force_all else remaining & (scores > float(delta))
    return chosen.astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return strict-tie, fallback, and invalid-shape cases."""
    return [
        {
            "setup": """
import numpy as np
scores = np.array([32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.25, 32.0, 8.0, 16.0])
delta = 16.0
projected = np.zeros(12, dtype=bool)
""",
            "call": "select_projection_mask(scores, delta, projected)",
            "gold_call": "_oracle_select_projection_mask(scores, delta, projected)",
        },
        {
            "setup": """
import numpy as np
scores = np.array([32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.25, 32.0, 8.0, 16.0])
delta = 4.0
projected = np.array([True, True, False, False, False, False, False, False, False, True, False, True])
""",
            "call": "select_projection_mask(scores, delta, projected, force_all=True)",
            "gold_call": "_oracle_select_projection_mask(scores, delta, projected, force_all=True)",
        },
        {
            "setup": """
import numpy as np
scores = np.array([1.0, 2.0])
delta = 1.0
projected = np.zeros(3, dtype=bool)

def run_model():
    try:
        select_projection_mask(scores, delta, projected)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_select_projection_mask(scores, delta, projected)
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
