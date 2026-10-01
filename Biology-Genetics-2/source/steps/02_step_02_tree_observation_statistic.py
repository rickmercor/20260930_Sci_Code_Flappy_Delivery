"""
Count, in each marginal tree, the coalescence events that fall strictly inside

the time span of the focal branch, and combine that count with the branch length

into the per-tree observation that the archaic ancestry model is fitted to.

Branch length alone is a weak indicator of archaic ancestry because a deep

genealogy can arise from a bottleneck, from incomplete lineage sorting, or from

error in the reconstructed graph. The genealogy carries a second and largely

independent signal over the same time window: while the focal lineage is waiting

in the divergent source population, the remaining lineages of the tree continue

to coalesce at the rate set by the recipient population size, so the branch is

accompanied by an accumulation of coalescences elsewhere in the tree. A long

branch that spans many other coalescence events is far more informative than a

long branch that spans none, and the two features are therefore combined into a

single observation per tree by multiplication, X = L * N, where L is the length

in generations of the focal branch spanning the archaic time cutoff and N is the

number of coalescence events of the same tree whose times lie strictly between

the two endpoints of that branch. Endpoints are excluded because they are the

coalescences that define the branch. A small positive floor is added so that the

observation stays inside the support of a gamma density even when a tree

contributes a zero-length branch or no interior coalescence.

Returns
-------
tuple of two np.ndarray of shape (m,): the interior coalescence counts as a float64 array and the per-tree observations as a float64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tree_observation_statistic(
    coal_times: "np.ndarray",
    intervals: "np.ndarray",
    x_floor: float = 1e-10,
) -> tuple:
    """Return the interior coalescence counts and the per-tree observation.

    Parameters
    ----------
    coal_times : np.ndarray
        Coalescence times of shape (m, c), ascending along each row.
    intervals : np.ndarray
        Focal branch endpoints of shape (m, 2).
    x_floor : float
        Positive floor added to every observation.

    Returns
    -------
    result : tuple
        The interior coalescence counts of shape (m,) followed by the per-tree
        observations of shape (m,).

    Raises
    ------
    ValueError
        If coal_times is not two dimensional, if intervals does not have shape
        (m, 2) matching coal_times, if any upper endpoint lies below its lower
        endpoint, or if x_floor is not strictly positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_tree_observation_statistic(
    coal_times: "np.ndarray",
    intervals: "np.ndarray",
    x_floor: float = 1e-10,
) -> tuple:
    """Reference implementation."""
    coal_times = np.asarray(coal_times, dtype=float)
    intervals = np.asarray(intervals, dtype=float)
    if coal_times.ndim != 2:
        raise ValueError("coal_times must be a two dimensional array of shape (m, c)")
    if intervals.shape != (coal_times.shape[0], 2):
        raise ValueError("intervals must have shape (m, 2) matching coal_times")
    if np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError("every upper endpoint must be at least its lower endpoint")
    if not np.isfinite(x_floor) or x_floor <= 0.0:
        raise ValueError("x_floor must be strictly positive")

    lower = intervals[:, 0][:, None]
    upper = intervals[:, 1][:, None]
    n_coal = np.sum((coal_times > lower) & (coal_times < upper), axis=1).astype(float)
    observations = n_coal * (intervals[:, 1] - intervals[:, 0]) + x_floor
    return n_coal, observations

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_COAL = np.array([
    [1300.0, 10930.0, 11700.0, 16900.0, 33800.0],
    [2183.0, 4400.0, 7570.0, 32420.0, 48283.0],
    [7300.0, 20920.0, 28530.0, 30470.0, 37800.0],
    [870.0, 2461.0, 3740.0, 15900.0, 25461.0],
])
_BRANCH_ENDS = np.array([
    [11700.0, 33800.0],
    [2183.0, 48283.0],
    [0.0, 20920.0],
    [2461.0, 25461.0],
])
'''
    return [
        # --- Normal scenario ---
        {
            "setup": fixture + """coal_times = _COAL.copy()
intervals = _BRANCH_ENDS.copy()
x_floor = 1e-10
""",
            "call": "[v.tolist() for v in tree_observation_statistic(coal_times, intervals, x_floor=x_floor)]",
            "gold_call": "[v.tolist() for v in _oracle_tree_observation_statistic(coal_times, intervals, x_floor=x_floor)]",
        },
        # --- Boundary case: a degenerate branch of zero length ---
        {
            "setup": fixture + """coal_times = np.array([[900.0, 3200.0, 8400.0]])
intervals = np.array([[8400.0, 8400.0]])
x_floor = 1e-10
""",
            "call": "[v.tolist() for v in tree_observation_statistic(coal_times, intervals, x_floor=x_floor)]",
            "gold_call": "[v.tolist() for v in _oracle_tree_observation_statistic(coal_times, intervals, x_floor=x_floor)]",
        },
        # --- Edge case: a non-positive floor must be rejected ---
        {
            "setup": fixture + """coal_times = _COAL.copy()
intervals = _BRANCH_ENDS.copy()
def run_model():
    try:
        tree_observation_statistic(coal_times, intervals, x_floor=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_tree_observation_statistic(coal_times, intervals, x_floor=0.0)
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
