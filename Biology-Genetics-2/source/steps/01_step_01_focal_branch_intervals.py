"""
Locate, in every marginal tree of an ancestral recombination graph, the branch on

the focal haplotype's ancestral lineage whose time span contains the archaic time

cutoff, and return the lower and upper endpoint of that branch for each tree.

A marginal tree records the genealogy of a set of sampled haplotypes at one

position of the genome. Walking from a focal sample towards the root passes

through a sequence of coalescence events whose times increase monotonically, and

the branch between two consecutive events is the interval during which the focal

lineage carried no coalescence. Gene flow from a deeply divergent source places

the focal lineage in a population that cannot coalesce with the recipient gene

pool until the two populations merge, so an introgressed haplotype is expected to

carry one unusually long branch that reaches from shortly after the admixture

pulse back past the population divergence. A time cutoff t chosen near the

suspected divergence therefore selects, in each tree, the single branch of the

focal lineage whose lower endpoint lies at or below t and whose upper endpoint

lies strictly above t. The terminal branch counts as part of that sequence, so

the walk starts at time zero, and a tree whose root is no deeper than t contains

no branch spanning the cutoff, in which case the interval collapses to the root

time and has zero length.

Returns
-------
np.ndarray of shape (m, 2), the focal branch endpoints in generations as a float64 array, with column 0 the lower endpoint and column 1 the upper endpoint
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def focal_branch_intervals(
    coal_times: "np.ndarray",
    focal_mask: "np.ndarray",
    t_archaic: float = 15000.0,
) -> "np.ndarray":
    """Return the focal branch endpoints spanning t_archaic in each marginal tree.

    Parameters
    ----------
    coal_times : np.ndarray
        Coalescence times of shape (m, c), ascending along each row.
    focal_mask : np.ndarray
        Binary indicator of shape (m, c) marking the coalescences that lie on the
        focal haplotype's path to the root.
    t_archaic : float
        Time cutoff in generations defining an archaic event.

    Returns
    -------
    intervals : np.ndarray
        Array of shape (m, 2) holding the lower and upper endpoint of the focal
        branch that spans t_archaic in each tree.

    Raises
    ------
    ValueError
        If coal_times is not two dimensional, if focal_mask has a different shape
        or holds entries outside {0, 1}, if a tree has no focal coalescence, if
        any coalescence time is not strictly positive, if the times do not
        increase strictly along a row, or if t_archaic is not positive and finite.
    """
    return intervals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_focal_branch_intervals(
    coal_times: "np.ndarray",
    focal_mask: "np.ndarray",
    t_archaic: float = 15000.0,
) -> "np.ndarray":
    """Reference implementation."""
    coal_times = np.asarray(coal_times, dtype=float)
    focal_mask = np.asarray(focal_mask, dtype=float)
    if coal_times.ndim != 2 or coal_times.shape[1] < 1:
        raise ValueError("coal_times must be a two dimensional array of shape (m, c)")
    if focal_mask.shape != coal_times.shape:
        raise ValueError("focal_mask must have the same shape as coal_times")
    if not np.all(np.isin(focal_mask, (0.0, 1.0))):
        raise ValueError("focal_mask entries must be 0 or 1")
    if np.any(focal_mask.sum(axis=1) < 1.0):
        raise ValueError("every marginal tree must have at least one focal coalescence")
    if np.any(coal_times <= 0.0):
        raise ValueError("coalescence times must be strictly positive")
    if np.any(np.diff(coal_times, axis=1) <= 0.0):
        raise ValueError("coal_times must increase strictly along each row")
    if not np.isfinite(t_archaic) or t_archaic <= 0.0:
        raise ValueError("t_archaic must be a positive finite time")

    m = coal_times.shape[0]
    intervals = np.zeros((m, 2), dtype=float)
    for i in range(m):
        path = np.concatenate(([0.0], coal_times[i][focal_mask[i] == 1.0]))
        root = path[-1]
        lower, upper = root, root
        for j in range(path.size - 1):
            if path[j] <= t_archaic < path[j + 1]:
                lower, upper = path[j], path[j + 1]
                break
        intervals[i] = (lower, upper)
    return intervals

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
_MASK = np.array([
    [0.0, 0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0, 1.0],
])
'''
    return [
        # --- Normal scenario: a mix of terminal, interior and long focal branches ---
        {
            "setup": fixture + """coal_times = _COAL.copy()
focal_mask = _MASK.copy()
t_archaic = 15000.0
""",
            "call": "focal_branch_intervals(coal_times, focal_mask, t_archaic=t_archaic).tolist()",
            "gold_call": "_oracle_focal_branch_intervals(coal_times, focal_mask, t_archaic=t_archaic).tolist()",
        },
        # --- Boundary case: a tree whose root is shallower than the cutoff ---
        {
            "setup": fixture + """coal_times = np.array([[900.0, 3200.0, 8400.0]])
focal_mask = np.array([[1.0, 0.0, 1.0]])
t_archaic = 15000.0
""",
            "call": "focal_branch_intervals(coal_times, focal_mask, t_archaic=t_archaic).tolist()",
            "gold_call": "_oracle_focal_branch_intervals(coal_times, focal_mask, t_archaic=t_archaic).tolist()",
        },
        # --- Edge case: coalescence times that do not increase must be rejected ---
        {
            "setup": fixture + """coal_times = np.array([[1200.0, 1200.0, 40000.0]])
focal_mask = np.array([[1.0, 0.0, 1.0]])
def run_model():
    try:
        focal_branch_intervals(coal_times, focal_mask, t_archaic=15000.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_focal_branch_intervals(coal_times, focal_mask, t_archaic=15000.0)
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
