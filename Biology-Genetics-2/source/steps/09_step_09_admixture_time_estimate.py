"""
Combine the focal branch endpoints of the marginal trees that fall inside the

retained archaic tracts into a single estimate of the admixture time, weighting

each tree by the physical span it covers.

An archaic branch in a marginal tree is bounded below by the coalescence that

joins the focal lineage to another lineage of the same archaic ancestry and above

by the coalescence that joins it to the recipient gene pool. Under an

instantaneous pulse of gene flow the introgressed lineages carried by a genome

begin to coalesce with one another only once they are back inside the source

population, which they enter at the time of admixture, so the recent endpoint of

the branch cannot be older than the pulse and its average across tracts is a lower

bound on the admixture time. The older endpoint is bounded the other way: the

archaic and the recipient lineage cannot have coalesced after the two populations

separated, so it supplies an upper bound on the divergence time instead. Trees

differ in how much sequence they represent, and a tree that spans a wider interval

of the chromosome carries proportionally more of the evidence, so the endpoints

are averaged with weights equal to the physical span of each tree.

Returns
-------
float, the span-weighted admixture time in generations as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def admixture_time_estimate(
    segments: list,
    intervals: "np.ndarray",
    span_bp: "np.ndarray",
) -> float:
    """Return the span-weighted admixture time implied by the retained tracts.

    Parameters
    ----------
    segments : list
        List of (start, end) inclusive tree index pairs of the retained tracts.
    intervals : np.ndarray
        Focal branch endpoints of shape (m, 2).
    span_bp : np.ndarray
        Physical span of each marginal tree in base pairs.

    Returns
    -------
    t_admix : float
        Span-weighted admixture time in generations.

    Raises
    ------
    ValueError
        If segments is empty, if intervals does not have shape (m, 2) matching
        span_bp, if a segment index falls outside the range of trees, if a segment
        end precedes its start, or if the covered trees carry no positive span.
    """
    return t_admix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_admixture_time_estimate(
    segments: list,
    intervals: "np.ndarray",
    span_bp: "np.ndarray",
) -> float:
    """Reference implementation."""
    intervals = np.asarray(intervals, dtype=float)
    span_bp = np.asarray(span_bp, dtype=float)
    if span_bp.ndim != 1:
        raise ValueError("span_bp must be one dimensional")
    if intervals.shape != (span_bp.size, 2):
        raise ValueError("intervals must have shape (m, 2) matching span_bp")
    if len(segments) == 0:
        raise ValueError("at least one retained tract is required")

    covered = []
    for start, end in segments:
        start, end = int(start), int(end)
        if end < start:
            raise ValueError("a segment end must not precede its start")
        if start < 0 or end >= span_bp.size:
            raise ValueError("segment indices must lie within the range of trees")
        covered.extend(range(start, end + 1))
    covered = np.array(sorted(set(covered)), dtype=int)

    weights = span_bp[covered]
    if weights.sum() <= 0.0:
        raise ValueError("the covered trees carry no positive physical span")
    return float((weights * intervals[covered, 0]).sum() / weights.sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_BRANCH_ENDS = np.array([
    [11700.0, 33800.0],
    [2183.0, 48283.0],
    [1846.0, 51346.0],
    [2027.0, 46827.0],
    [2314.0, 49914.0],
    [2461.0, 25461.0],
    [10100.0, 19250.0],
    [2650.0, 47850.0],
])
_PHYSICAL_SPAN = np.array([
    15800.0, 13000.0, 12000.0, 13700.0, 13000.0, 12000.0, 13300.0, 62000.0,
])
'''
    return [
        # --- Normal scenario: one retained tract of five trees ---
        {
            "setup": fixture + """segments = [(1, 5)]
intervals = _BRANCH_ENDS.copy()
span_bp = _PHYSICAL_SPAN.copy()
""",
            "call": "round(admixture_time_estimate(segments, intervals, span_bp), 9)",
            "gold_call": "round(_oracle_admixture_time_estimate(segments, intervals, span_bp), 9)",
        },
        # --- Boundary case: two tracts, one of them a single tree ---
        {
            "setup": fixture + """segments = [(1, 5), (7, 7)]
intervals = _BRANCH_ENDS.copy()
span_bp = _PHYSICAL_SPAN.copy()
""",
            "call": "round(admixture_time_estimate(segments, intervals, span_bp), 9)",
            "gold_call": "round(_oracle_admixture_time_estimate(segments, intervals, span_bp), 9)",
        },
        # --- Edge case: no retained tract leaves the estimate undefined ---
        {
            "setup": fixture + """segments = []
intervals = _BRANCH_ENDS.copy()
span_bp = _PHYSICAL_SPAN.copy()
def run_model():
    try:
        admixture_time_estimate(segments, intervals, span_bp)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_admixture_time_estimate(segments, intervals, span_bp)
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
