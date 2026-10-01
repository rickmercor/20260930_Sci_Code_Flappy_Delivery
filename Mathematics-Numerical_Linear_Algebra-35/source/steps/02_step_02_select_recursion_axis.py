"""
Select which axis of a sparsely observed, multiplicatively separable N-way array is resolved next in the recursive completion, from the observed cells alone.

The choice is made from the singular-value spectra of the per-axis pairwise constraint matrices produced by build_axis_constraint_matrix. Conventions the tests depend on: minimum singular values are compared with a relative tolerance of 1e-10 times the largest singular value found over all axes (values within that tolerance of the smallest count as tied); the returned value is the zero-based axis position; if the selection rule still leaves a tie, the smallest axis position is returned. Axes whose constraint matrix has no rows are not candidates.

Returns
-------
int: zero-based position of the axis selected for resolution next
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_recursion_axis(obs: dict) -> int:
    """Choose the axis to resolve next from the observed cells.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices (N >= 2) to observed float values
        of a sparsely sampled, multiplicatively separable array.

    Returns
    -------
    axis : int
        Zero-based position of the axis selected for resolution, following the
        tie conventions in the module docstring.

    Raises
    ------
    ValueError
        If obs is empty, has index tuples of fewer than two entries, or no axis
        has a constraint matrix with at least one row.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_recursion_axis(obs: dict) -> int:
    if not obs:
        raise ValueError("obs must be a non-empty dict of observations")
    n_dims = len(next(iter(obs)))
    if n_dims < 2:
        raise ValueError("index tuples must have at least two entries")
    cand = []
    s_max = 0.0
    for kp in range(n_dims):
        B = _oracle_build_axis_constraint_matrix(obs, kp)
        if B.ndim != 2 or B.shape[0] == 0:
            continue
        s = np.linalg.svd(B, compute_uv=False)
        s_min = float(s[-1])
        gap = float(s[-2] - s[-1]) if len(s) >= 2 else float(s[-1])
        s_max = max(s_max, float(s[0]))
        cand.append((s_min, gap, kp))
    if not cand:
        raise ValueError("no axis has a constraint matrix with at least one row")
    tol = 1e-10 * max(1.0, s_max)
    smallest = min(c[0] for c in cand)
    tied = [c for c in cand if c[0] <= smallest + tol]
    best_gap = max(c[1] for c in tied)
    winners = [c[2] for c in tied if c[1] >= best_gap - tol]
    return int(min(winners))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
obs = {(0,0,1):0.4864,(0,0,2):0.5472,(0,1,1):0.7296,(0,2,2):0.9576,
       (1,0,1):0.3584,(1,1,1):0.5376,(2,0,2):0.7416,(2,1,0):1.3596,
       (2,1,1):0.9888,(2,1,2):1.1124,(2,2,1):1.1536,(2,3,1):0.9064,
       (3,0,0):0.8976,(3,1,1):0.9792,(3,2,2):1.2852,(4,1,1):0.6816,
       (5,0,1):0.5120}
""",
            "call": "select_recursion_axis(obs.copy())",
            "gold_call": "_oracle_select_recursion_axis(obs.copy())",
        },
        {
            "setup": """
obs = {(0,0): 6.0, (0,1): 15.0, (0,2): 21.0, (1,0): 8.0, (1,2): 28.0}
""",
            "call": "select_recursion_axis(obs.copy())",
            "gold_call": "_oracle_select_recursion_axis(obs.copy())",
        },
        {
            "setup": """
obs = {(0,0,0): 10.0, (0,0,1): 20.0, (1,0,0): 15.0, (0,1,0): 14.0, (1,1,1): 42.0,
       (1,0,1): 30.0, (0,1,1): 28.0}
""",
            "call": "select_recursion_axis(obs.copy())",
            "gold_call": "_oracle_select_recursion_axis(obs.copy())",
        },
    ]
