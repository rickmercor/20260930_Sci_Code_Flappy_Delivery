"""
Build the pairwise homogeneous constraint matrix that a sparse, partially observed, multiplicatively separable N-way array imposes on the factor pattern of the axes complementary to one removed axis.

Two observed cells that agree on the removed axis share that axis's factor, so together they constrain the unknown complementary-axes factor pattern x (one unknown per distinct complementary index tuple). One matrix row is emitted per such pair. Conventions the tests depend on: columns are indexed by the distinct complementary index tuples in ascending sorted order; pairs are enumerated by ascending removed-axis value, and within one removed-axis value by ascending (complementary tuple, value) order, taking each unordered pair (a, b) with a before b; the row for pair (a, b) is oriented so that its entry in the column of b's complementary tuple is positive for positive observed values, and rows are not rescaled. If no removed-axis value carries two or more observations, the result is a one-dimensional empty array of shape (0,).

Returns
-------
np.ndarray: the (n_pairs, n_complementary) pairwise constraint matrix for the removed axis, or a 1-D empty array of shape (0,) when no removed-axis value has two or more observations
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_axis_constraint_matrix(obs: dict, axis: int) -> np.ndarray:
    """Build the pairwise homogeneous constraint matrix for one removed axis.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices to observed float values of a
        sparsely sampled array assumed to be an outer product of one factor
        vector per axis (no interaction terms).
    axis : int
        Which index position (0 <= axis < N) to remove; two observations are
        paired when they agree on this axis.

    Returns
    -------
    B : np.ndarray
        Array of shape (n_pairs, n_complementary): n_complementary is the
        number of distinct complementary index tuples in obs, columns ordered
        by the ascending sorted complementary tuple; rows follow the pair
        enumeration order in the module docstring, each oriented so that its
        entry in the column of the later observation's complementary tuple is
        positive for positive observed values, and are not rescaled. A one-dimensional empty array of
        shape (0,) when no removed-axis value has two or more observations.

    Raises
    ------
    ValueError
        If obs is empty, index tuples differ in length, or axis is out of range.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_axis_constraint_matrix(obs: dict, axis: int) -> np.ndarray:
    if not obs:
        raise ValueError("obs must be a non-empty dict of observations")
    n_dims = len(next(iter(obs)))
    if not (0 <= axis < n_dims):
        raise ValueError(f"axis must be in [0, {n_dims}), got {axis}")
    if any(len(idx) != n_dims for idx in obs):
        raise ValueError("all index tuples in obs must have the same length")
    complementary = sorted({idx[:axis] + idx[axis + 1:] for idx in obs})
    pos = {v: n for n, v in enumerate(complementary)}
    groups: dict = {}
    for idx, val in obs.items():
        groups.setdefault(idx[axis], []).append((idx[:axis] + idx[axis + 1:], val))
    rows = []
    for key in sorted(groups):
        entries = sorted(groups[key])
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                (idx_a, val_a), (idx_b, val_b) = entries[a], entries[b]
                row = np.zeros(len(complementary))
                row[pos[idx_b]] += val_a
                row[pos[idx_a]] -= val_b
                rows.append(row)
    return np.array(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
obs = {(0,0,0): 10.0, (0,0,1): 20.0, (1,0,0): 15.0, (0,1,0): 14.0}
axis = 0
""",
            "call": "build_axis_constraint_matrix(obs.copy(), axis)",
            "gold_call": "_oracle_build_axis_constraint_matrix(obs.copy(), axis)",
        },
        {
            "setup": """
obs = {(0,0,0): 5.0, (1,1,1): 7.0}
axis = 0
""",
            "call": "build_axis_constraint_matrix(obs.copy(), axis)",
            "gold_call": "_oracle_build_axis_constraint_matrix(obs.copy(), axis)",
        },
        {
            "setup": """
obs = {(0,0,1):0.4864,(0,0,2):0.5472,(0,1,1):0.7296,(0,2,2):0.9576,
       (1,0,1):0.3584,(1,1,1):0.5376,(2,0,2):0.7416,(2,1,0):1.3596,
       (2,1,1):0.9888,(2,1,2):1.1124,(2,2,1):1.1536,(2,3,1):0.9064,
       (3,0,0):0.8976,(3,1,1):0.9792,(3,2,2):1.2852,(4,1,1):0.6816,
       (5,0,1):0.5120}
axis = 2
""",
            "call": "build_axis_constraint_matrix(obs.copy(), axis)",
            "gold_call": "_oracle_build_axis_constraint_matrix(obs.copy(), axis)",
        },
    ]
