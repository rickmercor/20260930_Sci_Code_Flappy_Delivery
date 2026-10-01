"""
Extract, up to scale, the factor pattern shared by the axes complementary to one removed axis of a sparsely observed, multiplicatively separable N-way array: one value per distinct complementary index tuple, in the column order of build_axis_constraint_matrix.

Conventions the tests depend on: the returned vector has unit Euclidean norm and its first entry of magnitude above 1e-12 is positive; its entries are ordered by the ascending sorted complementary index tuple, matching the constraint matrix's columns.

Returns
-------
np.ndarray: 1-D unit-norm complementary factor pattern, one entry per distinct complementary index tuple in ascending sorted order, first entry of magnitude above 1e-12 positive
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def extract_complementary_pattern(obs: dict, axis: int) -> np.ndarray:
    """Extract the complementary-axes factor pattern for one removed axis.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices to observed float values of a
        sparsely sampled, multiplicatively separable array.
    axis : int
        Zero-based position of the axis being removed.

    Returns
    -------
    x : np.ndarray
        One-dimensional array with one entry per distinct complementary index
        tuple (ascending sorted order), unit Euclidean norm, first entry of
        magnitude above 1e-12 positive.

    Raises
    ------
    ValueError
        If obs is empty, axis is out of range, or no removed-axis value has
        two or more observations (no constraint rows).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_extract_complementary_pattern(obs: dict, axis: int) -> np.ndarray:
    B = _oracle_build_axis_constraint_matrix(obs, axis)
    if B.ndim != 2 or B.shape[0] == 0:
        raise ValueError("no constraint rows: some removed-axis value needs two or more observations")
    Vt = np.linalg.svd(B, full_matrices=True)[2]
    x = np.array(Vt[-1], dtype=float)
    x = x / np.linalg.norm(x)
    nz = np.flatnonzero(np.abs(x) > 1e-12)
    if nz.size and x[nz[0]] < 0:
        x = -x
    return x

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
obs = {(0,0,0): 10.0, (0,0,1): 20.0, (1,0,0): 15.0, (0,1,0): 14.0, (1,1,1): 42.0}
axis = 0
""",
            "call": "extract_complementary_pattern(obs.copy(), axis)",
            "gold_call": "_oracle_extract_complementary_pattern(obs.copy(), axis)",
        },
        {
            "setup": """
obs = {(0,0): 6.0, (0,1): 15.0, (0,2): 21.0, (1,0): 8.0, (1,2): 28.0}
axis = 1
""",
            "call": "extract_complementary_pattern(obs.copy(), axis)",
            "gold_call": "_oracle_extract_complementary_pattern(obs.copy(), axis)",
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
            "call": "extract_complementary_pattern(obs.copy(), axis)",
            "gold_call": "_oracle_extract_complementary_pattern(obs.copy(), axis)",
        },
    ]
