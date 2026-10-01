"""
Apply the credible-set endpoint sensitivity restriction to graph edges.

 

Retain an edge exactly when both indexed endpoints have a credible set. Preserve the input row order and all three columns. Endpoint columns must be integer-valued, distinct, ordered as i<j, in range, and free of duplicates.

Credible-set availability indicates whether a signal has a localized fine-mapping summary and enables endpoint sensitivity analysis.

Returns
-------
restricted_edges : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def restrict_credible_set_edges(
    edges: "np.ndarray",
    has_credible_set: "np.ndarray",
) -> "np.ndarray":
    """
    Keep graph edges whose two endpoints have credible sets.
 
    Parameters
    ----------
    edges : np.ndarray
        Float edge rows [i, j, weight].
    has_credible_set : np.ndarray
        Boolean flag per graph node.
 
    Returns
    -------
    np.ndarray
        Filtered float array with shape (n_retained_edges, 3).
 
    Raises
    ------
    ValueError
        If the edge table or endpoint mask is invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_restrict_credible_set_edges(
    edges: "np.ndarray",
    has_credible_set: "np.ndarray",
) -> "np.ndarray":
    e = np.asarray(edges, dtype=float)
    h_raw = np.asarray(has_credible_set)
    if e.ndim != 2 or e.shape[1] != 3:
        raise ValueError("edges must have shape (n_edges,3)")
    if h_raw.ndim != 1 or h_raw.size < 1 or h_raw.dtype.kind != "b":
        raise ValueError("has_credible_set must be a non-empty boolean vector")
    if not np.all(np.isfinite(e)):
        raise ValueError("edge values must be finite")
    if e.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    endpoints = e[:, :2]
    if not np.all(endpoints == np.floor(endpoints)):
        raise ValueError("edge endpoints must be integer-valued")
    ij = endpoints.astype(int)
    if np.any(ij < 0) or np.any(ij >= h_raw.size) or np.any(ij[:, 0] >= ij[:, 1]):
        raise ValueError("edge endpoints must satisfy 0 <= i < j < n")
    if len({tuple(row) for row in ij.tolist()}) != ij.shape[0]:
        raise ValueError("duplicate edges are not allowed")
    if np.any(e[:, 2] < 0.0) or np.any(e[:, 2] > 1.0):
        raise ValueError("edge weights must lie in [0,1]")
    keep = h_raw[ij[:, 0]] & h_raw[ij[:, 1]]
    return e[keep].astype(float, copy=True).reshape((-1, 3))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
e=np.array([[0.,1.,.9],[1.,2.,.8],[2.,3.,.95]]); h=np.array([True,True,False,True])""",
            "call": "restrict_credible_set_edges(e,h)",
            "gold_call": "_oracle_restrict_credible_set_edges(e,h)",
        },
        {
            "setup": """import numpy as np
e=np.empty((0,3)); h=np.array([True,False])""",
            "call": "restrict_credible_set_edges(e,h)",
            "gold_call": "_oracle_restrict_credible_set_edges(e,h)",
        },
        {
            "setup": """import numpy as np
e=np.array([[0.,1.,1.],[1.,2.,0.]]); h=np.ones(3,dtype=bool)""",
            "call": "restrict_credible_set_edges(e,h)",
            "gold_call": "_oracle_restrict_credible_set_edges(e,h)",
        },
        {
            "setup": """import numpy as np
e=np.array([[1.,1.,.9]]); h=np.ones(2,dtype=bool)
def candidate_code():
    try: restrict_credible_set_edges(e,h); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_restrict_credible_set_edges(e,h); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
