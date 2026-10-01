"""
Convert pairwise evidence into a canonically ordered colocalisation graph.

 

Normalize each pair's five log evidences by stable log-sum-exp. For i<j, keep an edge only when both directional coverage values are at least overlap_min and the normalized H4 posterior is at least h4_threshold. Return rows [i, j, PP.H4] in lexicographic endpoint order.

Posterior-supported signal relationships can be represented as an undirected graph after eligibility checks.

Returns
-------
edges : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_colocalisation_edges(
    log_evidence: "np.ndarray",
    coverage: "np.ndarray",
    overlap_min: float,
    h4_threshold: float,
) -> "np.ndarray":
    """
    Select eligible H4-supported signal-pair edges.
 
    Parameters
    ----------
    log_evidence : np.ndarray
        Pairwise H0-H4 log evidence with shape (n, n, 5).
    coverage : np.ndarray
        Directional coverage with shape (n, n, 2).
    overlap_min : float
        Inclusive minimum for both coverage directions.
    h4_threshold : float
        Inclusive normalized-H4 posterior threshold.
 
    Returns
    -------
    np.ndarray
        Float array of shape (n_edges, 3) containing [i, j, PP.H4].
 
    Raises
    ------
    ValueError
        If shapes, evidence, coverage, or thresholds are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _edge_lse(values):
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    if not np.any(finite):
        return -np.inf
    m = float(np.max(a[finite]))
    return m + float(np.log(np.sum(np.exp(a[finite] - m), dtype=float)))
 
def _oracle_select_colocalisation_edges(
    log_evidence: "np.ndarray",
    coverage: "np.ndarray",
    overlap_min: float,
    h4_threshold: float,
) -> "np.ndarray":
    e = np.asarray(log_evidence, dtype=float)
    c = np.asarray(coverage, dtype=float)
    if e.ndim != 3 or e.shape[0] < 1 or e.shape[0] != e.shape[1] or e.shape[2] != 5:
        raise ValueError("log_evidence must have shape (n, n, 5)")
    if c.shape != (e.shape[0], e.shape[1], 2):
        raise ValueError("coverage must have shape (n, n, 2)")
    if np.any(np.isnan(e)) or np.any(np.isposinf(e)):
        raise ValueError("log evidence may contain negative infinity but not NaN or positive infinity")
    if not np.all(np.isfinite(c)) or np.any(c < 0.0) or np.any(c > 1.0 + 1e-12):
        raise ValueError("coverage must be finite and lie in [0,1]")
    thresholds = np.asarray([overlap_min, h4_threshold], dtype=float)
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0.0) or np.any(thresholds > 1.0):
        raise ValueError("thresholds must lie in [0,1]")
    rows = []
    for i in range(e.shape[0]):
        for j in range(i + 1, e.shape[0]):
            if min(c[i, j, 0], c[i, j, 1]) < float(overlap_min):
                continue
            normalizer = _edge_lse(e[i, j])
            posterior = 0.0 if not np.isfinite(e[i, j, 4]) else float(np.exp(e[i, j, 4] - normalizer))
            if posterior >= float(h4_threshold):
                rows.append((float(i), float(j), posterior))
    return np.asarray(rows, dtype=float).reshape((-1, 3))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
e=np.full((3,3,5),-np.inf); e[:,:,0]=0.; e[0,1,4]=e[1,0,4]=10.; e[1,2,4]=e[2,1,4]=2.
c=np.ones((3,3,2))""",
            "call": "select_colocalisation_edges(e,c,0.5,0.8)",
            "gold_call": "_oracle_select_colocalisation_edges(e,c,0.5,0.8)",
        },
        {
            "setup": """import numpy as np
# Inclusive coverage at 0.5; H4 posterior = 0.9 (not exactly 0.8) so the edge is stable across float widths.
e=np.full((2,2,5),-np.inf)
e[0,1,0]=e[1,0,0]=0.0
e[0,1,4]=e[1,0,4]=np.log(9.0)
c=np.ones((2,2,2))
c[0,1]=[0.5, 0.5]
""",
            "call": "select_colocalisation_edges(e,c,0.5,0.8)",
            "gold_call": "_oracle_select_colocalisation_edges(e,c,0.5,0.8)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
e=np.zeros((2,2,5)); c=np.zeros((2,2,2))""",
            "call": "select_colocalisation_edges(e,c,0.5,0.8)",
            "gold_call": "_oracle_select_colocalisation_edges(e,c,0.5,0.8)",
        },
        {
            "setup": """import numpy as np
e=np.zeros((2,3,5)); c=np.ones((2,2,2))
def candidate_code():
    try: select_colocalisation_edges(e,c,.5,.8); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_select_colocalisation_edges(e,c,.5,.8); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
