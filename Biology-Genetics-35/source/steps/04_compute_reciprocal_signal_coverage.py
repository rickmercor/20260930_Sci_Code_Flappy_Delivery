"""
Compute directional Bayes-factor-mass coverage for every retained signal pair.

 

For direction i to j, the numerator is signal i's Bayes-factor mass over collection-shared variants observed in both signals. The denominator is signal i's mass over all variants observed in i. Use stable log-sum-exp. Store i-to-j and j-to-i in the last axis; a pair with no mutually observed shared variant has zero coverage in both directions.

Association-mass coverage is directional when variant availability differs between signals.

Returns
-------
coverage : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reciprocal_signal_coverage(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
    shared_masks: "np.ndarray",
) -> "np.ndarray":
    """
    Compute both directional coverage fractions for all signal pairs.
 
    Parameters
    ----------
    prepared : np.ndarray
        Two-plane retained signal matrix.
    retained_collection_id : np.ndarray
        Collection identifier per retained signal.
    shared_masks : np.ndarray
        Collection-pair shared-variant masks.
 
    Returns
    -------
    np.ndarray
        Float array of shape (n_signals, n_signals, 2).
 
    Raises
    ------
    ValueError
        If aligned shapes, masks, or observed values are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _coverage_logsumexp(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return -np.inf
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m), dtype=float)))
 
def _oracle_reciprocal_signal_coverage(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
    shared_masks: "np.ndarray",
) -> "np.ndarray":
    p = np.asarray(prepared, dtype=float)
    c_raw = np.asarray(retained_collection_id)
    s_raw = np.asarray(shared_masks)
    if p.ndim != 3 or p.shape[0] != 2 or p.shape[1] < 1 or p.shape[2] < 1:
        raise ValueError("prepared must have shape (2, signals, variants)")
    if not np.all(np.isfinite(p)):
        raise ValueError("prepared must be finite")
    obs = p[1]
    if np.any((obs != 0.0) & (obs != 1.0)) or np.any(np.sum(obs, axis=1) == 0.0):
        raise ValueError("each signal must have a valid binary observation row")
    if c_raw.ndim != 1 or c_raw.shape[0] != p.shape[1] or c_raw.dtype.kind not in "iu":
        raise ValueError("retained_collection_id must be aligned integers")
    if np.any(c_raw < 0):
        raise ValueError("collection identifiers must be non-negative")
    unique_raw = np.unique(c_raw)
    if not np.array_equal(unique_raw, np.arange(unique_raw.size, dtype=unique_raw.dtype)):
        raise ValueError("collection identifiers must be contiguous from zero")
    c = c_raw.astype(int, copy=False)
    n_collections = unique_raw.size
    if s_raw.ndim != 3 or s_raw.shape != (n_collections, n_collections, p.shape[2]):
        raise ValueError("shared_masks has the wrong shape")
    if s_raw.dtype.kind != "b" or not np.array_equal(s_raw, np.swapaxes(s_raw, 0, 1)):
        raise ValueError("shared_masks must be boolean and symmetric")
    n = p.shape[1]
    den = np.array([_coverage_logsumexp(p[0, i, obs[i] == 1.0]) for i in range(n)])
    out = np.zeros((n, n, 2), dtype=float)
    for i in range(n):
        for j in range(n):
            common = s_raw[c[i], c[j]] & (obs[i] == 1.0) & (obs[j] == 1.0)
            if np.any(common):
                out[i, j, 0] = np.exp(_coverage_logsumexp(p[0, i, common]) - den[i])
                out[i, j, 1] = np.exp(_coverage_logsumexp(p[0, j, common]) - den[j])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
p=np.array([[[0.,0.,-1e6],[0.,np.log(3.),0.]],[[1.,1.,0.],[1.,1.,1.]]])
c=np.array([0,0],dtype=int); s=np.ones((1,1,3),dtype=bool)""",
            "call": "reciprocal_signal_coverage(p,c,s)",
            "gold_call": "_oracle_reciprocal_signal_coverage(p,c,s)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[1000.,-1e6],[-1e6,1000.]],[[1.,0.],[0.,1.]]])
c=np.array([0,1],dtype=int); s=np.ones((2,2,2),dtype=bool)""",
            "call": "reciprocal_signal_coverage(p,c,s)",
            "gold_call": "_oracle_reciprocal_signal_coverage(p,c,s)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[2.,1.],[3.,4.]],[[1.,1.],[1.,1.]]]); p[0]+=700.
c=np.array([0,0],dtype=int); s=np.ones((1,1,2),dtype=bool)""",
            "call": "reciprocal_signal_coverage(p,c,s)",
            "gold_call": "_oracle_reciprocal_signal_coverage(p,c,s)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[0.,0.]],[[0.,0.]]]); c=np.array([0]); s=np.ones((1,1,2),dtype=bool)
def candidate_code():
    try: reciprocal_signal_coverage(p,c,s); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_reciprocal_signal_coverage(p,c,s); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
