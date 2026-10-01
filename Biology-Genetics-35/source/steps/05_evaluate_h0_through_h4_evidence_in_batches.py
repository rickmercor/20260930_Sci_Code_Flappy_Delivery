"""
Evaluate source-matched H0-H4 log evidence for every signal pair.

 

Work on the collection-pair shared variant axis, including padded values. H0 has log evidence 0. H1 and H2 use each signal alone with prior p1 or p2. H3 uses prior p1*p2 and sums every ordered distinct-variant assignment. H4 uses prior p12 and sums equal-variant assignments. Use stable log-sum-exp throughout. chunk_size controls only outer signal-row blocking and cannot change results.

Bayesian colocalisation partitions evidence among no association, single-trait association, distinct-causal, and shared-causal explanations.

Returns
-------
log_evidence : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def batched_hypothesis_evidence(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
    shared_masks: "np.ndarray",
    p1: float,
    p2: float,
    p12: float,
    chunk_size: int,
) -> "np.ndarray":
    """
    Return stable pairwise log evidence for hypotheses H0 through H4.
 
    Parameters
    ----------
    prepared : np.ndarray
        Two-plane retained signal matrix.
    retained_collection_id : np.ndarray
        Collection identifier per retained signal.
    shared_masks : np.ndarray
        Collection-pair shared-variant masks.
    p1, p2, p12 : float
        Positive finite prior coefficients.
    chunk_size : int
        Positive outer-row block size.
 
    Returns
    -------
    np.ndarray
        Float array of shape (n_signals, n_signals, 5).
 
    Raises
    ------
    ValueError
        If inputs, priors, or chunk size violate the contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _evidence_lse(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    if a.size == 0:
        return -np.inf
    finite = np.isfinite(a)
    if not np.any(finite):
        return -np.inf
    m = float(np.max(a[finite]))
    return m + float(np.log(np.sum(np.exp(a[finite] - m), dtype=float)))
 
def _oracle_batched_hypothesis_evidence(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
    shared_masks: "np.ndarray",
    p1: float,
    p2: float,
    p12: float,
    chunk_size: int,
) -> "np.ndarray":
    p = np.asarray(prepared, dtype=float)
    c_raw = np.asarray(retained_collection_id)
    s_raw = np.asarray(shared_masks)
    if p.ndim != 3 or p.shape[0] != 2 or p.shape[1] < 1 or p.shape[2] < 1 or not np.all(np.isfinite(p)):
        raise ValueError("prepared must be finite with shape (2, signals, variants)")
    if np.any((p[1] != 0.0) & (p[1] != 1.0)) or np.any(np.sum(p[1], axis=1) == 0.0):
        raise ValueError("observation plane must be binary")
    if c_raw.ndim != 1 or c_raw.shape[0] != p.shape[1] or c_raw.dtype.kind not in "iu":
        raise ValueError("retained_collection_id must be aligned integers")
    if np.any(c_raw < 0):
        raise ValueError("collection identifiers must be non-negative")
    unique_raw = np.unique(c_raw)
    if not np.array_equal(unique_raw, np.arange(unique_raw.size, dtype=unique_raw.dtype)):
        raise ValueError("collection identifiers must be contiguous from zero")
    c = c_raw.astype(int, copy=False)
    nc = unique_raw.size
    if s_raw.shape != (nc, nc, p.shape[2]) or s_raw.dtype.kind != "b" or not np.array_equal(s_raw, np.swapaxes(s_raw, 0, 1)):
        raise ValueError("shared_masks must have the correct boolean symmetric shape")
    observed_values = p[0][p[1] == 1.0]
    if observed_values.size == 0 or np.any(np.abs(observed_values) > np.finfo(float).max / 2.0):
        raise ValueError("observed values must permit finite pairwise sums")
    priors = np.asarray([p1, p2, p12], dtype=float)
    if not np.all(np.isfinite(priors)) or np.any(priors <= 0.0):
        raise ValueError("prior coefficients must be positive and finite")
    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer")
    n = p.shape[1]
    out = np.full((n, n, 5), -np.inf, dtype=float)
    out[:, :, 0] = 0.0
    for block_start in range(0, n, int(chunk_size)):
        for i in range(block_start, min(block_start + int(chunk_size), n)):
            for j in range(n):
                shared = s_raw[c[i], c[j]]
                if not np.any(shared):
                    continue
                a = p[0, i, shared]
                b = p[0, j, shared]
                h1_core = _evidence_lse(a)
                h2_core = _evidence_lse(b)
                h4_core = _evidence_lse(a + b)
                cross = a[:, None] + b[None, :]
                distinct = cross[~np.eye(a.size, dtype=bool)]
                h3_core = _evidence_lse(distinct)
                out[i, j, 1] = np.log(float(p1)) + h1_core
                out[i, j, 2] = np.log(float(p2)) + h2_core
                if np.isfinite(h3_core):
                    out[i, j, 3] = np.log(float(p1)) + np.log(float(p2)) + h3_core
                out[i, j, 4] = np.log(float(p12)) + h4_core
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
p=np.array([[[4.,2.],[3.,5.]],[[1.,1.],[1.,1.]]]); c=np.array([0,0]); s=np.ones((1,1,2),dtype=bool)""",
            "call": "batched_hypothesis_evidence(p,c,s,1e-4,2e-4,1e-6,1)",
            "gold_call": "_oracle_batched_hypothesis_evidence(p,c,s,1e-4,2e-4,1e-6,1)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[1000.,999.,-1e6],[1000.,998.,-1e6]],[[1.,1.,0.],[1.,1.,0.]]]); c=np.array([0,0]); s=np.ones((1,1,3),dtype=bool)""",
            "call": "batched_hypothesis_evidence(p,c,s,1e-4,1e-4,1e-6,7)",
            "gold_call": "_oracle_batched_hypothesis_evidence(p,c,s,1e-4,1e-4,1e-6,7)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[6.],[7.]],[[1.],[1.]]]); c=np.array([0,0]); s=np.ones((1,1,1),dtype=bool)""",
            "call": "batched_hypothesis_evidence(p,c,s,1e-4,1e-4,1e-6,2)",
            "gold_call": "_oracle_batched_hypothesis_evidence(p,c,s,1e-4,1e-4,1e-6,2)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[1.,2.]],[[1.,1.]]]); c=np.array([0]); s=np.ones((1,1,2),dtype=bool)
def candidate_code():
    try: batched_hypothesis_evidence(p,c,s,1e-4,1e-4,0.,1); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_batched_hypothesis_evidence(p,c,s,1e-4,1e-4,0.,1); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
