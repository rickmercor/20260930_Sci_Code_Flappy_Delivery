"""
Construct the benchmark-normalized retained signal matrix while preserving unavailable cells.

 

For each retained signal, a cell is observed only if the signal's collection tests the variant, the supplied value is finite, and the value is greater than -100000. Return two float planes: plane 0 contains observed log Bayes factors and -1000000 elsewhere; plane 1 contains the corresponding 1/0 observation indicator. Retained indices must be strictly increasing.

Missing variant measurements and genuinely weak observed evidence have different meanings; an explicit availability representation preserves this distinction.

Returns
-------
prepared : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prepare_signal_matrix(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
    retained_indices: "np.ndarray",
) -> "np.ndarray":
    """
    Build benchmark values and availability planes for retained signals.
 
    Parameters
    ----------
    log_bf : np.ndarray
        Signal-by-variant log Bayes factors, with NaN for missing cells.
    collection_id : np.ndarray
        Zero-based collection identifier per signal.
    tested_variant_mask : np.ndarray
        Boolean collection-by-variant coverage mask.
    retained_indices : np.ndarray
        Strictly increasing retained signal indices.
 
    Returns
    -------
    np.ndarray
        Float array of shape (2, n_retained, n_variants).
 
    Raises
    ------
    ValueError
        If the aligned input contract is violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_prepare_signal_matrix(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
    retained_indices: "np.ndarray",
) -> "np.ndarray":
    x = np.asarray(log_bf, dtype=float)
    c_raw = np.asarray(collection_id)
    t_raw = np.asarray(tested_variant_mask)
    r_raw = np.asarray(retained_indices)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("log_bf must be non-empty and two-dimensional")
    if c_raw.ndim != 1 or c_raw.shape[0] != x.shape[0] or c_raw.dtype.kind not in "iu":
        raise ValueError("collection_id must be an aligned integer vector")
    if np.any(c_raw < 0):
        raise ValueError("collection identifiers must be non-negative")
    if t_raw.ndim != 2 or t_raw.shape[1] != x.shape[1] or t_raw.dtype.kind != "b":
        raise ValueError("tested_variant_mask must be a boolean collection-by-variant array")
    if np.any(c_raw >= t_raw.shape[0]):
        raise ValueError("collection identifier out of range")
    c = c_raw.astype(int, copy=False)
    if r_raw.ndim != 1 or r_raw.size < 1 or r_raw.dtype.kind not in "iu":
        raise ValueError("retained_indices must be a non-empty integer vector")
    r = r_raw.astype(int, copy=False)
    if np.any(r < 0) or np.any(r >= x.shape[0]) or np.any(np.diff(r) <= 0):
        raise ValueError("retained_indices must be strictly increasing and in range")
    if np.any(np.isinf(x)):
        raise ValueError("log_bf cannot contain infinity")
    values = np.full((r.size, x.shape[1]), -1000000.0, dtype=float)
    observed = np.zeros((r.size, x.shape[1]), dtype=float)
    for out_i, src_i in enumerate(r):
        mask = t_raw[c[src_i]] & np.isfinite(x[src_i]) & (x[src_i] > -100000.0)
        values[out_i, mask] = x[src_i, mask]
        observed[out_i, mask] = 1.0
    return np.stack((values, observed), axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
x=np.array([[6.,np.nan,-100000.],[1.,2.,3.],[8.,9.,10.]])
c=np.array([0,1,0],dtype=int)
t=np.array([[True,True,True],[True,False,True]],dtype=bool)
r=np.array([0,2],dtype=int)""",
            "call": "prepare_signal_matrix(x,c,t,r)",
            "gold_call": "_oracle_prepare_signal_matrix(x,c,t,r)",
        },
        {
            "setup": """import numpy as np
x=np.array([[5.],[6.]])
c=np.array([0,0],dtype=int); t=np.array([[True]],dtype=bool); r=np.array([1],dtype=int)""",
            "call": "prepare_signal_matrix(x,c,t,r)",
            "gold_call": "_oracle_prepare_signal_matrix(x,c,t,r)",
        },
        {
            "setup": """import numpy as np
x=np.array([[7.,8.],[9.,10.]])
c=np.array([0,1],dtype=int); t=np.array([[True,False],[False,True]],dtype=bool); r=np.array([0,1],dtype=int)""",
            "call": "prepare_signal_matrix(x,c,t,r)",
            "gold_call": "_oracle_prepare_signal_matrix(x,c,t,r)",
        },
        {
            "setup": """import numpy as np
x=np.ones((3,2)); c=np.zeros(3,dtype=int); t=np.ones((1,2),dtype=bool); r=np.array([2,1])
def candidate_code():
    try: prepare_signal_matrix(x,c,t,r); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_prepare_signal_matrix(x,c,t,r); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
