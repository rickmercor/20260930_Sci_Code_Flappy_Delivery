"""
Select article-eligible association signals before any pairwise analysis.

 

For this benchmark's deterministic reconciliation with the pinned active-trim code, a cell is available for screening only when its collection tests that variant, the supplied value is finite, and the value is greater than -100000. A signal is retained when the maximum available log Bayes factor is at least 5. Preserve original signal order. NaN denotes an unavailable cell; infinity is invalid.

Weak association signals can destabilize pairwise catalogue analysis; screening defines which signals enter inference without combining evidence across variants.

Returns
-------
retained_indices : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screen_signal_indices(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
) -> "np.ndarray":
    """
    Return indices of signals meeting the source eligibility screen.
 
    Parameters
    ----------
    log_bf : np.ndarray
        Two-dimensional signal-by-variant log Bayes-factor array.
    collection_id : np.ndarray
        One-dimensional zero-based collection identifier per signal.
    tested_variant_mask : np.ndarray
        Boolean collection-by-variant test-coverage array.
 
    Returns
    -------
    np.ndarray
        One-dimensional integer indices in original signal order.
 
    Raises
    ------
    ValueError
        If shapes, identifiers, dtypes, or finite-value requirements fail.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_screen_signal_indices(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
) -> "np.ndarray":
    x = np.asarray(log_bf, dtype=float)
    c_raw = np.asarray(collection_id)
    t_raw = np.asarray(tested_variant_mask)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("log_bf must be a non-empty two-dimensional array")
    if c_raw.ndim != 1 or c_raw.shape[0] != x.shape[0]:
        raise ValueError("collection_id must align with signal rows")
    if c_raw.dtype.kind not in "iu" or np.any(c_raw < 0):
        raise ValueError("collection_id must contain non-negative integers")
    if t_raw.ndim != 2 or t_raw.shape[1] != x.shape[1] or t_raw.shape[0] < 1:
        raise ValueError("tested_variant_mask must be collection by variant")
    if t_raw.dtype.kind != "b":
        raise ValueError("tested_variant_mask must be boolean")
    if np.any(c_raw >= t_raw.shape[0]):
        raise ValueError("collection_id is out of range")
    c = c_raw.astype(int, copy=False)
    if np.any(np.isinf(x)):
        raise ValueError("log_bf cannot contain infinity")
    keep = []
    for i in range(x.shape[0]):
        available = t_raw[c[i]] & np.isfinite(x[i]) & (x[i] > -100000.0)
        if np.any(available) and float(np.max(x[i, available])) >= 5.0:
            keep.append(i)
    return np.asarray(keep, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
x=np.array([[6.,-2.],[4.,7.],[4.9,4.8]])
c=np.array([0,0,0],dtype=int)
t=np.array([[True,True]],dtype=bool)""",
            "call": "screen_signal_indices(x,c,t)",
            "gold_call": "_oracle_screen_signal_indices(x,c,t)",
        },
        {
            "setup": """import numpy as np
x=np.array([[5.,-100000.],[4.999999,9.]])
c=np.array([0,1],dtype=int)
t=np.array([[True,True],[True,False]],dtype=bool)""",
            "call": "screen_signal_indices(x,c,t)",
            "gold_call": "_oracle_screen_signal_indices(x,c,t)",
        },
        {
            "setup": """import numpy as np
x=np.array([[np.nan,8.],[7.,np.nan],[np.nan,np.nan]])
c=np.array([0,1,0],dtype=int)
t=np.array([[True,True],[False,True]],dtype=bool)""",
            "call": "screen_signal_indices(x,c,t)",
            "gold_call": "_oracle_screen_signal_indices(x,c,t)",
        },
        {
            "setup": """import numpy as np
x=np.ones((2,3)); c=np.array([0,2]); t=np.ones((2,3),dtype=bool)
def candidate_code():
    try: screen_signal_indices(x,c,t); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_screen_signal_indices(x,c,t); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
