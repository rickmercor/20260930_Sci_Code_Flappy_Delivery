"""
Derive collection-pair variant universes from retained signal availability.

 

A collection's universe contains every variant observed in at least one retained signal from that collection. The mask for an ordered collection pair is the intersection of the two collection universes. Collection identifiers must be contiguous from zero, and every collection must be represented.

Studies may cover different variant universes; group-level harmonization determines which coordinates can be compared.

Returns
-------
shared_masks : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def collection_shared_masks(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
) -> "np.ndarray":
    """
    Return collection-pair intersections of collection-level variant unions.
 
    Parameters
    ----------
    prepared : np.ndarray
        Two-plane retained signal matrix.
    retained_collection_id : np.ndarray
        Zero-based collection identifier per retained signal.
 
    Returns
    -------
    np.ndarray
        Boolean array of shape (n_collections, n_collections, n_variants).
 
    Raises
    ------
    ValueError
        If planes, values, or collection identifiers are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_collection_shared_masks(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
) -> "np.ndarray":
    p = np.asarray(prepared, dtype=float)
    c_raw = np.asarray(retained_collection_id)
    if p.ndim != 3 or p.shape[0] != 2 or p.shape[1] < 1 or p.shape[2] < 1:
        raise ValueError("prepared must have shape (2, signals, variants)")
    if c_raw.ndim != 1 or c_raw.shape[0] != p.shape[1] or c_raw.dtype.kind not in "iu":
        raise ValueError("retained_collection_id must be an aligned integer vector")
    if not np.all(np.isfinite(p)):
        raise ValueError("prepared planes must be finite")
    observed = p[1]
    if np.any((observed != 0.0) & (observed != 1.0)):
        raise ValueError("observation plane must contain only zero and one")
    if np.any(c_raw < 0):
        raise ValueError("collection identifiers must be non-negative")
    unique_raw = np.unique(c_raw)
    if not np.array_equal(unique_raw, np.arange(unique_raw.size, dtype=unique_raw.dtype)):
        raise ValueError("collection identifiers must be contiguous from zero")
    c = c_raw.astype(int, copy=False)
    unique = np.unique(c)
    universe = np.zeros((unique.size, p.shape[2]), dtype=bool)
    for k in range(unique.size):
        universe[k] = np.any(observed[c == k] == 1.0, axis=0)
        if not np.any(universe[k]):
            raise ValueError("every collection needs at least one observed variant")
    return universe[:, None, :] & universe[None, :, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
p=np.array([[[1.,2.,-1e6],[3.,-1e6,4.],[5.,6.,-1e6]],[[1.,1.,0.],[1.,0.,1.],[1.,1.,0.]]])
c=np.array([0,0,1],dtype=int)""",
            "call": "collection_shared_masks(p,c)",
            "gold_call": "_oracle_collection_shared_masks(p,c)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[2.]],[[1.]]]); c=np.array([0],dtype=int)""",
            "call": "collection_shared_masks(p,c)",
            "gold_call": "_oracle_collection_shared_masks(p,c)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[1.,-1e6],[-1e6,2.]],[[1.,0.],[0.,1.]]]); c=np.array([0,1],dtype=int)""",
            "call": "collection_shared_masks(p,c)",
            "gold_call": "_oracle_collection_shared_masks(p,c)",
        },
        {
            "setup": """import numpy as np
p=np.array([[[1.,2.],[3.,4.]],[[1.,1.],[1.,1.]]]); c=np.array([0,2],dtype=int)
def candidate_code():
    try: collection_shared_masks(p,c); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_collection_shared_masks(p,c); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
