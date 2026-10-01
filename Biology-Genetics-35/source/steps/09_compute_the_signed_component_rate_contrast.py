"""
Compute the signed percentage-point change in component collider fraction.

 

Each component table has canonical rows [minimum node, node count, edge count, collider flag] in strictly increasing minimum-node order. Both tables must contain at least one component. Return 100 times the restricted mean collider flag minus 100 times the full mean collider flag. Do not return a relative percent change.

A sensitivity contrast compares component-level rates under two graph definitions and depends on both numerator and denominator.

Returns
-------
contrast : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def collider_rate_contrast(
    full_components: "np.ndarray",
    restricted_components: "np.ndarray",
) -> float:
    """
    Return restricted-minus-full collider fraction in percentage points.
 
    Parameters
    ----------
    full_components : np.ndarray
        Full-graph component summary with four integer columns.
    restricted_components : np.ndarray
        Restricted-graph component summary with four integer columns.
 
    Returns
    -------
    float
        Signed percentage-point contrast.
 
    Raises
    ------
    ValueError
        If either component table is empty or structurally invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _validate_component_table_rate(table, name):
    raw = np.asarray(table)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] != 4 or raw.dtype.kind not in "iu":
        raise ValueError(name + " must be a non-empty integer array with four columns")
    x = raw.astype(int, copy=False)
    if np.any(x[:, 0] < 0) or np.any(x[:, 1] < 2) or np.any(x[:, 2] < 1):
        raise ValueError(name + " contains invalid component counts")
    if np.any(np.diff(x[:, 0]) <= 0):
        raise ValueError(name + " must be ordered by unique minimum node")
    if np.any(x[:, 2] < x[:, 1] - 1) or np.any(x[:, 2] > x[:, 1] * (x[:, 1] - 1) // 2):
        raise ValueError(name + " has impossible connected-component edge counts")
    if np.any((x[:, 3] != 0) & (x[:, 3] != 1)):
        raise ValueError(name + " collider flags must be binary")
    return x
 
def _oracle_collider_rate_contrast(
    full_components: "np.ndarray",
    restricted_components: "np.ndarray",
) -> float:
    full = _validate_component_table_rate(full_components, "full_components")
    restricted = _validate_component_table_rate(restricted_components, "restricted_components")
    value = 100.0 * float(np.mean(restricted[:, 3], dtype=float)) - 100.0 * float(np.mean(full[:, 3], dtype=float))
    if not np.isfinite(value):
        raise ValueError("contrast must be finite")
    return float(value)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
f=np.array([[0,4,3,1],[4,4,3,1],[8,3,2,1],[11,2,1,0],[13,2,1,0],[15,2,1,0],[17,2,1,0],[19,2,1,0]])
r=np.array([[0,4,3,1],[4,4,3,1],[6,2,1,0],[11,2,1,0]])""",
            "call": "collider_rate_contrast(f,r)",
            "gold_call": "_oracle_collider_rate_contrast(f,r)",
        },
        {
            "setup": """import numpy as np
f=np.array([[0,2,1,1],[2,2,1,1]]); r=np.array([[0,2,1,0],[2,2,1,1]])""",
            "call": "collider_rate_contrast(f,r)",
            "gold_call": "_oracle_collider_rate_contrast(f,r)",
        },
        {
            "setup": """import numpy as np
f=np.array([[0,2,1,0]]); r=np.array([[4,3,2,0]])""",
            "call": "collider_rate_contrast(f,r)",
            "gold_call": "_oracle_collider_rate_contrast(f,r)",
        },
        {
            "setup": """import numpy as np
f=np.empty((0,4),dtype=int); r=np.array([[0,2,1,0]])
def candidate_code():
    try: collider_rate_contrast(f,r); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_collider_rate_contrast(f,r); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
