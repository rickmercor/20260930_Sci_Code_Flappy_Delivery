"""
Classifies the final structure by computing the average distance from the center of mass.

Different morphological phases (e.g., monolayer vs multilayer, cavity vs compact) can be distinguished by their spatial distribution. The average distance from the center of mass serves as a proxy for cavity size.

Returns
-------
avg_distance : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def classify_structure(pos: np.ndarray, n_cells: int) -> float:
    '''
    Notes
    -----
    Does not modify input arrays.

    Parameters
    ----------
    pos
        Array of shape (max_cells, 2) for spatial coordinates.
    n_cells
        Current number of cells.

    Returns
    -------
    float
        Average distance from the center of mass.
    '''        
    return float(total_dist / n_cells)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_classify_structure(pos: np.ndarray, n_cells: int) -> float:
    import numpy as np
    if n_cells == 0:
        return 0.0
    com_x = np.mean(pos[:n_cells, 0])
    com_y = np.mean(pos[:n_cells, 1])
    
    total_dist = 0.0
    for i in range(n_cells):
        dx = pos[i, 0] - com_x
        dy = pos[i, 1] - com_y
        total_dist += np.sqrt(dx*dx + dy*dy)
        
    return float(total_dist / n_cells)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0], [2.0, 0.0]])\nn_cells = 2""",
            "call": "classify_structure(pos, n_cells)",
            "gold_call": "_oracle_classify_structure(pos, n_cells)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0]])\nn_cells = 1""",
            "call": "classify_structure(pos, n_cells)",
            "gold_call": "_oracle_classify_structure(pos, n_cells)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0], [0.0, 0.0]])\nn_cells = 0""",
            "call": "classify_structure(pos, n_cells)",
            "gold_call": "_oracle_classify_structure(pos, n_cells)"
        }
    ]
