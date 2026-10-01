"""
Spawns a new daughter cell.

Cells divide at a slow, fixed rate. A new cell is placed at the equilibrium distance r* from the parent cell, allowing the system to mechanically relax between divisions.

Returns
-------
new_x : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def cell_division(pos: np.ndarray, angles: np.ndarray, n_cells: int, r_star: float, seed: int) -> float:
    '''
    Notes
    -----
    Modifies pos and angles in place.

    Parameters
    ----------
    pos
        Array of shape (max_cells, 2) for spatial coordinates.
    angles
        Array of shape (max_cells,) for polarity angles.
    n_cells
        Current number of cells.
    r_star
        Equilibrium distance for new cell placement.
    seed
        Random seed for reproducibility.

    Returns
    -------
    float
        The x-coordinate of the newly spawned cell
    '''   
    return float(new_x)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cell_division(pos: np.ndarray, angles: np.ndarray, n_cells: int, r_star: float, seed: int) -> float:
    import numpy as np
    np.random.seed(seed)
    parent_idx = np.random.randint(0, n_cells)
    spatial_angle = np.random.uniform(0, 2 * np.pi)
    polarity_angle = np.random.uniform(0, 2 * np.pi)
    
    new_x = pos[parent_idx, 0] + r_star * np.cos(spatial_angle)
    new_y = pos[parent_idx, 1] + r_star * np.sin(spatial_angle)
    
    pos[n_cells, 0] = new_x
    pos[n_cells, 1] = new_y
    angles[n_cells] = polarity_angle
    
    return float(new_x)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np\npos = np.zeros((2, 2))\nangles = np.zeros(2)\nn_cells = 1\nr_star = 2.0\nseed = 42""",
            "call": "cell_division(pos, angles, n_cells, r_star, seed)",
            "gold_call": "_oracle_cell_division(pos, angles, n_cells, r_star, seed)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np\npos = np.zeros((10, 2))\nangles = np.zeros(10)\nn_cells = 9\nr_star = 2.0\nseed = 0""",
            "call": "cell_division(pos, angles, n_cells, r_star, seed)",
            "gold_call": "_oracle_cell_division(pos, angles, n_cells, r_star, seed)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np\npos = np.zeros((5, 2))\nangles = np.zeros(5)\nn_cells = 1\nr_star = 0.0\nseed = 99""",
            "call": "cell_division(pos, angles, n_cells, r_star, seed)",
            "gold_call": "_oracle_cell_division(pos, angles, n_cells, r_star, seed)"
        }
    ]
