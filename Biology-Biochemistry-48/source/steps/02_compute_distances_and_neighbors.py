"""
Computes pairwise distances and unit vectors between all cells.

Cells interact mechanically with their neighbors. The distance and direction (unit vector) are required to compute the Morse-like potential and polarity-dependent adhesion.

Returns
-------
total_distance : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_distances_and_neighbors(pos: np.ndarray, n_cells: int, dist_matrix: np.ndarray, r_hat_matrix: np.ndarray) -> float:
    '''
    Notes
    -----
    Modifies dist_matrix and r_hat_matrix in place.

    Parameters
    ----------
    pos
        Array of shape (max_cells, 2) for spatial coordinates.
    n_cells
        Current number of cells.
    dist_matrix
        Array of shape (max_cells, max_cells) for distances.
    r_hat_matrix
        Array of shape (max_cells, max_cells, 2) for unit vectors.

    Returns
    -------
    float
        Sum of all pairwise distances.
    '''
    return float(np.sum(dist_matrix))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_distances_and_neighbors(pos: np.ndarray, n_cells: int, dist_matrix: np.ndarray, r_hat_matrix: np.ndarray) -> float:
    import numpy as np
    for i in range(n_cells):
        for j in range(n_cells):
            if i == j:
                dist_matrix[i, j] = 0.0
                r_hat_matrix[i, j, 0] = 0.0
                r_hat_matrix[i, j, 1] = 0.0
            else:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                r = np.sqrt(dx*dx + dy*dy)
                dist_matrix[i, j] = r
                if r > 0:
                    r_hat_matrix[i, j, 0] = dx / r
                    r_hat_matrix[i, j, 1] = dy / r
    return float(np.sum(dist_matrix))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0], [2.0, 0.0]])\nn_cells = 2\ndist_matrix = np.zeros((2, 2))\nr_hat_matrix = np.zeros((2, 2, 2))""",
            "call": "compute_distances_and_neighbors(pos, n_cells, dist_matrix, r_hat_matrix)",
            "gold_call": "_oracle_compute_distances_and_neighbors(pos, n_cells, dist_matrix, r_hat_matrix)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0]])\nn_cells = 1\ndist_matrix = np.zeros((1, 1))\nr_hat_matrix = np.zeros((1, 1, 2))""",
            "call": "compute_distances_and_neighbors(pos, n_cells, dist_matrix, r_hat_matrix)",
            "gold_call": "_oracle_compute_distances_and_neighbors(pos, n_cells, dist_matrix, r_hat_matrix)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0], [0.0, 0.0]])\nn_cells = 2\ndist_matrix = np.zeros((2, 2))\nr_hat_matrix = np.zeros((2, 2, 2))""",
            "call": "compute_distances_and_neighbors(pos, n_cells, dist_matrix, r_hat_matrix)",
            "gold_call": "_oracle_compute_distances_and_neighbors(pos, n_cells, dist_matrix, r_hat_matrix)"
        }
    ]
