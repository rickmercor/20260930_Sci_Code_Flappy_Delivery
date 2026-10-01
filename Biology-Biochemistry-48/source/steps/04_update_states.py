"""
Integrates the state using the Euler method.

Updates positions and polarity angles based on the computed forces and torques over a time step dt.

Returns
-------
total_state_sum : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def update_states(pos: np.ndarray, angles: np.ndarray, forces: np.ndarray, torques: np.ndarray, n_cells: int, dt: float) -> float:
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
    forces
        Array of shape (max_cells, 2) for forces.
    torques
        Array of shape (max_cells,) for torques.
    n_cells
        Current number of cells.
    dt
        Time step size.

    Returns
    -------
    float
        Sum of all updated positions and angles.
    '''
    return float(np.sum(pos[:n_cells]) + np.sum(angles[:n_cells]))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_update_states(pos: np.ndarray, angles: np.ndarray, forces: np.ndarray, torques: np.ndarray, n_cells: int, dt: float) -> float:
    import numpy as np
    for i in range(n_cells):
        pos[i, 0] += dt * forces[i, 0]
        pos[i, 1] += dt * forces[i, 1]
        angles[i] += dt * torques[i]
        angles[i] = angles[i] % (2 * np.pi)
    return float(np.sum(pos[:n_cells]) + np.sum(angles[:n_cells]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0]])\nangles = np.array([0.0])\nforces = np.array([[1.0, -1.0]])\ntorques = np.array([0.5])\nn_cells = 1\ndt = 0.1""",
            "call": "update_states(pos.copy(), angles.copy(), forces, torques, n_cells, dt)",
            "gold_call": "_oracle_update_states(pos.copy(), angles.copy(), forces, torques, n_cells, dt)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0]])\nangles = np.array([2 * np.pi - 0.1])\nforces = np.array([[0.0, 0.0]])\ntorques = np.array([0.2])\nn_cells = 1\ndt = 1.0""",
            "call": "update_states(pos.copy(), angles.copy(), forces, torques, n_cells, dt)",
            "gold_call": "_oracle_update_states(pos.copy(), angles.copy(), forces, torques, n_cells, dt)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0]])\nangles = np.array([0.0])\nforces = np.array([[0.0, 0.0]])\ntorques = np.array([0.0])\nn_cells = 0\ndt = 0.1""",
            "call": "update_states(pos.copy(), angles.copy(), forces, torques, n_cells, dt)",
            "gold_call": "_oracle_update_states(pos.copy(), angles.copy(), forces, torques, n_cells, dt)"
        }
    ]
