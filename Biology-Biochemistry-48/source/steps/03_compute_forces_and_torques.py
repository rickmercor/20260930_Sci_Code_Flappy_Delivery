"""
Computes spatial forces and angular torques for each cell.

The system evolves via overdamped Langevin dynamics. Forces are derived from the spatial gradient of the potential (computed via finite difference). Torques arise from adhesion alignment and contact inhibition (computed analytically).

Returns
-------
total_force_torque : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_forces_and_torques(pos: np.ndarray, angles: np.ndarray, dist_matrix: np.ndarray, r_hat_matrix: np.ndarray, n_cells: int, p: float, beta: float, tau_V: float, tau_B: float, r_max: float, forces: np.ndarray, torques: np.ndarray) -> float:
    '''
    Notes
    -----
    Modifies forces and torques in place.

    Parameters
    ----------
    pos
        Array of shape (max_cells, 2) for spatial coordinates.
    angles
        Array of shape (max_cells,) for polarity angles.
    dist_matrix
        Array of shape (max_cells, max_cells) for distances.
    r_hat_matrix
        Array of shape (max_cells, max_cells, 2) for unit vectors.
    n_cells
        Current number of cells.
    p
        Polarity strength.
    beta
        Potential range parameter.
    tau_V
        Relaxation timescale.
    tau_B
        Contact inhibition timescale.
    r_max
        Maximum interaction radius.
    forces
        Array of shape (max_cells, 2) for forces.
    torques
        Array of shape (max_cells,) for torques.

    Returns
    -------
    float
        Sum of all forces and torques.
    '''
       
    return float(np.sum(forces[:n_cells]) + np.sum(torques[:n_cells]))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_forces_and_torques(pos: np.ndarray, angles: np.ndarray, dist_matrix: np.ndarray, r_hat_matrix: np.ndarray, n_cells: int, p: float, beta: float, tau_V: float, tau_B: float, r_max: float, forces: np.ndarray, torques: np.ndarray) -> float:
    import numpy as np

    def calc_V(x, y, theta, i):
        V = 0.0
        for j in range(n_cells):
            if i == j: continue
            dx = pos[j, 0] - x
            dy = pos[j, 1] - y
            r = np.sqrt(dx*dx + dy*dy)
            if r > r_max or r == 0: continue
            hx = dx / r
            hy = dy / r
            U = np.exp(-r) - np.exp(-r/beta)
            Ci = p * np.cos(theta) * hy - p * np.sin(theta) * hx
            Cj = p * np.cos(angles[j]) * hy - p * np.sin(angles[j]) * hx
            S = Ci * Cj + 1.0
            V += S * U
        return V

    if beta == 0.0:
        forces[:n_cells] = 0.0
        torques[:n_cells] = 0.0
        return 0.0

    eps = 1e-5
    for i in range(n_cells):
        vx_plus = calc_V(pos[i,0]+eps, pos[i,1], angles[i], i)
        vx_minus = calc_V(pos[i,0]-eps, pos[i,1], angles[i], i)
        vy_plus = calc_V(pos[i,0], pos[i,1]+eps, angles[i], i)
        vy_minus = calc_V(pos[i,0], pos[i,1]-eps, angles[i], i)
        
        forces[i, 0] = -tau_V * (vx_plus - vx_minus) / (2 * eps)
        forces[i, 1] = -tau_V * (vy_plus - vy_minus) / (2 * eps)
        
        t_adhesion = 0.0
        t_contact = 0.0
        for j in range(n_cells):
            if i == j: continue
            r = dist_matrix[i, j]
            if r > r_max or r == 0: continue
            hx = r_hat_matrix[i, j, 0]
            hy = r_hat_matrix[i, j, 1]
            
            U = np.exp(-r) - np.exp(-r/beta)
            Cj = p * np.cos(angles[j]) * hy - p * np.sin(angles[j]) * hx
            p_dot_r = np.cos(angles[i]) * hx + np.sin(angles[i]) * hy
            p_cross_r = np.cos(angles[i]) * hy - np.sin(angles[i]) * hx
            
            dV_dtheta = -p * p_dot_r * Cj * U
            t_adhesion += dV_dtheta
            t_contact += p_cross_r
            
        torques[i] = -tau_V * t_adhesion - tau_B * t_contact
        
    return float(np.sum(forces[:n_cells]) + np.sum(torques[:n_cells]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0], [1.0, 0.0]])\nangles = np.array([0.0, np.pi/2])\ndist_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])\nr_hat_matrix = np.array([[[0.0, 0.0], [1.0, 0.0]], [[-1.0, 0.0], [0.0, 0.0]]])\nn_cells = 2\np = 1.0\nbeta = 2.0\ntau_V = 1.0\ntau_B = 0.5\nr_max = 5.0\nforces = np.zeros((2, 2))\ntorques = np.zeros(2)\n""",
            "call": "compute_forces_and_torques(pos,angles, dist_matrix, r_hat_matrix, n_cells, p, beta, tau_V, tau_B, r_max, forces, torques)",
            "gold_call": "_oracle_compute_forces_and_torques(pos, angles, dist_matrix, r_hat_matrix, n_cells, p, beta, tau_V, tau_B, r_max, forces, torques)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0]])\nangles = np.array([0.0])\ndist_matrix = np.zeros((1, 1))\nr_hat_matrix = np.zeros((1, 1, 2))\nn_cells = 1\np = 1.0\nbeta = 0.2\ntau_V = 10.0\ntau_B = 1.0\nr_max = 2.5\nforces = np.zeros((1, 2))\ntorques = np.zeros(1)""",
            "call": "compute_forces_and_torques(pos, angles, dist_matrix, r_hat_matrix, n_cells, p, beta, tau_V, tau_B, r_max, forces, torques)",
            "gold_call": "_oracle_compute_forces_and_torques(pos, angles, dist_matrix, r_hat_matrix, n_cells, p, beta, tau_V, tau_B, r_max, forces, torques)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np\npos = np.array([[0.0, 0.0], [3.0, 0.0]])\nangles = np.array([0.0, 0.0])\ndist_matrix = np.array([[0.0, 3.0], [3.0, 0.0]])\nr_hat_matrix = np.array([[[0.0, 0.0], [1.0, 0.0]], [[-1.0, 0.0], [0.0, 0.0]]])\nn_cells = 2\np = 0.0\nbeta = 0.1\ntau_V = 1.0\ntau_B = 0.0\nr_max = 2.5\nforces = np.zeros((2, 2))\ntorques = np.zeros(2)""",
            "call": "compute_forces_and_torques(pos, angles, dist_matrix, r_hat_matrix, n_cells, p, beta, tau_V, tau_B, r_max, forces, torques)",
            "gold_call": "_oracle_compute_forces_and_torques(pos, angles, dist_matrix, r_hat_matrix, n_cells, p, beta, tau_V, tau_B, r_max, forces, torques)"
        }
    ]
