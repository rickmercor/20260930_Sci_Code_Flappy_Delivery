#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def initialize_system(pos: np.ndarray, angles: np.ndarray, seed: int) -> float:
    import numpy as np
    np.random.seed(seed)
    pos[0, 0] = 0.0
    pos[0, 1] = 0.0
    angles[0] = np.random.uniform(0, 2 * np.pi)
    return float(angles[0])

def compute_distances_and_neighbors(pos: np.ndarray, n_cells: int, dist_matrix: np.ndarray, r_hat_matrix: np.ndarray) -> float:
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

def compute_forces_and_torques(pos: np.ndarray, angles: np.ndarray, dist_matrix: np.ndarray, r_hat_matrix: np.ndarray, n_cells: int, p: float, beta: float, tau_V: float, tau_B: float, r_max: float, forces: np.ndarray, torques: np.ndarray) -> float:
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

def update_states(pos: np.ndarray, angles: np.ndarray, forces: np.ndarray, torques: np.ndarray, n_cells: int, dt: float) -> float:
    import numpy as np
    for i in range(n_cells):
        pos[i, 0] += dt * forces[i, 0]
        pos[i, 1] += dt * forces[i, 1]
        angles[i] += dt * torques[i]
        angles[i] = angles[i] % (2 * np.pi)
    return float(np.sum(pos[:n_cells]) + np.sum(angles[:n_cells]))

def cell_division(pos: np.ndarray, angles: np.ndarray, n_cells: int, r_star: float, seed: int) -> float:
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

def classify_structure(pos: np.ndarray, n_cells: int) -> float:
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

def simulate_morphogenesis(max_cells: int, p: float, tau_B: float, beta: float, dt: float, tau_V: float, tau_div: int, r_max: float, r_star: float, seed: int) -> float:
    import numpy as np
    
    pos = np.zeros((max_cells, 2))
    angles = np.zeros(max_cells)
    dist_matrix = np.zeros((max_cells, max_cells))
    r_hat_matrix = np.zeros((max_cells, max_cells, 2))
    forces = np.zeros((max_cells, 2))
    torques = np.zeros(max_cells)
    
    # Chain the oracle sub-steps
    initialize_system(pos, angles, seed)
    n_cells = 1
    
    step = 0
    while n_cells < max_cells:
        compute_distances_and_neighbors(pos, n_cells, dist_matrix, r_hat_matrix)
        compute_forces_and_torques(pos, angles, dist_matrix, r_hat_matrix, n_cells, p, beta, tau_V, tau_B, r_max, forces, torques)
        update_states(pos, angles, forces, torques, n_cells, dt)
        
        step += 1
        if step % tau_div == 0:
            cell_division(pos, angles, n_cells, r_star, seed + step)
            n_cells += 1 # Explicitly increment here
            
    return classify_structure(pos, n_cells)
SCICODE_GOLD_EOF
