"""
Runs the full morphogenesis simulation.

This orchestrator integrates all the physical rules: cell-cell adhesion, apico-basal polarity, contact inhibition, and cell division. By tuning the polarity strength (p) and the timescale of polarity reorientation (tau_B), the model can generate diverse morphological phases such as monolayers, multilayers, and cavities formed by wraparound or inflation.

Returns
-------
final_metric : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def simulate_morphogenesis(max_cells: int, p: float, tau_B: float, beta: float, dt: float, tau_V: float, tau_div: int, r_max: float, r_star: float, seed: int) -> float:
    '''
    Notes
    -----
    End-to-end pipeline.

    Parameters
    ----------
    max_cells
        Target number of cells to simulate.
    p
        Polarity strength.
    tau_B
        Contact inhibition timescale.
    beta
        Potential range parameter.
    dt
        Time step size.
    tau_V
        Relaxation timescale.
    tau_div
        Number of time steps between cell divisions.
    r_max
        Maximum interaction radius.
    r_star
        Equilibrium distance for new cell placement.
    seed
        Random seed for reproducibility.

    Returns
    -------
    float
        The final structural classification metric (average distance from center of mass).
    '''
    return classify_structure(pos, n_cells)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_simulate_morphogenesis(max_cells: int, p: float, tau_B: float, beta: float, dt: float, tau_V: float, tau_div: int, r_max: float, r_star: float, seed: int) -> float:
    import numpy as np
    
    pos = np.zeros((max_cells, 2))
    angles = np.zeros(max_cells)
    dist_matrix = np.zeros((max_cells, max_cells))
    r_hat_matrix = np.zeros((max_cells, max_cells, 2))
    forces = np.zeros((max_cells, 2))
    torques = np.zeros(max_cells)
    
    # Chain the oracle sub-steps
    _oracle_initialize_system(pos, angles, seed)
    n_cells = 1
    
    step = 0
    while n_cells < max_cells:
        _oracle_compute_distances_and_neighbors(pos, n_cells, dist_matrix, r_hat_matrix)
        _oracle_compute_forces_and_torques(pos, angles, dist_matrix, r_hat_matrix, n_cells, p, beta, tau_V, tau_B, r_max, forces, torques)
        _oracle_update_states(pos, angles, forces, torques, n_cells, dt)
        
        step += 1
        if step % tau_div == 0:
            _oracle_cell_division(pos, angles, n_cells, r_star, seed + step)
            n_cells += 1 # Explicitly increment here
            
    return _oracle_classify_structure(pos, n_cells)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Integration normal scenario ---
        # Simulates a small organoid growth to 3 cells.
        {
            "setup": """import numpy as np\nmax_cells = 3\np = 1.0\ntau_B = 1.0\nbeta = 0.2\ndt = 0.01\ntau_V = 10.0\ntau_div = 10\nr_max = 2.5\nr_star = 2.0\nseed = 42""",
            "call": "simulate_morphogenesis(max_cells, p, tau_B, beta, dt, tau_V, tau_div, r_max, r_star, seed)",
            "gold_call": "_oracle_simulate_morphogenesis(max_cells, p, tau_B, beta, dt, tau_V, tau_div, r_max, r_star, seed)"
        },
        # --- Integration boundary scenario ---
        # Simulates immediate termination (max_cells = 1).
        {
            "setup": """import numpy as np\nmax_cells = 1\np = 0.5\ntau_B = 0.5\nbeta = 0.1\ndt = 0.1\ntau_V = 1.0\ntau_div = 5\nr_max = 2.5\nr_star = 2.0\nseed = 0""",
            "call": "simulate_morphogenesis(max_cells, p, tau_B, beta, dt, tau_V, tau_div, r_max, r_star, seed)",
            "gold_call": "_oracle_simulate_morphogenesis(max_cells, p, tau_B, beta, dt, tau_V, tau_div, r_max, r_star, seed)"
        },
        # --- Integration edge scenario ---
        # Simulates growth with zero polarity and zero contact inhibition (pure Morse potential).
        {
            "setup": """import numpy as np\nmax_cells = 2\np = 0.0\ntau_B = 0.0\nbeta = 0.2\ndt = 0.05\ntau_V = 5.0\ntau_div = 2\nr_max = 2.5\nr_star = 2.0\nseed = 99""",
            "call": "simulate_morphogenesis(max_cells, p, tau_B, beta, dt, tau_V, tau_div, r_max, r_star, seed)",
            "gold_call": "_oracle_simulate_morphogenesis(max_cells, p, tau_B, beta, dt, tau_V, tau_div, r_max, r_star, seed)"
        }
    ]
