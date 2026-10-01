"""
Initializes the first cell at the origin with a random polarity angle.

The simulation begins with a single cell (zygote) placed at the origin. Its apico-basal polarity is initialized randomly, representing the initial symmetry breaking of the embryo.

Returns
-------
initial_angle : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def initialize_system(pos: np.ndarray, angles: np.ndarray, seed: int) -> float:
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
    seed
        Random seed for reproducibility.

    Returns
    -------
    float
        The initial angle of the first cell.
    '''
    return float(angles[0])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_initialize_system(pos: np.ndarray, angles: np.ndarray, seed: int) -> float:
    import numpy as np
    np.random.seed(seed)
    pos[0, 0] = 0.0
    pos[0, 1] = 0.0
    angles[0] = np.random.uniform(0, 2 * np.pi)
    return float(angles[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np\npos = np.zeros((10, 2))\nangles = np.zeros(10)\nseed = 42""",
            "call": "initialize_system(pos, angles, seed)",
            "gold_call": "_oracle_initialize_system(pos, angles, seed)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np\npos = np.zeros((1, 2))\nangles = np.zeros(1)\nseed = 0""",
            "call": "initialize_system(pos, angles, seed)",
            "gold_call": "_oracle_initialize_system(pos, angles, seed)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np\npos = np.zeros((100, 2))\nangles = np.zeros(100)\nseed = 4294967290""",
            "call": "initialize_system(pos, angles, seed)",
            "gold_call": "_oracle_initialize_system(pos, angles, seed)"
        }
    ]
