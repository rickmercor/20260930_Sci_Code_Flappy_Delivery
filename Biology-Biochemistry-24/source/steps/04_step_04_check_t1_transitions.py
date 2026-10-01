"""
Check for topological transitions (T1 events) by identifying edges shorter than a threshold.

T1 transitions are inelastic cellular rearrangements where cells swap neighbors. They are triggered when an edge length falls below a critical threshold d_T, acting as a mechanism for stress relaxation.

Returns
-------
num_transitions : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def check_t1_transitions(vertices: np.ndarray, d_T: float) -> float:
    '''
    Notes
    -----
    Returns the number of edges below the threshold.
 
    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (V, 2) containing the x, y coordinates of the cell vertices.
    d_T : float
        Edge length threshold for T1 transitions.
 
    Returns
    -------
    float
        Number of edges shorter than d_T.
 
    Raises
    ------
    ValueError
        If `vertices` is not an (n, 2) array of at least 3 finite rows, or if
        `d_T` is not a finite real number.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_check_t1_transitions(vertices: np.ndarray, d_T: float) -> float:
    import numpy as np
    def _num(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value
    def _poly(name, value, min_vertices=3):
        value = np.asarray(value, dtype=float)
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2), got {value.shape}")
        if value.shape[0] < min_vertices:
            raise ValueError(f"{name} needs at least {min_vertices} rows, got {value.shape[0]}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        return value
    vertices = _poly("vertices", vertices)
    d_T = _num("d_T", d_T)
    x = vertices[:, 0]
    y = vertices[:, 1]
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    L = np.sqrt(dx**2 + dy**2)
    num_transitions = np.sum(L < d_T)
    return float(num_transitions)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
d_T = 0.5""",
            "call": "check_t1_transitions(vertices, d_T)",
            "gold_call": "_oracle_check_t1_transitions(vertices, d_T)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [0.05, 0.0], [0.05, 0.05], [0.0, 0.05]])
d_T = 0.1""",
            "call": "check_t1_transitions(vertices, d_T)",
            "gold_call": "_oracle_check_t1_transitions(vertices, d_T)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
d_T = 20.0""",
            "call": "check_t1_transitions(vertices, d_T)",
            "gold_call": "_oracle_check_t1_transitions(vertices, d_T)"
        }
    ]
