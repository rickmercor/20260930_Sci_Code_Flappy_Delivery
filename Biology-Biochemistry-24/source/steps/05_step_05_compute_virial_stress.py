"""
Calculate the trace of the Virial stress tensor for a cell.

The virial stress averages the discrete point forces acting at the vertices over the area of the cell. This bridges the discrete and continuum scales, outputting a continuum-like stress tensor.

Returns
-------
stress_trace : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def compute_virial_stress(vertices: np.ndarray, forces: np.ndarray) -> float:
    '''
    Notes
    -----
    Computes the trace of the stress tensor.
 
    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (V, 2) containing the x, y coordinates of the cell vertices.
    forces : np.ndarray
        Array of shape (V, 2) containing the forces on the vertices.
 
    Returns
    -------
    float
        Trace of the Virial stress tensor.
 
    Raises
    ------
    ValueError
        If `vertices` or `forces` is not an (n, 2) array of at least 3 finite
        rows, or if their shapes differ.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_virial_stress(vertices: np.ndarray, forces: np.ndarray) -> float:
    import numpy as np
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
    forces = _poly("forces", forces)
    if forces.shape != vertices.shape:
        raise ValueError(f"forces must match vertices in shape, got {forces.shape} and "
                         f"{vertices.shape}")
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if area == 0:
        return 0.0
    
    sigma_xx = np.sum(vertices[:, 0] * forces[:, 0]) / area
    sigma_yy = np.sum(vertices[:, 1] * forces[:, 1]) / area
    
    return float(sigma_xx + sigma_yy)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
        return [
            # --- Normal scenario ---
            {
                "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
forces = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])""",
                "call": "compute_virial_stress(vertices, forces)",
                "gold_call": "_oracle_compute_virial_stress(vertices, forces)"
            },
            # --- Boundary case ---
            {
                "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
forces = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])""",
                "call": "compute_virial_stress(vertices, forces)",
                "gold_call": "_oracle_compute_virial_stress(vertices, forces)"
            },
            # --- Edge case ---
            {
                "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
forces = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])""",
                "call": "compute_virial_stress(vertices, forces)",
                "gold_call": "_oracle_compute_virial_stress(vertices, forces)"
            }
        ]
