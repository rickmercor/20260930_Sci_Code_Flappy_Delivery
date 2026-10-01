"""
Calculate the force vector acting on every vertex of a cell.

Forces are derived from the negative gradient of the cell's potential energy with respect to vertex positions. This drives the overdamped vertex dynamics to a local mechanical equilibrium.

Returns
-------
forces : np.ndarray  Array of shape (V, 2) containing the force vector on each vertex
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def compute_vertex_forces(vertices: np.ndarray, kappa: float, chi: float) -> np.ndarray:
    '''
    Notes
    -----
    Computes the force vector acting on every vertex.
 
    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (V, 2) containing the x, y coordinates of the cell vertices.
    kappa : float
        Rigidity ratio.
    chi : float
        Preferred shape index.
 
    Returns
    -------
    np.ndarray
        Array of shape (V, 2) containing the x and y components of the
        force on each vertex.
 
    Raises
    ------
    ValueError
        If `vertices` is not an (n, 2) array of at least 3 finite rows, or if
        `kappa` or `chi` is not a finite real number.
    '''
    return np.zeros_like(vertices, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_vertex_forces(vertices: np.ndarray, kappa: float, chi: float) -> np.ndarray:
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
    kappa = _num("kappa", kappa)
    chi = _num("chi", chi)

    V = vertices.shape[0]
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    edge_lengths = np.sqrt(dx**2 + dy**2)
    perimeter = np.sum(edge_lengths)
    L = edge_lengths + 1e-12
    
    dE_dA = area - 1.0
    dE_dP = kappa * (perimeter - chi)
    
    forces = np.zeros_like(vertices)

    for i in range(V):
        prev_i = (i - 1) % V
        next_i = (i + 1) % V

        dA_dxi = 0.5 * (y[next_i] - y[prev_i])
        dA_dyi = 0.5 * (x[prev_i] - x[next_i])
        
        dP_dxi = (
            (x[i] - x[prev_i]) / L[prev_i]
            + (x[i] - x[next_i]) / L[i]
        )
        dP_dyi = (
            (y[i] - y[prev_i]) / L[prev_i]
            + (y[i] - y[next_i]) / L[i]
        )
        
        fx = -(dE_dA * dA_dxi + dE_dP * dP_dxi)
        fy = -(dE_dA * dA_dyi + dE_dP * dP_dyi)
        forces[i] = [fx, fy]
        
    return forces

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
vertices = np.column_stack((np.cos(angles), np.sin(angles)))
kappa = 1.0
chi = 3.81""",
            "call": "compute_vertex_forces(vertices, kappa, chi)",
            "gold_call": "_oracle_compute_vertex_forces(vertices, kappa, chi)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
kappa = 0.5
chi = 4.0""",
            "call": "compute_vertex_forces(vertices, kappa, chi)",
            "gold_call": "_oracle_compute_vertex_forces(vertices, kappa, chi)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
kappa = 10.0
chi = 1.0""",
            "call": "compute_vertex_forces(vertices, kappa, chi)",
            "gold_call": "_oracle_compute_vertex_forces(vertices, kappa, chi)"
        }
    ]
