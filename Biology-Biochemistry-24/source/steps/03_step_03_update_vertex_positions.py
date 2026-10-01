"""
Update vertex positions using overdamped dynamics and return the maximum displacement.

Update vertex positions using overdamped dynamics and return the updated vertex coordinates.

 

Biological tissues exist in a low-Reynolds-number regime where viscous drag dominates inertia. The velocity of a cell vertex multiplied by viscous drag (gamma) is balanced by the mechanical forces.

 

Returns

-------

updated_vertices : np.ndarray

    Array of shape (V, 2) containing the updated vertex coordinates.

Returns
-------
pdated_vertices : np.ndarray    Array of shape (V, 2) containing the updated vertex coordinates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def update_vertex_positions(
    vertices: np.ndarray,
    forces: np.ndarray,
    gamma: float,
    dt: float
) -> np.ndarray:
    '''
    Notes
    -----
    Computes the updated vertex coordinates.
 
    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (V, 2) containing the x, y coordinates of the cell vertices.
    forces : np.ndarray
        Array of shape (V, 2) containing the forces on the vertices.
    gamma : float
        Viscous drag coefficient.
    dt : float
        Time step size.
 
    Returns
    -------
    np.ndarray
        Array of shape (V, 2) containing the updated vertex coordinates.
 
    Raises
    ------
    ValueError
        If `vertices` or `forces` is not an (n, 2) array of at least 3 finite
        rows, if their shapes differ, if `gamma` or `dt` is not a finite real
        number, or if `gamma` is zero.
    '''
    return np.zeros_like(vertices, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_update_vertex_positions(
    vertices: np.ndarray,
    forces: np.ndarray,
    gamma: float,
    dt: float
) -> np.ndarray:
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
    forces = _poly("forces", forces)

    if forces.shape != vertices.shape:
        raise ValueError(
            f"forces must match vertices in shape, got {forces.shape} and "
            f"{vertices.shape}"
        )

    gamma = _num("gamma", gamma)
    dt = _num("dt", dt)

    if gamma == 0.0:
        raise ValueError("gamma must be non-zero; the overdamped update divides by it")

    displacement = (forces / gamma) * dt
    return vertices + displacement

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal scenario ---
        {
            "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
forces = np.array([[0.1, 0.1], [-0.1, 0.0], [0.0, -0.1]])
gamma = 1.0
dt = 0.1""",
            "call": "update_vertex_positions(vertices, forces, gamma, dt)",
            "gold_call": "_oracle_update_vertex_positions(vertices, forces, gamma, dt)"
        },
        # --- Boundary case ---
        {
            "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
forces = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
gamma = 0.5
dt = 0.01""",
            "call": "update_vertex_positions(vertices, forces, gamma, dt)",
            "gold_call": "_oracle_update_vertex_positions(vertices, forces, gamma, dt)"
        },
        # --- Edge case ---
        {
            "setup": """import numpy as np
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
forces = np.array([[100.0, -100.0], [50.0, 50.0], [-50.0, -50.0]])
gamma = 0.1
dt = 1.0""",
            "call": "update_vertex_positions(vertices, forces, gamma, dt)",
            "gold_call": "_oracle_update_vertex_positions(vertices, forces, gamma, dt)"
        }
    ]
