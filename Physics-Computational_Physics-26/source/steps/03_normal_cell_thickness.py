"""
Return the characteristic cell thickness measured in the boundary-normal direction. Divide each component of the unit normal by the mesh spacing in that same coordinate direction, take the Euclidean norm of the resulting triple, and return its reciprocal. It equals the spacing exactly when the normal is aligned with that axis and lies between the smallest and largest spacing otherwise, so it is not simply the smallest of the three.

On an anisotropic mesh the distance a cell spans along the boundary normal depends on the normal's orientation relative to the three spacings. Using a single spacing instead changes which cells are tagged.

Returns
-------
ndarray with the shape of one normal component, float64: the thickness.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normal_cell_thickness(normals: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    """Return the characteristic cell thickness measured in the boundary-normal direction. Divide each component of the unit normal by the mesh spacing in that same coordinate direction, take the Euclidean norm of the resulting triple, and return its reciprocal. It equals the spacing exactly when the normal is aligned with that axis and lies between the smallest and largest spacing otherwise, so it is not simply the smallest of the three.

    Returns
    -------
    ndarray with the shape of one normal component, float64: the thickness.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_normal_cell_thickness(normals: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    n = np.asarray(normals, dtype=float)
    if n.shape[0] != 3:
        raise ValueError("normals must have shape (3,) + grid shape")
    dx = float(dx); dy = float(dy); dz = float(dz)
    if not (dx > 0.0 and dy > 0.0 and dz > 0.0):
        raise ValueError("dx, dy and dz must be strictly positive")
    q = (n[0] / dx) ** 2 + (n[1] / dy) ** 2 + (n[2] / dz) ** 2
    if np.any(q <= 0.0):
        raise ValueError("a normal vector is degenerate, so the thickness is undefined")
    return q ** (-0.5)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nth=np.linspace(0,np.pi,9)[1:-1]; ph=np.linspace(0,2*np.pi,9)[:-1]\nT,Pp=np.meshgrid(th,ph,indexing='ij')\nn=np.stack([np.sin(T)*np.cos(Pp),np.sin(T)*np.sin(Pp),np.cos(T)])",
            "call": 'normal_cell_thickness(n.copy(), 0.1, 0.08, 0.06)',
            "gold_call": '_oracle_normal_cell_thickness(n.copy(), 0.1, 0.08, 0.06)',
        },
        {
            "setup": 'import numpy as np\nn=np.stack([np.ones((1,1)),np.zeros((1,1)),np.zeros((1,1))])',
            "call": 'normal_cell_thickness(n.copy(), 0.1, 0.08, 0.06)',
            "gold_call": '_oracle_normal_cell_thickness(n.copy(), 0.1, 0.08, 0.06)',
        },
        {
            "setup": 'import numpy as np\nn=np.stack([np.zeros((1,1)),np.zeros((1,1)),np.ones((1,1))])',
            "call": 'normal_cell_thickness(n.copy(), 0.1, 0.08, 0.06)',
            "gold_call": '_oracle_normal_cell_thickness(n.copy(), 0.1, 0.08, 0.06)',
        },
    ]
