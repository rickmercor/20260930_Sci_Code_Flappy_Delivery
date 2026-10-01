"""
Return the point where the boundary normal through each cell centre meets the spherical embedded boundary. This is the location at which the boundary condition is imposed.

For a sphere this intersection is the centre plus the radius times the outward unit normal, so no iteration is needed. The three coordinate arrays hold one value per cell and already share the grid's shape; they are not one-dimensional axis vectors and must not be meshed again.

Returns
-------
ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 hold the x, y and z coordinates of the intersection.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def boundary_point(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float, radius: float) -> "np.ndarray":
    """Return the point where the boundary normal through each cell centre meets the spherical embedded boundary. This is the location at which the boundary condition is imposed.

    Returns
    -------
    ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 hold the x, y and z coordinates of the intersection.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_boundary_point(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float, radius: float) -> "np.ndarray":
    n = _oracle_boundary_normal(xc, yc, zc, centre_x, centre_y, centre_z)
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be strictly positive")
    return np.stack([float(centre_x) + radius * n[0],
                     float(centre_y) + radius * n[1],
                     float(centre_z) + radius * n[2]], axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')",
            "call": 'boundary_point(X.copy(), Y.copy(), Z.copy(), 0.07, -0.11, 0.05, 0.61)',
            "gold_call": '_oracle_boundary_point(X.copy(), Y.copy(), Z.copy(), 0.07, -0.11, 0.05, 0.61)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([0.0]),np.array([0.0]),np.array([2.0]),indexing='ij')",
            "call": 'boundary_point(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0, 0.5)',
            "gold_call": '_oracle_boundary_point(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0, 0.5)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([0.2]),np.array([0.3]),np.array([0.4]),indexing='ij')",
            "call": 'boundary_point(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0, 0.9)',
            "gold_call": '_oracle_boundary_point(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0, 0.9)',
        },
    ]
