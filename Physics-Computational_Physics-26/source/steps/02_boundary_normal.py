"""
Return the unit normal to the spherical embedded boundary at each cell centre, oriented to point out of the solid and into the fluid region.

The normal is radial for a sphere. Orientation matters downstream because the reconstruction stencil is selected from the signs of the normal components. The three coordinate arrays hold one value per cell and already share the grid's shape; they are not one-dimensional axis vectors and must not be meshed again.

Returns
-------
ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 are the x, y and z components of the unit normal.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def boundary_normal(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float) -> "np.ndarray":
    """Return the unit normal to the spherical embedded boundary at each cell centre, oriented to point out of the solid and into the fluid region.

    Returns
    -------
    ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 are the x, y and z components of the unit normal.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_boundary_normal(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float) -> "np.ndarray":
    xc = np.asarray(xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    zc = np.asarray(zc, dtype=float)
    if not (xc.shape == yc.shape == zc.shape):
        raise ValueError("xc, yc and zc must have the same shape")
    ux = xc - float(centre_x)
    uy = yc - float(centre_y)
    uz = zc - float(centre_z)
    r = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    if np.any(r == 0.0):
        raise ValueError("a cell centre coincides with the sphere centre, so the normal is undefined")
    return np.stack([ux / r, uy / r, uz / r], axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')",
            "call": 'boundary_normal(X.copy(), Y.copy(), Z.copy(), 0.07, -0.11, 0.05)',
            "gold_call": '_oracle_boundary_normal(X.copy(), Y.copy(), Z.copy(), 0.07, -0.11, 0.05)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([0.0]),np.array([0.0]),np.array([-1.0,1.0]),indexing='ij')",
            "call": 'boundary_normal(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0)',
            "gold_call": '_oracle_boundary_normal(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([-2.0,2.0]),np.array([0.0]),np.array([0.0]),indexing='ij')",
            "call": 'boundary_normal(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0)',
            "gold_call": '_oracle_boundary_normal(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0)',
        },
    ]
