"""
Return the signed distance from each cell centre to the spherical embedded boundary, measured along the boundary normal through that centre. The sign convention is positive on the solid side and negative on the fluid side. The solid is the ball interior and the fluid is everything outside it.

For a sphere the normal through a point is radial, so the distance along it to the boundary is the radius minus the radial distance and needs no search. Arrays are indexed [k, j, i] with k running along z, j along y and i along x. The three coordinate arrays hold one value per cell and already share the grid's shape; they are not one-dimensional axis vectors and must not be meshed again.

Returns
-------
ndarray with the same shape as xc, float64: the signed normal distance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def signed_normal_distance(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float, radius: float) -> "np.ndarray":
    """Return the signed distance from each cell centre to the spherical embedded boundary, measured along the boundary normal through that centre. The sign convention is positive on the solid side and negative on the fluid side. The solid is the ball interior and the fluid is everything outside it.

    Returns
    -------
    ndarray with the same shape as xc, float64: the signed normal distance.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_signed_normal_distance(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float, radius: float) -> "np.ndarray":
    xc = np.asarray(xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    zc = np.asarray(zc, dtype=float)
    if not (xc.shape == yc.shape == zc.shape):
        raise ValueError("xc, yc and zc must have the same shape")
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be strictly positive")
    r = np.sqrt((xc - float(centre_x)) ** 2 + (yc - float(centre_y)) ** 2
                + (zc - float(centre_z)) ** 2)
    return radius - r

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')",
            "call": 'signed_normal_distance(X.copy(), Y.copy(), Z.copy(), 0.07, -0.11, 0.05, 0.61)',
            "gold_call": '_oracle_signed_normal_distance(X.copy(), Y.copy(), Z.copy(), 0.07, -0.11, 0.05, 0.61)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([0.0,2.0]),np.array([0.0,1.0]),np.array([0.0,1.0]),indexing='ij')",
            "call": 'signed_normal_distance(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0, 1.0)',
            "gold_call": '_oracle_signed_normal_distance(X.copy(), Y.copy(), Z.copy(), 0.0, 0.0, 0.0, 1.0)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.linspace(-3,3,5),np.linspace(-3,3,5),np.linspace(-3,3,5),indexing='ij')",
            "call": 'signed_normal_distance(X.copy(), Y.copy(), Z.copy(), 2.9, -2.9, 2.9, 0.05)',
            "gold_call": '_oracle_signed_normal_distance(X.copy(), Y.copy(), Z.copy(), 2.9, -2.9, 2.9, 0.05)',
        },
    ]
