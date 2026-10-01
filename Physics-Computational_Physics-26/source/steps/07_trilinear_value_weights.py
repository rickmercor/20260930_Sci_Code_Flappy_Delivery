"""
Return the three interpolation weights of the closed-form ghost-cell reconstruction that applies where the field value itself is prescribed at the boundary. The diagonally opposite stencil point sits one multiplier's worth of cells away in each direction, on the side the matching normal component points to, counting a vanishing component as the positive side. Each weight is then the ghost-cell coordinate minus the boundary-point coordinate, divided by the opposite stencil point's coordinate minus the boundary-point coordinate, taken in x for the first weight, in y for the second and in z for the third.

The reconstruction assumes a trilinear solution through seven Cartesian neighbours and the boundary point, so the weights follow from the geometry alone and no linear system is solved. The three weights are not interchangeable: each is built from one coordinate direction. The three coordinate arrays hold one value per cell and already share the grid's shape; they are not one-dimensional axis vectors and must not be meshed again.

Returns
-------
ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 are the weights built from the x, y and z coordinates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trilinear_value_weights(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", boundary: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    """Return the three interpolation weights of the closed-form ghost-cell reconstruction that applies where the field value itself is prescribed at the boundary. The diagonally opposite stencil point sits one multiplier's worth of cells away in each direction, on the side the matching normal component points to, counting a vanishing component as the positive side. Each weight is then the ghost-cell coordinate minus the boundary-point coordinate, divided by the opposite stencil point's coordinate minus the boundary-point coordinate, taken in x for the first weight, in y for the second and in z for the third.

    Returns
    -------
    ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 are the weights built from the x, y and z coordinates.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_trilinear_value_weights(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", boundary: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    c = [np.asarray(v, dtype=float) for v in (xc, yc, zc)]
    b = np.asarray(boundary, dtype=float)
    n = np.asarray(normals, dtype=float)
    r = np.asarray(mult, dtype=float)
    if b.shape[0] != 3 or n.shape[0] != 3 or r.shape[0] != 3:
        raise ValueError("boundary, normals and mult must have leading dimension 3")
    h = (float(dx), float(dy), float(dz))
    if not all(v > 0.0 for v in h):
        raise ValueError("dx, dy and dz must be strictly positive")
    out = []
    for a in range(3):
        # the diagonally opposite stencil point: sign of the normal component, +1 when it
        # vanishes, times that direction's multiplier in whole cells
        far = c[a] + np.where(n[a] < 0.0, -1.0, 1.0) * r[a] * h[a]
        den = far - b[a]
        if np.any(den == 0.0):
            raise ValueError("degenerate stencil: the boundary point coincides with a stencil coordinate")
        out.append((c[a] - b[a]) / den)
    return np.stack(out, axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')\npsi_c=signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\npsi_o=_oracle_signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\nn_c=boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nn_o=_oracle_boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nb_c=boundary_point(X,Y,Z,0.07,-0.11,0.05,0.61)\nb_o=_oracle_boundary_point(X,Y,Z,0.07,-0.11,0.05,0.61)\nr_c=stencil_multipliers(psi_c,n_c,0.1,0.08,0.06)\nr_o=_oracle_stencil_multipliers(psi_o,n_o,0.1,0.08,0.06)",
            "call": 'trilinear_value_weights(X.copy(), Y.copy(), Z.copy(), b_c.copy(), n_c.copy(), r_c.copy(), 0.1, 0.08, 0.06)',
            "gold_call": '_oracle_trilinear_value_weights(X.copy(), Y.copy(), Z.copy(), b_o.copy(), n_o.copy(), r_o.copy(), 0.1, 0.08, 0.06)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([0.21]),np.array([0.3]),np.array([0.4]),indexing='ij')\nn_c=boundary_normal(X,Y,Z,0.0,0.0,0.0)\nn_o=_oracle_boundary_normal(X,Y,Z,0.0,0.0,0.0)\nb_c=boundary_point(X,Y,Z,0.0,0.0,0.0,0.2)\nb_o=_oracle_boundary_point(X,Y,Z,0.0,0.0,0.0,0.2)\nr=np.ones((3,)+X.shape)",
            "call": 'trilinear_value_weights(X.copy(), Y.copy(), Z.copy(), b_c.copy(), n_c.copy(), r.copy(), 0.1, 0.1, 0.1)',
            "gold_call": '_oracle_trilinear_value_weights(X.copy(), Y.copy(), Z.copy(), b_o.copy(), n_o.copy(), r.copy(), 0.1, 0.1, 0.1)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([0.0]),np.array([0.0]),np.array([0.48]),indexing='ij')\nn_c=boundary_normal(X,Y,Z,0.0,0.0,0.0)\nn_o=_oracle_boundary_normal(X,Y,Z,0.0,0.0,0.0)\nb_c=boundary_point(X,Y,Z,0.0,0.0,0.0,0.5)\nb_o=_oracle_boundary_point(X,Y,Z,0.0,0.0,0.0,0.5)\nr=np.ones((3,)+X.shape)",
            "call": 'trilinear_value_weights(X.copy(), Y.copy(), Z.copy(), b_c.copy(), n_c.copy(), r.copy(), 0.1, 0.1, 0.1)',
            "gold_call": '_oracle_trilinear_value_weights(X.copy(), Y.copy(), Z.copy(), b_o.copy(), n_o.copy(), r.copy(), 0.1, 0.1, 0.1)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([-0.44]),np.array([0.05]),np.array([0.03]),indexing='ij')\nn_c=boundary_normal(X,Y,Z,0.0,0.0,0.0)\nn_o=_oracle_boundary_normal(X,Y,Z,0.0,0.0,0.0)\nb_c=boundary_point(X,Y,Z,0.0,0.0,0.0,0.5)\nb_o=_oracle_boundary_point(X,Y,Z,0.0,0.0,0.0,0.5)\nr=np.stack([np.full(X.shape,2.0),np.ones(X.shape),np.full(X.shape,3.0)])",
            "call": 'trilinear_value_weights(X.copy(), Y.copy(), Z.copy(), b_c.copy(), n_c.copy(), r.copy(), 0.2, 0.05, 0.1)',
            "gold_call": '_oracle_trilinear_value_weights(X.copy(), Y.copy(), Z.copy(), b_o.copy(), n_o.copy(), r.copy(), 0.2, 0.05, 0.1)',
        },
    ]
