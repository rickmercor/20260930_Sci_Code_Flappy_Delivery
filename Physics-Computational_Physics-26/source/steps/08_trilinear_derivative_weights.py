"""
Return the eight weights of the closed-form ghost-cell reconstruction that applies where the normal derivative is prescribed at the boundary: seven on the Cartesian neighbours, in the P1 to P7 order fixed below, and one on that prescribed derivative. The weights are defined by the interpolation problem itself. Over the stencil formed by the ghost cell and its seven Cartesian neighbours there is a unique function of the form C0 + C1 x + C2 y + C3 z + C4 xy + C5 yz + C6 xz + C7 xyz whose values at the seven neighbours are the seven neighbour values and whose gradient dotted with the outward unit normal equals the prescribed derivative at the boundary point. That function's value at the ghost cell depends linearly on those eight data; return the eight coefficients of that dependence, the seven neighbour coefficients first and the coefficient of the prescribed derivative last. The opposite stencil point sits one multiplier's worth of cells away in each direction, on the side the matching normal component points to, counting a vanishing component as the positive side, which is what keeps the problem non-degenerate. Writing i2, j2 and k2 for that point's indices in x, y and z, the seven neighbours in order are the points differing from the ghost cell in x alone, in y alone, in z alone, in x and y, in y and z, in x and z, and in all three.

The derivative-prescribed reconstruction differs from the value-prescribed one in what is known at the boundary. Because the boundary datum constrains a directional derivative rather than a value, all seven neighbours carry independent coefficients instead of the products the value-prescribed case produces, and an eighth coefficient multiplies the prescribed derivative itself. The three coordinate arrays hold one value per cell and already share the grid's shape; they are not one-dimensional axis vectors and must not be meshed again.

Returns
-------
ndarray of shape (8,) + xc.shape, float64: entries 0 to 6 are the seven neighbour weights in the order the description lists them, matching stencil points P1 to P7, and entry 7 is the weight on the prescribed normal derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trilinear_derivative_weights(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", boundary: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    """Return the eight weights of the closed-form ghost-cell reconstruction that applies where the normal derivative is prescribed at the boundary: seven on the Cartesian neighbours, in the P1 to P7 order fixed below, and one on that prescribed derivative. The weights are defined by the interpolation problem itself. Over the stencil formed by the ghost cell and its seven Cartesian neighbours there is a unique function of the form C0 + C1 x + C2 y + C3 z + C4 xy + C5 yz + C6 xz + C7 xyz whose values at the seven neighbours are the seven neighbour values and whose gradient dotted with the outward unit normal equals the prescribed derivative at the boundary point. That function's value at the ghost cell depends linearly on those eight data; return the eight coefficients of that dependence, the seven neighbour coefficients first and the coefficient of the prescribed derivative last. The opposite stencil point sits one multiplier's worth of cells away in each direction, on the side the matching normal component points to, counting a vanishing component as the positive side, which is what keeps the problem non-degenerate. Writing i2, j2 and k2 for that point's indices in x, y and z, the seven neighbours in order are the points differing from the ghost cell in x alone, in y alone, in z alone, in x and y, in y and z, in x and z, and in all three.

    Returns
    -------
    ndarray of shape (8,) + xc.shape, float64: entries 0 to 6 are the seven neighbour weights in the order the description lists them, matching stencil points P1 to P7, and entry 7 is the weight on the prescribed normal derivative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_trilinear_derivative_weights(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", boundary: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    c = [np.asarray(v, dtype=float) for v in (xc, yc, zc)]
    b = np.asarray(boundary, dtype=float)
    n = np.asarray(normals, dtype=float)
    r = np.asarray(mult, dtype=float)
    if b.shape[0] != 3 or n.shape[0] != 3 or r.shape[0] != 3:
        raise ValueError("boundary, normals and mult must have leading dimension 3")
    h = (float(dx), float(dy), float(dz))
    if not all(v > 0.0 for v in h):
        raise ValueError("dx, dy and dz must be strictly positive")
    far = [c[a] + np.where(n[a] < 0.0, -1.0, 1.0) * r[a] * h[a] for a in range(3)]
    # X[1], Y[1], Z[1] are the ghost cell's own coordinates; X[2], Y[2], Z[2] the opposite
    # stencil point's.  Offsets are measured from the boundary point.
    Xo = {1: c[0] - b[0], 2: far[0] - b[0]}
    Yo = {1: c[1] - b[1], 2: far[1] - b[1]}
    Zo = {1: c[2] - b[2], 2: far[2] - b[2]}
    a_ = {(u, v): n[0] * Yo[u] * Zo[v] for u in (1, 2) for v in (1, 2)}
    b_ = {(u, v): n[1] * Xo[u] * Zo[v] for u in (1, 2) for v in (1, 2)}
    c_ = {(u, v): n[2] * Xo[u] * Yo[v] for u in (1, 2) for v in (1, 2)}
    den = a_[2, 2] + b_[2, 2] + c_[2, 2]
    if np.any(den == 0.0):
        raise ValueError("degenerate stencil: the derivative-type denominator vanishes")
    w1 = (a_[2, 2] + b_[1, 2] + c_[1, 2]) / den
    w2 = (a_[1, 2] + b_[2, 2] + c_[2, 1]) / den
    w3 = (a_[2, 1] + b_[2, 1] + c_[2, 2]) / den
    w4 = -(a_[1, 2] + b_[1, 2] + c_[1, 1]) / den
    w5 = -(a_[1, 1] + b_[2, 1] + c_[2, 1]) / den
    w6 = -(a_[2, 1] + b_[1, 1] + c_[1, 2]) / den
    w7 = (a_[1, 1] + b_[1, 1] + c_[1, 1]) / den
    ws = -((far[0] - c[0]) * (far[1] - c[1]) * (far[2] - c[2])) / den
    return np.stack([w1, w2, w3, w4, w5, w6, w7, ws], axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')\npsi_c=signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\npsi_o=_oracle_signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\nn_c=boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nn_o=_oracle_boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nb_c=boundary_point(X,Y,Z,0.07,-0.11,0.05,0.61)\nb_o=_oracle_boundary_point(X,Y,Z,0.07,-0.11,0.05,0.61)\nr_c=stencil_multipliers(psi_c,n_c,0.1,0.08,0.06)\nr_o=_oracle_stencil_multipliers(psi_o,n_o,0.1,0.08,0.06)",
            "call": 'trilinear_derivative_weights(X.copy(), Y.copy(), Z.copy(), b_c.copy(), n_c.copy(), r_c.copy(), 0.1, 0.08, 0.06)',
            "gold_call": '_oracle_trilinear_derivative_weights(X.copy(), Y.copy(), Z.copy(), b_o.copy(), n_o.copy(), r_o.copy(), 0.1, 0.08, 0.06)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([0.0]),np.array([0.0]),np.array([0.9]),indexing='ij')\nn_c=boundary_normal(X,Y,Z,0.0,0.0,0.0)\nn_o=_oracle_boundary_normal(X,Y,Z,0.0,0.0,0.0)\nb_c=boundary_point(X,Y,Z,0.0,0.0,0.0,0.5)\nb_o=_oracle_boundary_point(X,Y,Z,0.0,0.0,0.0,0.5)\nr=np.ones((3,)+X.shape)",
            "call": 'trilinear_derivative_weights(X.copy(), Y.copy(), Z.copy(), b_c.copy(), n_c.copy(), r.copy(), 0.1, 0.1, 0.1)',
            "gold_call": '_oracle_trilinear_derivative_weights(X.copy(), Y.copy(), Z.copy(), b_o.copy(), n_o.copy(), r.copy(), 0.1, 0.1, 0.1)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([0.9]),np.array([0.0]),np.array([0.0]),indexing='ij')\nn_c=boundary_normal(X,Y,Z,0.0,0.0,0.0)\nn_o=_oracle_boundary_normal(X,Y,Z,0.0,0.0,0.0)\nb_c=boundary_point(X,Y,Z,0.0,0.0,0.0,0.5)\nb_o=_oracle_boundary_point(X,Y,Z,0.0,0.0,0.0,0.5)\nr=np.ones((3,)+X.shape)",
            "call": 'trilinear_derivative_weights(X.copy(), Y.copy(), Z.copy(), b_c.copy(), n_c.copy(), r.copy(), 0.1, 0.1, 0.1)',
            "gold_call": '_oracle_trilinear_derivative_weights(X.copy(), Y.copy(), Z.copy(), b_o.copy(), n_o.copy(), r.copy(), 0.1, 0.1, 0.1)',
        },
        {
            "setup": "import numpy as np\nZ,Y,X=np.meshgrid(np.array([-0.31]),np.array([0.22]),np.array([0.17]),indexing='ij')\nn_c=boundary_normal(X,Y,Z,0.0,0.0,0.0)\nn_o=_oracle_boundary_normal(X,Y,Z,0.0,0.0,0.0)\nb_c=boundary_point(X,Y,Z,0.0,0.0,0.0,0.6)\nb_o=_oracle_boundary_point(X,Y,Z,0.0,0.0,0.0,0.6)\nr=np.stack([np.full(X.shape,3.0),np.full(X.shape,2.0),np.ones(X.shape)])",
            "call": 'trilinear_derivative_weights(X.copy(), Y.copy(), Z.copy(), b_c.copy(), n_c.copy(), r.copy(), 0.12, 0.07, 0.09)',
            "gold_call": '_oracle_trilinear_derivative_weights(X.copy(), Y.copy(), Z.copy(), b_o.copy(), n_o.copy(), r.copy(), 0.12, 0.07, 0.09)',
        },
    ]
