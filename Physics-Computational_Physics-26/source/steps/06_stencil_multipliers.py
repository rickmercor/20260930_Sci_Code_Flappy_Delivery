"""
Return the per-direction integer by which the reconstruction stencil is widened at each cell. For each coordinate direction take twice the absolute value of the product of that direction's normal component with the signed distance, divide by that direction's mesh spacing, round up to the next integer, and take the larger of that and one. The multiplier is therefore one wherever the normal component vanishes or the cell sits close to the boundary.

A ghost cell lying deeper inside the solid would otherwise sit further from its boundary point than the reconstruction tolerates, so the Cartesian stencil is widened in whole cells, independently in each coordinate direction.

Returns
-------
ndarray of shape (3,) + psi.shape, float64: entries 0, 1 and 2 are the multipliers in the x, y and z directions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stencil_multipliers(psi: "np.ndarray", normals: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    """Return the per-direction integer by which the reconstruction stencil is widened at each cell. For each coordinate direction take twice the absolute value of the product of that direction's normal component with the signed distance, divide by that direction's mesh spacing, round up to the next integer, and take the larger of that and one. The multiplier is therefore one wherever the normal component vanishes or the cell sits close to the boundary.

    Returns
    -------
    ndarray of shape (3,) + psi.shape, float64: entries 0, 1 and 2 are the multipliers in the x, y and z directions.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_stencil_multipliers(psi: "np.ndarray", normals: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    ps = np.asarray(psi, dtype=float)
    n = np.asarray(normals, dtype=float)
    if n.shape[0] != 3 or n.shape[1:] != ps.shape:
        raise ValueError("normals must have shape (3,) + psi.shape")
    h = (float(dx), float(dy), float(dz))
    if not all(v > 0.0 for v in h):
        raise ValueError("dx, dy and dz must be strictly positive")
    # Eq 18, per direction, floored at one cell.
    return np.stack([np.maximum(1.0, np.ceil(2.0 * np.abs(n[a] * ps) / h[a])) for a in range(3)],
                    axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')\npsi_c=signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\npsi_o=_oracle_signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\nn_c=boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nn_o=_oracle_boundary_normal(X,Y,Z,0.07,-0.11,0.05)",
            "call": 'stencil_multipliers(psi_c.copy(), n_c.copy(), 0.1, 0.08, 0.06)',
            "gold_call": '_oracle_stencil_multipliers(psi_o.copy(), n_o.copy(), 0.1, 0.08, 0.06)',
        },
        {
            "setup": 'import numpy as np\npsi=np.zeros((1,1,1))\nn=np.stack([np.full((1,1,1),0.6),np.full((1,1,1),0.0),np.full((1,1,1),0.8)])',
            "call": 'stencil_multipliers(psi.copy(), n.copy(), 0.1, 0.1, 0.1)',
            "gold_call": '_oracle_stencil_multipliers(psi.copy(), n.copy(), 0.1, 0.1, 0.1)',
        },
        {
            "setup": 'import numpy as np\npsi=np.full((1,1,1),0.37)\nn=np.stack([np.zeros((1,1,1)),np.zeros((1,1,1)),np.ones((1,1,1))])',
            "call": 'stencil_multipliers(psi.copy(), n.copy(), 0.1, 0.1, 0.1)',
            "gold_call": '_oracle_stencil_multipliers(psi.copy(), n.copy(), 0.1, 0.1, 0.1)',
        },
        {
            "setup": 'import numpy as np\n# 2|n psi|/dx is exactly 1.0 here, so the ceiling must not round it up.\npsi=np.array([[[0.25]]])\nn=np.stack([np.ones((1,1,1)),np.zeros((1,1,1)),np.zeros((1,1,1))])',
            "call": 'stencil_multipliers(psi.copy(), n.copy(), 0.5, 0.5, 0.5)',
            "gold_call": '_oracle_stencil_multipliers(psi.copy(), n.copy(), 0.5, 0.5, 0.5)',
        },
    ]
