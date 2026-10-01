"""
Tag every cell as fluid, solid, or a hybrid ghost cell. A cell whose signed distance is positive is a ghost candidate when that distance is at most one and a half times its thickness, which keeps a second layer of them on the solid side; a cell whose signed distance is negative is a ghost candidate when the distance measured into the fluid is strictly less than half its thickness. Then make one follow-up pass over the fluid-side candidates only: a fluid-side candidate with no solid cell among its six Cartesian neighbours becomes plain fluid. Solid-side candidates are all retained, because the deeper threshold admitted them deliberately. Both the candidacy tests and this follow-up pass classify neighbours by the sign of the signed distance, never by a tag this same call has already written. Cells at an array face have their off-grid neighbours replaced by the face cell itself.

Hybrid ghost cells may sit on either side of the embedded boundary, unlike the classical variety which is confined to the solid. Which side a ghost centre lies on is recorded in the sign of its tag.

Returns
-------
ndarray with the shape of psi, float64: 0.0 fluid, 1.0 solid, 2.0 a ghost whose centre lies in the solid, -2.0 a ghost whose centre lies in the fluid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hybrid_ghost_tags(psi: "np.ndarray", thickness: "np.ndarray") -> "np.ndarray":
    """Tag every cell as fluid, solid, or a hybrid ghost cell. A cell whose signed distance is positive is a ghost candidate when that distance is at most one and a half times its thickness, which keeps a second layer of them on the solid side; a cell whose signed distance is negative is a ghost candidate when the distance measured into the fluid is strictly less than half its thickness. Then make one follow-up pass over the fluid-side candidates only: a fluid-side candidate with no solid cell among its six Cartesian neighbours becomes plain fluid. Solid-side candidates are all retained, because the deeper threshold admitted them deliberately. Both the candidacy tests and this follow-up pass classify neighbours by the sign of the signed distance, never by a tag this same call has already written. Cells at an array face have their off-grid neighbours replaced by the face cell itself.

    Returns
    -------
    ndarray with the shape of psi, float64: 0.0 fluid, 1.0 solid, 2.0 a ghost whose centre lies in the solid, -2.0 a ghost whose centre lies in the fluid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _shift(a, axis, step):
    """Translate by one cell along `axis`, repeating the edge value rather than wrapping."""
    out = np.roll(a, step, axis=axis)
    idx = [slice(None)] * a.ndim
    idx[axis] = 0 if step > 0 else -1
    out[tuple(idx)] = a[tuple(idx)]
    return out


def _oracle_hybrid_ghost_tags(psi: "np.ndarray", thickness: "np.ndarray") -> "np.ndarray":
    psi = np.asarray(psi, dtype=float)
    th = np.asarray(thickness, dtype=float)
    if psi.shape != th.shape:
        raise ValueError("psi and thickness must have the same shape")
    if np.any(th <= 0.0):
        raise ValueError("thickness must be strictly positive everywhere")
    solid = psi > 0.0
    half = th / 2.0
    # Eq 7, with the Sec 7.1 deeper threshold (1.5 Delta n = 3 * half) on the solid side so
    # that a second layer of ghost cells is retained, and the shallower half-thickness on
    # the fluid side.  The solid test is non-strict and the fluid test is strict.
    cand_solid = solid & (psi <= 3.0 * half)
    cand_fluid = (~solid) & (-psi < half)
    tags = np.where(solid, 1.0, 0.0)
    tags = np.where(cand_solid, 2.0, tags)
    tags = np.where(cand_fluid, -2.0, tags)
    # Sec 3.1 reclassification, FLUID side only.  Sec 7.1 states that the qualifying solid
    # cells are included in the ghost-cell set, and the solid-side clause exists to drop
    # solid ghosts the fluid stencil does not need; a configuration that asks for a second
    # layer needs them by construction.  The source does not settle the interaction, so the
    # prompt states which reading this configuration uses.  Neighbours are judged against
    # the ORIGINAL solid/fluid split, never against a tag this same call has written.
    has_solid = np.zeros_like(psi, dtype=bool)
    for ax in range(psi.ndim):
        for k in (1, -1):
            has_solid |= _shift(solid, ax, k)
    tags = np.where((tags == -2.0) & (~has_solid), 0.0, tags)
    return tags

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nz=np.linspace(-0.9,0.9,31); y=np.linspace(-1.2,1.2,31); x=np.linspace(-1.5,1.5,31)\nZ,Y,X=np.meshgrid(z,y,x,indexing='ij')\npsi_c=signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\npsi_o=_oracle_signed_normal_distance(X,Y,Z,0.07,-0.11,0.05,0.61)\nn_c=boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nn_o=_oracle_boundary_normal(X,Y,Z,0.07,-0.11,0.05)\nth_c=normal_cell_thickness(n_c,0.1,0.08,0.06)\nth_o=_oracle_normal_cell_thickness(n_o,0.1,0.08,0.06)",
            "call": 'hybrid_ghost_tags(psi_c.copy(), th_c.copy())',
            "gold_call": '_oracle_hybrid_ghost_tags(psi_o.copy(), th_o.copy())',
        },
        {
            "setup": 'import numpy as np\npsi=np.full((2,2,2),5.0); th=np.full((2,2,2),0.1)',
            "call": 'hybrid_ghost_tags(psi.copy(), th.copy())',
            "gold_call": '_oracle_hybrid_ghost_tags(psi.copy(), th.copy())',
        },
        {
            "setup": 'import numpy as np\npsi=np.full((3,3,3),-1.0); psi[1,1,1]=-0.01\nth=np.full((3,3,3),0.1)',
            "call": 'hybrid_ghost_tags(psi.copy(), th.copy())',
            "gold_call": '_oracle_hybrid_ghost_tags(psi.copy(), th.copy())',
        },
        {
            "setup": 'import numpy as np\npsi=np.full((3,3,3),1.0); psi[1,1,1]=0.01\nth=np.full((3,3,3),0.1)',
            "call": 'hybrid_ghost_tags(psi.copy(), th.copy())',
            "gold_call": '_oracle_hybrid_ghost_tags(psi.copy(), th.copy())',
        },
        {
            "setup": 'import numpy as np\npsi=np.tile(np.array([0.25,0.15,0.05,-0.02,-0.12]),(3,3,1))\nth=np.full((3,3,5),0.1)',
            "call": 'hybrid_ghost_tags(psi.copy(), th.copy())',
            "gold_call": '_oracle_hybrid_ghost_tags(psi.copy(), th.copy())',
        },
        {
            "setup": 'import numpy as np\npsi=np.array([[[0.02,-0.02],[-0.02,0.02]],[[-0.02,0.02],[0.02,-0.02]]])\nth=np.full((2,2,2),0.1)',
            "call": 'hybrid_ghost_tags(psi.copy(), th.copy())',
            "gold_call": '_oracle_hybrid_ghost_tags(psi.copy(), th.copy())',
        },
        {
            "setup": 'import numpy as np\n# Eq 7 strictness: the solid test is non-strict, the fluid test is strict.\n# psi = 1.5*th exactly on the solid side, and -psi = 0.5*th exactly on the fluid side.\nth=np.full((1,1,2),0.2)\npsi=np.array([[[0.30,-0.10]]])',
            "call": 'hybrid_ghost_tags(psi.copy(), th.copy())',
            "gold_call": '_oracle_hybrid_ghost_tags(psi.copy(), th.copy())',
        },
        {
            "setup": 'import numpy as np\n# the follow-up pass classifies a neighbour by the SIGN OF THE SIGNED DISTANCE,\n# so a neighbour retained as a solid-side ghost still counts as solid.\nth=np.full((1,1,3),0.2)\npsi=np.array([[[0.28,-0.02,-1.0]]])',
            "call": 'hybrid_ghost_tags(psi.copy(), th.copy())',
            "gold_call": '_oracle_hybrid_ghost_tags(psi.copy(), th.copy())',
        },
    ]
